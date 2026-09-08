#!/usr/bin/env python3
"""
detection_check.py
------------------
Answers one question: does the model's peak score tell you WHETHER the word was
spoken, or only WHERE?

The model was trained and measured on localization. Every recording contains the
keyword exactly once, so the score only ever had to be higher at the right
moment than elsewhere in the same recording - a within-recording ranking. It was
never asked to be comparable across recordings, and that is exactly what a
detector needs.

Testing it needs recordings without the keyword. There are only 14 of those, but
each of the 355 positive recordings also contains about 19.6 seconds of ordinary
book reading. That material is the negative class, and it is already on disk.

Two ways of using it, run side by side:

  MASKED   Score the full recording once, then take the highest score OUTSIDE a
           guard band around the annotated word. The guard is +/- 0.7 s, longer
           than half the network's 1.26 s receptive field, so no frame counted as
           negative can see any part of the word. No audio is altered.

  EXCISED  Physically cut the word out with a short crossfade and re-run the
           model on the shortened recording. More faithful to a genuinely
           keyword-free take, at the cost of a splice artefact at the join.

If the positive and negative peaks separate cleanly, a threshold is all that is
missing and the recording effort is only about calibration. If they overlap, the
score measures "where" and not "whether", and a detector needs to be built and
trained differently.

Each fold's checkpoint is evaluated only on ITS test speaker, so every number
here comes from a speaker that model never saw.

Usage:
    python detection_check.py --models outputs/help_v5 --laser-dir ./laser \
        --meta metadata_clean.csv --plot detection_check.png
"""

import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from train_help_v5_mac import (SR, REC_SEC, OUT_FPS, AudioCache, Features,
                                   HelpLocalizerV5 as Net, get_device)
except ImportError:
    from train_help_v6_mac import (SR, REC_SEC, OUT_FPS, AudioCache, Features,
                                   HelpLocalizer as Net, get_device)

GUARD_S = 0.7        # exceeds half the 1.26 s receptive field
PEAK_PAD_S = 0.3     # slack when locating the positive peak
CROSSFADE_S = 0.02


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu")
    model = Net(ck["n_bins"], width=ck.get("width", 32))
    model.load_state_dict(ck["model_state_dict"])
    model.to(device).eval()
    return model


@torch.no_grad()
def curve(model, wav, feat, device):
    x = feat(torch.from_numpy(wav)).unsqueeze(0).to(device)
    logits = model(x).squeeze(0).float().cpu().numpy()
    t = (np.arange(len(logits)) + 0.5) / OUT_FPS
    return t, logits


def excise(wav, start_s, end_s, guard=GUARD_S):
    """Remove the word plus its guard band, crossfading over the join."""
    a = max(0, int((start_s - guard) * SR))
    b = min(len(wav), int((end_s + guard) * SR))
    if b <= a:
        return wav.copy()
    left, right = wav[:a], wav[b:]
    n = int(CROSSFADE_S * SR)
    if len(left) < n or len(right) < n:
        return np.concatenate([left, right])
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    join = left[-n:] * (1 - ramp) + right[:n] * ramp
    return np.concatenate([left[:-n], join, right[n:]])


def auc(pos, neg):
    """Rank-based AUC: the chance a random positive outscores a random negative."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def best_threshold(pos, neg):
    """Threshold maximising balanced accuracy, and the equal error rate."""
    cand = np.unique(np.concatenate([pos, neg]))
    best_t, best_acc = cand[0], 0.0
    for t in cand:
        tpr = (np.asarray(pos) >= t).mean()
        tnr = (np.asarray(neg) < t).mean()
        if 0.5 * (tpr + tnr) > best_acc:
            best_acc, best_t = 0.5 * (tpr + tnr), t
    eer, eer_t = 1.0, cand[0]
    for t in cand:
        fnr = (np.asarray(pos) < t).mean()
        fpr = (np.asarray(neg) >= t).mean()
        if abs(fnr - fpr) < eer:
            eer, eer_t = abs(fnr - fpr), t
    fnr = (np.asarray(pos) < eer_t).mean()
    fpr = (np.asarray(neg) >= eer_t).mean()
    return best_t, best_acc, 0.5 * (fnr + fpr)


def verdict(a):
    if not np.isfinite(a):
        return "not enough data"
    if a >= 0.90:
        return "clean separation - a threshold is all that is missing"
    if a >= 0.75:
        return "partial separation - a detector is plausible but needs real negatives"
    if a >= 0.60:
        return "weak separation - the score is mostly about 'where', not 'whether'"
    return "no separation - the peak score carries no evidence of presence"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="outputs/help_v5",
                    help="directory containing the per-fold checkpoints")
    ap.add_argument("--laser-dir", required=True)
    ap.add_argument("--meta", default="metadata_clean.csv")
    ap.add_argument("--out", default="detection_check.csv")
    ap.add_argument("--plot", default="detection_check.png")
    ap.add_argument("--no-excise", action="store_true",
                    help="skip the slower physically-excised variant")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")

    meta = pd.read_csv(args.meta, dtype={"recording_id": str, "speaker_id": str})
    meta = meta[~meta.exclude].reset_index(drop=True)

    ckpts = sorted(glob(os.path.join(args.models, "*", "best_help_v*.pt")))
    if not ckpts:
        raise SystemExit(f"no checkpoints under {args.models}")
    print(f"found {len(ckpts)} fold checkpoints")

    laser = AudioCache(args.laser_dir, meta.recording_id.tolist(), "laser")
    meta = meta[meta.recording_id.isin(laser.data.keys())].reset_index(drop=True)
    feat = Features()

    rows = []
    for ck in ckpts:
        fold = Path(ck).parent.name
        test_spk = fold.split("_")[1]
        sub = meta[meta.speaker_id == test_spk]
        if not len(sub):
            continue
        model = load_model(ck, device)
        print(f"  {fold}: scoring {len(sub)} recordings from unseen speaker {test_spk}")

        for row in sub.itertuples():
            wav = laser.get(row.recording_id)
            t, lg = curve(model, wav, feat, device)
            med = float(np.median(lg))

            r = {"recording_id": row.recording_id, "speaker_id": row.speaker_id,
                 "fold": fold, "has_keyword": bool(row.has_keyword),
                 "global_peak": float(lg.max()),
                 "global_peak_rel": float(lg.max() - med)}

            if row.has_keyword and np.isfinite(row.start_s):
                inside = (t >= row.start_s - PEAK_PAD_S) & (t <= row.end_s + PEAK_PAD_S)
                outside = (t < row.start_s - GUARD_S) | (t > row.end_s + GUARD_S)
                r["pos_peak"] = float(lg[inside].max()) if inside.any() else np.nan
                r["neg_peak_masked"] = float(lg[outside].max()) if outside.any() else np.nan
                r["pos_peak_rel"] = r["pos_peak"] - med
                r["neg_peak_masked_rel"] = r["neg_peak_masked"] - med

                if not args.no_excise:
                    cut = excise(wav, row.start_s, row.end_s)
                    _, lg2 = curve(model, cut, feat, device)
                    r["neg_peak_excised"] = float(lg2.max())
                    r["neg_peak_excised_rel"] = float(lg2.max() - np.median(lg2))
            rows.append(r)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}\n")

    kw = df[df.has_keyword]
    true_neg = df[~df.has_keyword]

    print("=" * 70)
    print("DOES THE PEAK SCORE SEPARATE PRESENT FROM ABSENT?")
    print("=" * 70)

    comparisons = [
        ("raw peak, masked negatives", kw.pos_peak, kw.neg_peak_masked),
        ("peak minus median, masked", kw.pos_peak_rel, kw.neg_peak_masked_rel),
    ]
    if "neg_peak_excised" in kw:
        comparisons += [
            ("raw peak, excised negatives", kw.pos_peak, kw.neg_peak_excised),
            ("peak minus median, excised", kw.pos_peak_rel, kw.neg_peak_excised_rel),
        ]
    if len(true_neg) >= 5:
        comparisons.append(
            ("raw peak vs the 14 real negatives", kw.pos_peak, true_neg.global_peak))
        comparisons.append(
            ("peak minus median vs real negatives",
             kw.pos_peak_rel, true_neg.global_peak_rel))

    for name, p, n in comparisons:
        p = np.asarray(p.dropna()) if hasattr(p, "dropna") else np.asarray(p)
        n = np.asarray(n.dropna()) if hasattr(n, "dropna") else np.asarray(n)
        if len(p) < 5 or len(n) < 5:
            continue
        a = auc(p, n)
        thr, bacc, eer = best_threshold(p, n)
        print(f"\n{name}   ({len(p)} positive, {len(n)} negative)")
        print(f"  positive: mean {p.mean():+.2f}  median {np.median(p):+.2f}")
        print(f"  negative: mean {n.mean():+.2f}  median {np.median(n):+.2f}")
        print(f"  AUC {a:.3f}   balanced accuracy {bacc:.1%} at threshold {thr:+.2f}"
              f"   EER {eer:.1%}")
        print(f"  -> {verdict(a)}")

    print("\nper speaker (peak minus median, masked negatives):")
    for spk, g in kw.groupby("speaker_id"):
        p = g.pos_peak_rel.dropna().to_numpy()
        n = g.neg_peak_masked_rel.dropna().to_numpy()
        if len(p) >= 5 and len(n) >= 5:
            print(f"  {spk}: AUC {auc(p, n):.3f}  ({len(p)} recordings)")

    text_histogram(kw.pos_peak_rel.dropna().to_numpy(),
                   kw.neg_peak_masked_rel.dropna().to_numpy(),
                   true_neg.global_peak_rel.dropna().to_numpy()
                   if len(true_neg) else np.array([]))

    if args.plot:
        try:
            make_plot(kw, true_neg, args.plot)
        except ImportError:
            print("\n(matplotlib is not installed, so no image was written - the "
                  "text histogram above shows the same thing.\n pip install "
                  "matplotlib if you want the plot too.)")


def text_histogram(pos, neg, real_neg, bins=28, width=44):
    """ASCII overlay of the two score distributions, so no plotting library is
    needed to read the result."""
    if len(pos) < 5 or len(neg) < 5:
        return
    lo = min(pos.min(), neg.min())
    hi = max(pos.max(), neg.max())
    edges = np.linspace(lo, hi, bins + 1)
    hp, _ = np.histogram(pos, bins=edges)
    hn, _ = np.histogram(neg, bins=edges)
    peak = max(hp.max(), hn.max()) or 1

    print("\n" + "=" * 70)
    print("SCORE DISTRIBUTIONS (peak minus median)")
    print("=" * 70)
    print(f"{'score':>8}  {'keyword present':<{width}} {'no keyword':<{width}}")
    for i in range(bins):
        a = "#" * int(width * hp[i] / peak)
        b = "." * int(width * hn[i] / peak)
        mark = ""
        if len(real_neg):
            k = int(((real_neg >= edges[i]) & (real_neg < edges[i + 1])).sum())
            if k:
                mark = f"  <- {k} real negative" + ("s" if k > 1 else "")
        print(f"{edges[i]:8.2f}  {a:<{width}} {b:<{width}}{mark}")
    print(f"\n  # = the {len(pos)} recordings containing the keyword")
    print(f"  . = the same recordings scored outside the keyword")
    if len(real_neg):
        print(f"  arrows mark the {len(real_neg)} recordings that genuinely have none")
    print("\n  Overlapping rows mean the score cannot tell presence from absence.")
    print("  Separated rows mean a threshold placed between them would work.")


def make_plot(kw, true_neg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("raw peak", "pos_peak", "neg_peak_masked", "global_peak"),
              ("peak minus median", "pos_peak_rel", "neg_peak_masked_rel",
               "global_peak_rel")]
    fig, axes = plt.subplots(1, len(panels), figsize=(13, 4.5))
    for ax, (title, pc, nc, tc) in zip(np.atleast_1d(axes), panels):
        p = kw[pc].dropna().to_numpy()
        n = kw[nc].dropna().to_numpy()
        lo, hi = min(p.min(), n.min()), max(p.max(), n.max())
        bins = np.linspace(lo, hi, 40)
        ax.hist(n, bins=bins, alpha=0.6, label="no keyword (masked)", density=True)
        ax.hist(p, bins=bins, alpha=0.6, label="keyword present", density=True)
        if len(true_neg) >= 3 and tc in true_neg:
            for v in true_neg[tc].dropna():
                ax.axvline(v, color="black", lw=0.6, alpha=0.5)
        ax.set_title(f"{title}   AUC {auc(p, n):.3f}")
        ax.set_xlabel("score")
        ax.legend(fontsize=8)
    fig.suptitle("Vertical lines are the 14 recordings that genuinely have no keyword")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
