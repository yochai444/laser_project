#!/usr/bin/env python3
"""
locate.py
=========
The localizer counterpart to predict.py. Given a WAV from the optical receiver,
it returns WHEN "help" was spoken, on the assumption that it was spoken once.

It does not decide WHETHER the word is there - that is predict.py's job. The
localizer is trained with a softmax over the time axis, so its scores sum to one
within a recording and can only express where the best moment is. Given a
recording with no keyword it will still point somewhere, confidently.

ENSEMBLING
    All seven fold checkpoints are run and their time distributions averaged
    before taking the peak. Each was trained without a different speaker, so no
    single one has seen every speaking style. The spread of the individual peaks
    is reported alongside: when the models disagree about the moment, the
    averaged answer means little.

EVALUATION MODE
    With --meta, each recording is scored by the ONE model whose fold excluded
    that speaker. That is the honest measurement - the other six models saw the
    speaker during training and would flatter the result. This reproduces the
    cross-validation numbers rather than inventing better-looking ones.

Measured accuracy, peak within 0.5 s of the annotated centre, on speakers the
model never saw: 72.2% on the two folds where a constant-time guess scores near
zero. On the other five folds the keyword always falls near 11 s, so a fixed
answer already scores 61-92% and those folds cannot discriminate.

Usage:
    python locate.py recording.wav
    python locate.py ./laser --meta metadata_clean.csv     # honest evaluation
    python locate.py ./new_recordings --csv times.csv
"""

import argparse
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal as sps
from scipy.io import wavfile

from train_help_v5_mac import (SR, REC_SEC, OUT_FPS, Features, HelpLocalizerV5,
                               get_device)

HIT_TOL = 0.5


def load_audio(path):
    sr, x = wavfile.read(path)
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)

    notes = []
    if sr != SR:
        g = np.gcd(int(sr), SR)
        x = sps.resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        notes.append(f"resampled from {sr} Hz")
    return x, len(x) / SR, notes


def load_models(model_dir, device):
    """Load every fold checkpoint, remembering which speaker each one held out.

    model_dir may be several directories separated by commas, so runs made with
    different --seed values can be pooled into one ensemble.
    """
    members = []
    dirs = [d.strip() for d in str(model_dir).split(",") if d.strip()]
    cks = [c for d in dirs for c in sorted(glob(os.path.join(d, "*", "best_help_v*.pt")))]
    for ck in cks:
        fold = Path(ck).parent.name
        m = re.match(r"test_(\w+?)_val_(\w+)", fold)
        held = m.group(1) if m else None

        c = torch.load(ck, map_location="cpu")
        model = HelpLocalizerV5(c["n_bins"], width=c.get("width", 32))
        model.load_state_dict(c["model_state_dict"])
        model.to(device).eval()
        tag = (f"{Path(ck).parent.parent.name}/{fold}" if len(dirs) > 1 else fold)
        members.append({"fold": fold, "tag": tag, "held_out": held, "model": model})
        print(f"  {tag}: held out speaker {held}")
    if not members:
        raise SystemExit(f"no localizer checkpoints under {model_dir}")
    return members


@torch.no_grad()
def curve(model, wav, feat, device):
    x = feat(torch.from_numpy(wav)).unsqueeze(0).to(device)
    logits = model(x).squeeze(0).float().cpu()
    t = (torch.arange(len(logits), dtype=torch.float32) + 0.5) / OUT_FPS
    return t.numpy(), logits


def locate(wav, members, feat, device, topk=3):
    """Average the time distributions, then take the peak."""
    dists, peaks, times = [], [], None
    for m in members:
        t, logits = curve(m["model"], wav, feat, device)
        times = t
        p = F.softmax(logits, dim=0).numpy()
        dists.append(p)
        peaks.append(float(t[int(np.argmax(p))]))

    avg = np.mean(dists, axis=0)
    k = int(np.argmax(avg))

    # Candidate moments, keeping them at least 0.5 s apart so the list is not
    # three neighbouring frames of the same peak.
    order = np.argsort(-avg)
    cands = []
    for i in order:
        if all(abs(times[i] - c[0]) > HIT_TOL for c in cands):
            cands.append((float(times[i]), float(avg[i])))
        if len(cands) >= topk:
            break

    return {"time_s": float(times[k]),
            "confidence": float(avg[k] / avg.mean()),   # peak over a flat curve
            "spread_s": float(np.std(peaks)),
            "member_times": peaks,
            "candidates": cands}


def describe(r):
    if r["spread_s"] > 1.0:
        return "the models point at different moments; treat this as unresolved"
    if r["confidence"] > 30:
        return "a sharp, confident peak"
    if r["confidence"] > 10:
        return "a clear peak"
    return "a weak peak; the score is spread over much of the recording"


def evaluate(paths, members, feat, device, meta_path, model_dir):
    """Score each recording with the one model that never saw its speaker."""
    import pandas as pd
    meta = pd.read_csv(meta_path, dtype={"recording_id": str, "speaker_id": str})
    meta = meta[~meta.exclude]
    by_speaker = {m["held_out"]: m for m in members}

    idx = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        idx[stem.split("_")[-1]] = p

    rows, skipped = [], 0
    for row in meta.itertuples():
        if not (row.has_keyword and np.isfinite(row.start_s)):
            continue
        p = idx.get(row.recording_id)
        m = by_speaker.get(row.speaker_id)
        if p is None or m is None:
            skipped += 1
            continue
        wav, _, _ = load_audio(p)
        t, logits = curve(m["model"], wav, feat, device)
        peak = float(t[int(torch.argmax(logits))])
        centre = 0.5 * (row.start_s + row.end_s)
        rows.append({"recording_id": row.recording_id,
                     "speaker_id": row.speaker_id,
                     "fold": m["fold"],
                     "predicted_s": round(peak, 3),
                     "true_centre_s": round(centre, 3),
                     "error_s": round(peak - centre, 3),
                     "hit": abs(peak - centre) <= HIT_TOL})

    df = pd.DataFrame(rows)
    if not len(df):
        raise SystemExit("nothing to evaluate; check that the wav names match "
                         "the recording ids in the metadata")

    print(f"\nscored {len(df)} recordings, each by the model that never saw its "
          f"speaker" + (f" ({skipped} skipped)" if skipped else ""))
    print(f"\n{'speaker':>8} {'n':>4} {'accuracy':>9} {'median err':>11}")
    for spk, g in df.groupby("speaker_id"):
        print(f"{spk:>8} {len(g):4d} {g.hit.mean():9.1%} "
              f"{g.error_s.median():+10.2f}s")
    print(f"\n{'overall':>8} {len(df):4d} {df.hit.mean():9.1%} "
          f"{df.error_s.median():+10.2f}s")

    # The five folds where a constant guess already scores 61-92% cannot
    # discriminate; speakers 06 and 07 spoke earlier than the training speakers,
    # so only those two are a real test.
    clean = df[df.speaker_id.isin(["06", "07"])]
    if len(clean):
        print(f"\nspeakers 06 and 07 only - the folds where a fixed-time guess "
              f"scores near zero:\n  {clean.hit.mean():.1%} over {len(clean)} "
              f"recordings")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="a .wav file or a folder of them")
    ap.add_argument("--models", default="outputs/help_v5_fixed")
    ap.add_argument("--meta", default=None,
                    help="run the honest per-speaker evaluation instead")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}\nloading localizers from {args.models}")
    members = load_models(args.models, device)
    feat = Features()
    print(f"{len(members)} models")

    paths = (sorted(glob(os.path.join(args.input, "*.wav")))
             if os.path.isdir(args.input) else [args.input])

    if args.meta:
        df = evaluate(paths, members, feat, device, args.meta, args.models)
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"\nwrote {args.csv}")
        return

    rows = []
    for p in paths:
        try:
            wav, dur, notes = load_audio(p)
            r = locate(wav, members, feat, device)
        except Exception as exc:
            print(f"{os.path.basename(p)}: could not read ({exc})")
            continue
        print(f"\n{os.path.basename(p)}   ({dur:.1f}s)")
        print(f"  most likely moment: {r['time_s']:.2f}s")
        print(f"  {describe(r)}")
        print(f"  peak {r['confidence']:.0f}x a flat curve, "
              f"models disagree by {r['spread_s']:.2f}s")
        others = ", ".join(f"{t:.2f}s" for t, _ in r["candidates"][1:])
        if others:
            print(f"  other candidates: {others}")
        for n in notes:
            print(f"  note: {n}")
        rows.append({"file": os.path.basename(p), "duration_s": round(dur, 2),
                     "time_s": round(r["time_s"], 3),
                     "confidence": round(r["confidence"], 2),
                     "spread_s": round(r["spread_s"], 3)})

    if args.csv and rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
