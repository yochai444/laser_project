#!/usr/bin/env python3
"""
detector_metrics.py
===================
Reports the detector as a classifier rather than as a ranker: recall for HELP,
specificity, balanced accuracy, precision and F1, each at a stated threshold and
each with a confidence interval.

WHY AUC IS NOT ENOUGH
    AUC 0.786 says that given one recording with the keyword and one without,
    the detector ranks them correctly 78.6% of the time. It says nothing about
    what happens at any particular threshold, and a system in use always has a
    threshold. Recall and specificity are the two numbers that describe that
    system, and they trade against each other as the threshold moves.

TWO NEGATIVE SETS, REPORTED SEPARATELY
    EXCISED   The 359 keyword recordings with the word and a 0.7 s guard cut out
              and the ends crossfaded. Plentiful, so the interval is tight, but
              synthetic: a take that never contained the word is not identical
              to one with a hole in it.

    REAL      The 9 recordings where the word was genuinely never spoken. The
              right kind of data, but nine of them, and the detector saw all
              nine during training. The interval on any rate from nine samples
              spans more than 30 points.

    They are never pooled. Specificity from the excised set is the one with a
    usable interval; specificity from the real set is the one that answers the
    question you actually care about. Neither alone is sufficient, and that is a
    property of the corpus, not of the reporting.

NO LEAKAGE
    Each recording is scored only by the fold whose training set excluded its
    speaker. Running the full ensemble would let six of seven models score a
    speaker they trained on.

Usage:
    python detector_metrics.py --laser-dir ./laser --meta metadata_clean.csv
    python detector_metrics.py --laser-dir ./laser --meta metadata_clean.csv \
        --threshold 0.45 --csv metrics.csv
"""

import argparse
import math
import os
import re
from glob import glob

import numpy as np
import pandas as pd
import torch

from predict import load_ensemble, predict_one
from train_detector_mac import GUARD_S, excise, recording_score
from train_help_v5_mac import SR, AudioCache, Features, get_device


def wilson(k, n, z=1.96):
    """Interval for a proportion. Used instead of k/n +/- z*se because the
    normal approximation is badly wrong at small n and near 0 or 1 - and both
    situations occur here."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(c - h, 0.0), 100 * min(c + h, 1.0)


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def held_out_speaker(fold_name):
    m = re.match(r"test_(\w+?)_val_(\w+)", fold_name or "")
    return m.group(1) if m else None


def metrics(pos_scores, neg_scores, t):
    """Confusion counts and the rates derived from them, at threshold t."""
    pos, neg = np.asarray(pos_scores, float), np.asarray(neg_scores, float)
    tp = int((pos >= t).sum())
    fn = len(pos) - tp
    fp = int((neg >= t).sum())
    tn = len(neg) - fp

    recall = tp / len(pos) if len(pos) else float("nan")
    spec = tn / len(neg) if len(neg) else float("nan")
    bal = 0.5 * (recall + spec)
    prec = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * prec * recall / (prec + recall)
          if np.isfinite(prec) and (prec + recall) > 0 else float("nan"))
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "recall": recall,
            "specificity": spec, "balanced": bal, "precision": prec, "f1": f1}


def show(name, m, n_pos, n_neg, note=""):
    print(f"\n{name}")
    if note:
        print(f"  {note}")
    print(f"                       said HELP    said no")
    print(f"    HELP present      {m['tp']:9d}  {m['fn']:9d}")
    print(f"    HELP absent       {m['fp']:9d}  {m['tn']:9d}")

    lo, hi = wilson(m["tp"], n_pos)
    print(f"\n    recall (HELP)      {m['recall']:7.1%}   95% CI [{lo:5.1f}, {hi:5.1f}]"
          f"   {m['tp']}/{n_pos}")
    lo, hi = wilson(m["tn"], n_neg)
    print(f"    specificity        {m['specificity']:7.1%}   95% CI [{lo:5.1f}, {hi:5.1f}]"
          f"   {m['tn']}/{n_neg}")
    print(f"    balanced accuracy  {m['balanced']:7.1%}")
    if np.isfinite(m["precision"]):
        print(f"    precision          {m['precision']:7.1%}"
              f"   (depends on how common HELP is in the mix)")
        print(f"    F1                 {m['f1']:7.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laser-dir", required=True)
    ap.add_argument("--meta", default="metadata_clean.csv")
    ap.add_argument("--models", default="outputs/detector_v2")
    ap.add_argument("--threshold", type=float, default=0.45)
    ap.add_argument("--csv", default="detector_metrics.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}\nloading detectors from {args.models}")
    members = load_ensemble(args.models, device)

    by_spk = {}
    for m in members:
        by_spk.setdefault(held_out_speaker(m.get("name")), []).append(m)
    print(f"\n{len(by_spk)} speakers covered by held-out folds")

    meta = pd.read_csv(args.meta, dtype={"recording_id": str, "speaker_id": str})
    meta = meta[~meta.exclude].reset_index(drop=True)
    cache = AudioCache(args.laser_dir, meta.recording_id.tolist(), "laser")
    meta = meta[meta.recording_id.isin(cache.data.keys())].reset_index(drop=True)
    feat = Features()

    rows = []
    for k, row in enumerate(meta.itertuples(), 1):
        mem = by_spk.get(row.speaker_id)
        if not mem:
            continue
        wav = cache.get(row.recording_id)
        p = predict_one(wav, mem, feat, device)["probability"]
        rec = {"recording_id": row.recording_id, "speaker_id": row.speaker_id,
               "has_keyword": bool(row.has_keyword), "probability": p,
               "kind": "positive" if row.has_keyword else "real_negative"}
        rows.append(rec)

        if row.has_keyword and np.isfinite(row.start_s):
            cut = excise(wav, row.start_s, row.end_s, GUARD_S)
            pc = predict_one(cut, mem, feat, device)["probability"]
            rows.append({"recording_id": row.recording_id + "_excised",
                         "speaker_id": row.speaker_id, "has_keyword": False,
                         "probability": pc, "kind": "excised_negative"})
        if k % 50 == 0:
            print(f"  {k}/{len(meta)}")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)

    pos = df[df.kind == "positive"].probability.to_numpy()
    exc = df[df.kind == "excised_negative"].probability.to_numpy()
    real = df[df.kind == "real_negative"].probability.to_numpy()
    t = args.threshold

    print("\n" + "=" * 68)
    print(f"DETECTOR AT THRESHOLD {t:.2f}")
    print("=" * 68)
    print(f"{len(pos)} recordings with the keyword, {len(exc)} excised "
          f"negatives, {len(real)} real negatives")

    print(f"\nAUC vs excised negatives : {auc(pos, exc):.3f}")
    if len(real):
        print(f"AUC vs real negatives    : {auc(pos, real):.3f}   "
              f"(only {len(real)} recordings)")
    print("0.500 is a coin.")

    show("AGAINST EXCISED NEGATIVES", metrics(pos, exc, t), len(pos), len(exc),
         "the keyword cut out of real recordings - plentiful, but synthetic")
    if len(real) >= 3:
        show("AGAINST REAL NEGATIVES", metrics(pos, real, t), len(pos), len(real),
             f"only {len(real)} exist and the detector trained on all of them; "
             f"the specificity interval below is too wide to act on")

    print("\n\nPER SPEAKER (excised negatives)")
    print(f"{'spk':>5} {'n':>5} {'recall':>8} {'specificity':>12} "
          f"{'balanced':>9} {'AUC':>7}")
    print("-" * 50)
    for spk, g in df.groupby("speaker_id"):
        p = g[g.kind == "positive"].probability.to_numpy()
        e = g[g.kind == "excised_negative"].probability.to_numpy()
        if len(p) < 5 or len(e) < 5:
            continue
        m = metrics(p, e, t)
        print(f"{spk:>5} {len(p):5d} {m['recall']:8.1%} {m['specificity']:12.1%} "
              f"{m['balanced']:9.1%} {auc(p, e):7.3f}")

    print("\n\nACROSS THRESHOLDS (excised negatives)")
    print(f"{'thresh':>7} {'recall':>8} {'specificity':>12} {'balanced':>9} "
          f"{'F1':>7}")
    print("-" * 46)
    best_t, best_b = t, -1.0
    for tt in np.arange(0.20, 0.86, 0.05):
        m = metrics(pos, exc, tt)
        star = ""
        if m["balanced"] > best_b:
            best_b, best_t = m["balanced"], tt
        if abs(tt - t) < 1e-9:
            star = "  <- chosen"
        print(f"{tt:7.2f} {m['recall']:8.1%} {m['specificity']:12.1%} "
              f"{m['balanced']:9.1%} {m['f1']:7.1%}{star}")

    print(f"\nbalanced accuracy peaks at {best_t:.2f} ({best_b:.1%}).")
    if abs(best_t - t) > 0.02:
        print("That is where this data happens to peak; adopting it because it")
        print("looks best here is tuning on the evaluation set. Change the")
        print("threshold only against data the models have not been measured on.")

    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
