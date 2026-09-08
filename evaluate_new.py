#!/usr/bin/env python3
"""
evaluate_new.py
===============
Measures how often the system is right, on recordings it has never seen.

WHY THIS IS THE ONLY WAY TO GET THAT NUMBER
    Everything measured so far used negatives synthesised by cutting the keyword
    out of real recordings. That is fine for ranking, but a take that never
    contained the word is not the same thing. Only nine genuinely keyword-free
    recordings exist in the original corpus, and the detector saw all nine during
    training, so the false-alarm rate has never actually been measured. A fresh
    set fixes that and nothing else does.

THE RULE THAT MAKES IT HONEST
    The threshold must be chosen BEFORE looking at these recordings, from the
    old data:

        python analyze.py ./laser --meta metadata_clean.csv --sweep

    Pick a row from that table, pass it here as --threshold, and run once.
    Choosing the threshold that happens to score best on this set would inflate
    the result and would burn the only clean test data available. --all-thresholds
    exists for curiosity and prints a warning; the number it produces is not the
    one to report.

LABELS
    A CSV with one row per recording:

        recording_id,has_keyword,start_s,end_s
        n001,1,7.30,7.70
        n002,0,,

    start_s and end_s are optional and only used to score timing. Run with
    --make-template to generate the file with one row per wav, then fill it in.

WHAT 50 RECORDINGS BUY
    With roughly 25 per class, each rate carries a 95% interval of about
    +/- 16 points. That is a real measurement and far better than nine, but it
    will not separate 70% from 80%. The script prints the intervals so the
    number is never read as more precise than it is.

Usage:
    python evaluate_new.py ./new_recordings --make-template labels.csv
    python evaluate_new.py ./new_recordings --labels labels.csv --threshold 0.5
"""

import argparse
import math
import os
from glob import glob

import numpy as np
import pandas as pd

from locate import HIT_TOL, load_audio, load_models, locate
from predict import load_ensemble, predict_one
from train_help_v5_mac import Features, get_device


def wilson(k, n, z=1.96):
    """Confidence interval for a proportion; honest at small n, unlike k/n +/- z*se."""
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


def rec_id(path):
    return os.path.splitext(os.path.basename(path))[0].split("_")[-1]


def make_template(paths, out):
    rows = [{"recording_id": rec_id(p), "file": os.path.basename(p),
             "has_keyword": "", "start_s": "", "end_s": ""} for p in paths]
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"wrote {out} with {len(rows)} rows")
    print("\nFill in has_keyword with 1 or 0. start_s and end_s are optional and")
    print("only needed to score the timing of the recordings that do contain it.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="folder of new .wav recordings")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--threshold", type=float, default=None,
                    help="chosen in advance from the old data, not from these")
    ap.add_argument("--make-template", default=None)
    ap.add_argument("--all-thresholds", action="store_true",
                    help="print every threshold; invalidates this as a clean test")
    ap.add_argument("--detector-models", default="outputs/detector_v2")
    ap.add_argument("--localizer-models", default="outputs/help_v5_fixed")
    ap.add_argument("--csv", default="new_results.csv")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    paths = sorted(glob(os.path.join(args.input, "*.wav")))
    if not paths:
        raise SystemExit(f"no .wav files in {args.input}")
    print(f"{len(paths)} recordings in {args.input}")

    if args.make_template:
        make_template(paths, args.make_template)
        return
    if not args.labels:
        raise SystemExit("--labels is required; generate one with --make-template")
    if args.threshold is None and not args.all_thresholds:
        raise SystemExit(
            "--threshold is required.\n"
            "Choose it from the old data first:\n"
            "  python analyze.py ./laser --meta metadata_clean.csv --sweep\n"
            "Picking it from these recordings would inflate the result.")

    lab = pd.read_csv(args.labels, dtype={"recording_id": str})
    lab["has_keyword"] = lab.has_keyword.astype(float).astype(bool)
    lab = lab.set_index("recording_id")

    device = get_device(args.device)
    print(f"device: {device}\n\ndetector ({args.detector_models})")
    detector = load_ensemble(args.detector_models, device)
    print(f"\nlocalizer ({args.localizer_models})")
    localizer = load_models(args.localizer_models, device)
    feat = Features()

    rows, missing = [], []
    for p in paths:
        rid = rec_id(p)
        if rid not in lab.index:
            missing.append(rid)
            continue
        row = lab.loc[rid]
        wav, dur, _ = load_audio(p)
        d = predict_one(wav, detector, feat, device)
        rec = {"recording_id": rid, "file": os.path.basename(p),
               "has_keyword": bool(row.has_keyword),
               "probability": d["probability"], "det_spread": d["spread"]}
        if row.has_keyword:
            l = locate(wav, localizer, feat, device)
            rec["time_s"] = l["time_s"]
            rec["loc_spread_s"] = l["spread_s"]
            s, e = row.get("start_s"), row.get("end_s")
            if pd.notna(s) and pd.notna(e):
                rec["true_centre_s"] = 0.5 * (float(s) + float(e))
                rec["error_s"] = l["time_s"] - rec["true_centre_s"]
                rec["located_ok"] = abs(rec["error_s"]) <= HIT_TOL
        rows.append(rec)
        if len(rows) % 10 == 0:
            print(f"  {len(rows)}/{len(paths)}")

    if missing:
        print(f"\n{len(missing)} recordings had no label row: "
              f"{', '.join(missing[:10])}")

    df = pd.DataFrame(rows)
    df.to_csv(args.csv, index=False)
    pos, neg = df[df.has_keyword], df[~df.has_keyword]

    print("\n" + "=" * 66)
    print("HELD-OUT MEASUREMENT")
    print("=" * 66)
    print(f"{len(pos)} recordings with the keyword, {len(neg)} without")

    a = auc(pos.probability, neg.probability)
    print(f"\nAUC {a:.3f}   (0.500 is a coin; 0.786 was measured on synthesised "
          f"negatives)")

    if args.all_thresholds:
        print("\n--all-thresholds was used. Reporting the best row below would be")
        print("choosing a threshold on the test set, which inflates the number.\n")
        print(f"{'thresh':>7} {'detect rate':>12} {'false alarm':>12} {'balanced':>10}")
        print("-" * 44)
        for t in (0.25, 0.35, 0.45, 0.5, 0.6, 0.75, 0.85):
            tpr = (pos.probability >= t).mean()
            fpr = (neg.probability >= t).mean() if len(neg) else float("nan")
            print(f"{t:7.2f} {tpr:12.1%} {fpr:12.1%} {0.5*(tpr+(1-fpr)):10.1%}")
        return

    t = args.threshold
    tp = int((pos.probability >= t).sum())
    fn = len(pos) - tp
    fp = int((neg.probability >= t).sum())
    tn = len(neg) - fp

    print(f"\nat the threshold chosen in advance ({t:.2f}):\n")
    print(f"                    said yes    said no")
    print(f"  keyword present   {tp:8d}   {fn:8d}")
    print(f"  keyword absent    {fp:8d}   {tn:8d}")

    lo, hi = wilson(tp, len(pos))
    print(f"\n  detection rate  {tp}/{len(pos)} = {tp/max(len(pos),1):.1%}"
          f"   95% CI [{lo:.0f}, {hi:.0f}]")
    if len(neg):
        lo, hi = wilson(fp, len(neg))
        print(f"  false alarms    {fp}/{len(neg)} = {fp/len(neg):.1%}"
              f"   95% CI [{lo:.0f}, {hi:.0f}]")
    lo, hi = wilson(tp + tn, len(df))
    print(f"  overall correct {tp+tn}/{len(df)} = {(tp+tn)/len(df):.1%}"
          f"   95% CI [{lo:.0f}, {hi:.0f}]")
    print("\n  'overall correct' depends on how many of these recordings contain")
    print("  the keyword. If the real-world mix differs, it will differ too.")

    if "located_ok" in df:
        ok = df[df.located_ok.notna()]
        caught = ok[ok.probability >= t]
        if len(caught):
            k = int(caught.located_ok.sum())
            lo, hi = wilson(k, len(caught))
            print(f"\n  of the {len(caught)} caught, {k} were located within "
                  f"{HIT_TOL}s: {k/len(caught):.1%}   95% CI [{lo:.0f}, {hi:.0f}]")
            print(f"  median timing error {caught.error_s.median():+.2f}s")

    n_un = int((df.det_spread > 0.25).sum())
    if n_un:
        print(f"\n  {n_un} recordings had detectors disagreeing (spread > 0.25); "
              f"the system flagged those as unresolved")

    print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
