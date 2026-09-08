#!/usr/bin/env python3
"""
analyze.py
==========
One command over a recording from the optical receiver. The detector decides
whether "help" was said; only if it clears the threshold does the localizer say
when.

WHY THE TWO ARE SEPARATE MODELS
    The localizer is trained with a softmax over the time axis, so its scores
    sum to one within a recording. It can express WHERE the best moment is and
    is mathematically incapable of expressing that no moment is good - asked
    about a recording with no keyword it points somewhere anyway, and measured
    on the presence question it scores AUC 0.60, near a coin. The detector is
    trained directly on presence with binary cross entropy and reaches 0.786.
    Gating one behind the other is what makes "no answer" possible at all.

CHOOSING THE THRESHOLD
    There is no single right value. At AUC 0.786 misses and false alarms trade
    against each other and the balance depends on what the system is for:

      0.35  catch as much as possible, accept false alarms
      0.50  balanced (default)
      0.75  only report when fairly sure, accept misses

    --sweep prints what each threshold would have done on recordings whose
    labels you already have, which is the honest way to pick one.

A CAVEAT ABOUT CALIBRATION
    Probabilities come from a logistic fit on held-out scores, and the negatives
    used to fit it were synthesised by cutting the keyword out of real
    recordings. Only nine genuinely keyword-free recordings exist, all of which
    the detector saw during training, so the false-alarm rate at any threshold
    is the least trustworthy number here. Around 100-150 real keyword-free
    takes would fix that, and nothing else will.

Usage:
    python analyze.py recording.wav
    python analyze.py ./recordings --threshold 0.35 --csv results.csv
    python analyze.py ./laser --meta metadata_clean.csv --sweep
"""

import argparse
import os
import re
from glob import glob

import numpy as np

from locate import HIT_TOL, load_audio, load_models, locate
from predict import load_ensemble, predict_one
from train_help_v5_mac import Features, get_device


def analyze_one(wav, detector, localizer, feat, device, threshold):
    d = predict_one(wav, detector, feat, device)
    out = {"probability": d["probability"], "det_spread": d["spread"],
           "detected": d["probability"] >= threshold}

    if out["detected"]:
        l = locate(wav, localizer, feat, device)
        out.update(time_s=l["time_s"], confidence=l["confidence"],
                   loc_spread_s=l["spread_s"], candidates=l["candidates"])
    else:
        out.update(time_s=None, confidence=None, loc_spread_s=None,
                   candidates=[])
    return out


def report(name, dur, r, threshold, notes):
    print(f"\n{name}   ({dur:.1f}s)")
    if r["detected"]:
        print(f"  DETECTED at {r['time_s']:.2f}s")
        print(f"    probability {r['probability']:.0%} "
              f"(threshold {threshold:.0%})")
        print(f"    peak {r['confidence']:.0f}x a flat curve, "
              f"localizers disagree by {r['loc_spread_s']:.2f}s")
        if r["loc_spread_s"] > 1.0:
            print("    the localizers point at different moments, so the "
                  "timing is unreliable")
        others = ", ".join(f"{t:.2f}s" for t, _ in r["candidates"][1:])
        if others:
            print(f"    other candidates: {others}")
    else:
        print(f"  not detected")
        print(f"    probability {r['probability']:.0%}, below the "
              f"{threshold:.0%} threshold")
    if r["det_spread"] > 0.25:
        print("    the detectors disagree with each other; treat this as "
              "unresolved rather than as a decision")
    for n in notes:
        print(f"    note: {n}")


def held_out_speaker(fold_name):
    m = re.match(r"test_(\w+?)_val_(\w+)", fold_name or "")
    return m.group(1) if m else None


def sweep(paths, detector, localizer, feat, device, meta_path):
    """Show what each threshold would have done on labelled recordings.

    Every recording is scored ONLY by the fold whose training set excluded its
    speaker. Running the full ensemble here would let six of the seven models
    score a speaker they trained on, which inflates every column: an earlier
    version did exactly that and reported 95% localization against the 71%
    that the honest per-speaker evaluation gives.
    """
    import pandas as pd
    meta = pd.read_csv(meta_path, dtype={"recording_id": str, "speaker_id": str})
    meta = meta[~meta.exclude].set_index("recording_id")

    # A dict comprehension would keep only the last member per speaker, which
    # silently discards the other seeds once several runs are pooled.
    det_by_spk, loc_by_spk = {}, {}
    for m in detector:
        det_by_spk.setdefault(held_out_speaker(m.get("name")), []).append(m)
    for m in localizer:
        loc_by_spk.setdefault(m["held_out"], []).append(m)
    common = set(det_by_spk) & set(loc_by_spk) - {None}
    n_per = len(det_by_spk[next(iter(common))]) if common else 0
    print(f"\nscoring each speaker only with folds that never saw it "
          f"({len(common)} speakers, {n_per} model(s) each)")

    idx = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        idx[stem.split("_")[-1]] = p

    rows, skipped = [], 0
    for rid, p in sorted(idx.items()):
        if rid not in meta.index:
            continue
        row = meta.loc[rid]
        spk = row.speaker_id
        if spk not in det_by_spk or spk not in loc_by_spk:
            skipped += 1
            continue
        wav, _, _ = load_audio(p)
        d = predict_one(wav, det_by_spk[spk], feat, device)
        has = bool(row.has_keyword)
        rec = {"recording_id": rid, "speaker_id": spk, "has_keyword": has,
               "probability": d["probability"]}
        if has and np.isfinite(row.start_s):
            l = locate(wav, loc_by_spk[spk], feat, device)
            centre = 0.5 * (row.start_s + row.end_s)
            rec["time_s"] = l["time_s"]
            rec["located_correctly"] = bool(abs(l["time_s"] - centre) <= HIT_TOL)
        rows.append(rec)
        if len(rows) % 50 == 0:
            print(f"  {len(rows)} scored")
    if skipped:
        print(f"  {skipped} skipped: no fold held their speaker out")

    df = pd.DataFrame(rows)
    if "located_correctly" in df:
        df["located_correctly"] = df.located_correctly.fillna(False).astype(bool)
    pos = df[df.has_keyword]
    neg = df[~df.has_keyword]

    print(f"\n{len(pos)} recordings with the keyword, {len(neg)} without\n")
    if len(neg) < 30:
        print(f"Only {len(neg)} keyword-free recordings exist, and the detector")
        print("saw them during training. Every false-alarm number below rests on")
        print("that handful and should be read as indicative, not measured.\n")

    print(f"{'threshold':>10} {'caught':>8} {'missed':>8} {'false alarms':>13} "
          f"{'located ok':>11}")
    print("-" * 56)
    for t in (0.25, 0.35, 0.45, 0.5, 0.6, 0.75, 0.85):
        fired = pos.probability >= t
        caught = int(fired.sum())
        fa = int((neg.probability >= t).sum()) if len(neg) else 0
        if "located_correctly" in pos:
            loc_rate = float(pos.loc[fired, "located_correctly"].mean()) if caught else 0.0
        else:
            loc_rate = float("nan")
        print(f"{t:10.2f} {caught/len(pos):8.1%} {1-caught/len(pos):8.1%} "
              f"{fa:5d}/{len(neg):<7d} {loc_rate:11.1%}")

    print("\n'located ok' is the share of caught recordings whose predicted")
    print("moment falls within 0.5 s of the annotation - the end-to-end number.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="a .wav file or a folder of them")
    ap.add_argument("--detector-models", default="outputs/detector_v2")
    ap.add_argument("--localizer-models", default="outputs/help_v5_fixed")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--meta", default=None)
    ap.add_argument("--sweep", action="store_true",
                    help="with --meta, show what each threshold would do")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = get_device(args.device)
    print(f"device: {device}")
    print(f"\ndetector  ({args.detector_models})")
    detector = load_ensemble(args.detector_models, device)
    print(f"\nlocalizer ({args.localizer_models})")
    localizer = load_models(args.localizer_models, device)
    feat = Features()

    paths = (sorted(glob(os.path.join(args.input, "*.wav")))
             if os.path.isdir(args.input) else [args.input])

    if args.sweep:
        if not args.meta:
            ap.error("--sweep needs --meta")
        df = sweep(paths, detector, localizer, feat, device, args.meta)
        if args.csv:
            df.to_csv(args.csv, index=False)
            print(f"\nwrote {args.csv}")
        return

    rows = []
    for p in paths:
        try:
            wav, dur, notes = load_audio(p)
            r = analyze_one(wav, detector, localizer, feat, device,
                            args.threshold)
        except Exception as exc:
            print(f"{os.path.basename(p)}: could not read ({exc})")
            continue
        report(os.path.basename(p), dur, r, args.threshold, notes)
        rows.append({"file": os.path.basename(p), "duration_s": round(dur, 2),
                     "detected": r["detected"],
                     "probability": round(r["probability"], 4),
                     "time_s": round(r["time_s"], 3) if r["detected"] else "",
                     "confidence": round(r["confidence"], 2) if r["detected"] else "",
                     "det_spread": round(r["det_spread"], 3),
                     "loc_spread_s": round(r["loc_spread_s"], 3) if r["detected"] else ""})

    if args.csv and rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
