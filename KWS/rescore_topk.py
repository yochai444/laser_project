"""
rescore_topk.py
---------------
Re-evaluates prediction CSVs that V2/V3/V4 already produced, without retraining
anything.

Two things it measures that the existing metrics do not:

1. TOP-1 LOCALIZATION ACCURACY. The task is offline and every recording holds
   exactly one keyword, so the right decoder is "take the highest-scoring window
   in this recording", not "threshold every window and count false alarms per
   minute". Under argmax there is no threshold and no false alarm rate - either
   the peak lands on the word or it does not. A model that looks mediocre at
   FA/min <= 3.0 can be excellent under argmax.

2. THE TIME-PRIOR BASELINE. Across the corpus the keyword always falls between
   8.7 s and 13.1 s. A predictor that ignores the audio entirely and always
   answers with the most likely position learned from the training speakers will
   already score well. That number is the real floor. If the model does not
   clearly beat it, the model has learned the schedule of the experiment rather
   than the sound of the word.

Usage:
    python rescore_topk.py --pred outputs/help_cnn_v3_cv/test_07_val_01/test_predictions_v3.csv
    python rescore_topk.py --pred "outputs/help_cnn_v3_cv/*/test_predictions_v3.csv" --glob
"""

import argparse
import glob as globmod
import os

import numpy as np
import pandas as pd

# Column aliases, so this works on V2, V3 and V4 output alike.
ALIASES = {
    "recording_id": ["recording_id", "recording", "rec_id"],
    "speaker_id": ["recorder_id", "speaker_id", "RecorderID"],
    "window_start": ["window_start", "start", "win_start"],
    "window_end": ["window_end", "end", "win_end"],
    "help_start": ["help_start", "start_s", "StartTime"],
    "help_end": ["help_end", "end_s"],
    "probability": ["probability", "prob", "score", "p"],
    "label": ["true_label", "label", "y"],
}


def normalize_columns(df):
    out = {}
    lower = {c.lower(): c for c in df.columns}
    for canonical, options in ALIASES.items():
        for opt in options:
            if opt.lower() in lower:
                out[canonical] = df[lower[opt.lower()]]
                break
    missing = [k for k in ("recording_id", "window_start", "probability") if k not in out]
    if missing:
        raise ValueError(f"could not find columns for {missing}; saw {list(df.columns)}")
    res = pd.DataFrame(out)
    res["recording_id"] = res["recording_id"].astype(str).str.zfill(3)
    if "window_end" not in res:
        res["window_end"] = res["window_start"] + 1.0
    return res


def hit(peak_start, peak_end, hs, he, tol):
    """True if the winning window overlaps the annotated keyword, within tol."""
    return (peak_start - tol) < he and (peak_end + tol) > hs


def hit_center(peak_start, peak_end, hs, he, tol):
    """Stricter: the winning window's centre must be near the word's centre.

    The overlap criterion saturates. Inside one speaker the keyword occupies a
    window of about two seconds out of twenty, so a fixed guess overlaps it most
    of the time and both model and baseline reach 100%. Centre distance keeps
    discriminating after that.
    """
    return abs(0.5 * (peak_start + peak_end) - 0.5 * (hs + he)) <= tol


def topk_accuracy(df, k, tol, criterion=hit):
    """Fraction of recordings where one of the k best windows lands on the word."""
    ok = 0
    total = 0
    for _, g in df.groupby("recording_id"):
        if not np.isfinite(g["help_start"].iloc[0]):
            continue  # recording has no keyword; not a localization case
        total += 1
        best = g.nlargest(k, "probability")
        hs, he = g["help_start"].iloc[0], g["help_end"].iloc[0]
        if any(criterion(r.window_start, r.window_end, hs, he, tol)
               for r in best.itertuples()):
            ok += 1
    return ok / total if total else float("nan"), total


def peak_error(df):
    """Seconds between the centre of the winning window and the word centre."""
    errs = []
    for _, g in df.groupby("recording_id"):
        if not np.isfinite(g["help_start"].iloc[0]):
            continue
        r = g.loc[g["probability"].idxmax()]
        peak_c = 0.5 * (r["window_start"] + r["window_end"])
        word_c = 0.5 * (g["help_start"].iloc[0] + g["help_end"].iloc[0])
        errs.append(peak_c - word_c)
    return np.asarray(errs)


def time_prior_baseline(df, tol, criterion=hit, prior_start=None):
    """Audio-blind control: always answer at the same clock position.

    prior_start defaults to the window start that is correct most often within
    this file, which makes the baseline optimistic - it is tuned on the very set
    it is scored on. Beating an optimistic baseline is the bar worth clearing.
    """
    grouped = [g for _, g in df.groupby("recording_id")
               if np.isfinite(g["help_start"].iloc[0])]
    if not grouped:
        return float("nan"), None

    starts = sorted(df["window_start"].unique())
    if prior_start is None:
        scores = []
        for s in starts:
            n = sum(1 for g in grouped
                    if criterion(s, s + 1.0, g["help_start"].iloc[0],
                                 g["help_end"].iloc[0], tol))
            scores.append(n)
        prior_start = starts[int(np.argmax(scores))]
        best = max(scores) / len(grouped)
    else:
        best = sum(1 for g in grouped
                   if criterion(prior_start, prior_start + 1.0,
                                g["help_start"].iloc[0],
                                g["help_end"].iloc[0], tol)) / len(grouped)
    return best, prior_start


def verdict(acc, base):
    if not np.isfinite(base):
        return ""
    if base >= 0.98:
        return "<-- baseline saturates; this criterion cannot discriminate here"
    lift = acc - base
    err_red = lift / (1.0 - base)
    if lift <= 0.0:
        return "<-- losing to the clock; no evidence the audio is being used"
    if err_red < 0.25:
        return "<-- marginal; the time prior is doing most of the work"
    if err_red < 0.60:
        return "<-- real signal, but the time prior still carries a lot"
    return "<-- genuinely using the audio"


def line(name, acc, base, total=None):
    lift = acc - base
    err_red = lift / (1.0 - base) if np.isfinite(base) and base < 1.0 else float("nan")
    tail = f"({total} recordings)" if total else ""
    print(f"  {name:<22} model {acc:6.1%}   baseline {base:6.1%}   "
          f"lift {lift:+6.1%}   errors removed {err_red:+6.1%}  {tail}")
    print(f"  {'':<22} {verdict(acc, base)}")


def report(path, tol, ctol):
    raw = pd.read_csv(path)
    df = normalize_columns(raw)

    if "help_start" not in df:
        raise ValueError(f"{path}: no keyword annotation columns, cannot localize")

    print("=" * 72)
    print(os.path.relpath(path))
    print("=" * 72)

    n_rec = df["recording_id"].nunique()
    speakers = sorted(df["speaker_id"].astype(str).unique()) if "speaker_id" in df else ["?"]
    print(f"{len(df)} windows over {n_rec} recordings, speaker(s) {', '.join(speakers)}")

    acc1, total = topk_accuracy(df, 1, tol)
    acc2, _ = topk_accuracy(df, 2, tol)
    base, prior_s = time_prior_baseline(df, tol)

    print(f"\n  criterion: peak window overlaps the annotated word "
          f"(tolerance {tol:.2f}s), {total} recordings, baseline answers t={prior_s:.1f}s")
    line("top-1 overlap", acc1, base)
    line("top-2 overlap", acc2, base)

    tight, _ = topk_accuracy(df, 1, ctol, hit_center)
    tight_base, tight_s = time_prior_baseline(df, ctol, hit_center)
    print(f"\n  criterion: peak centre within {ctol:.2f}s of the word centre "
          f"(baseline answers t={tight_s:.1f}s)")
    line("top-1 centre", tight, tight_base)

    err = peak_error(df)
    if len(err):
        print(f"\n  peak offset from word centre: median {np.median(err):+.2f}s, "
              f"mean {err.mean():+.2f}s, IQR {np.percentile(err, 25):+.2f} to "
              f"{np.percentile(err, 75):+.2f}s")
        if abs(np.median(err)) > 0.15:
            print("  a consistent non-zero median points at a mic/laser timing offset, "
                  "not at model error")

    if "speaker_id" in df and len(speakers) > 1:
        print("\n  per speaker (centre criterion):")
        for spk, g in df.groupby(df["speaker_id"].astype(str)):
            a, n = topk_accuracy(g, 1, ctol, hit_center)
            b, _ = time_prior_baseline(g, ctol, hit_center)
            print(f"    {spk}: model {a:6.1%}  baseline {b:6.1%}  ({n} recordings)")
    print()


def discover(root):
    """Walk an output tree and return every prediction CSV in it."""
    found = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(".csv") and "prediction" in low:
                found.append(os.path.join(dirpath, fn))
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True,
                    help="a prediction CSV, a directory to search, or a glob with --glob")
    ap.add_argument("--glob", action="store_true")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="extra seconds of slack for the overlap criterion")
    ap.add_argument("--center-tol", type=float, default=0.5,
                    help="max distance between peak centre and word centre, in seconds")
    args = ap.parse_args()

    if args.glob:
        paths = sorted(globmod.glob(args.pred))
    elif os.path.isdir(args.pred):
        paths = discover(args.pred)
        print(f"found {len(paths)} prediction files under {args.pred}\n")
    else:
        paths = [args.pred]

    if not paths:
        print(f"nothing matched {args.pred}")
        print("Try pointing --pred at your outputs/ directory to search it.")
        return

    # Test files are the ones that matter; report them last so they end up at the
    # bottom of the terminal, where they are easiest to read.
    paths.sort(key=lambda p: ("test" in os.path.basename(p).lower(), p))

    for p in paths:
        try:
            report(p, args.tol, args.center_tol)
        except Exception as exc:
            print(f"{p}: {exc}\n")


if __name__ == "__main__":
    main()
