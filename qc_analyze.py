"""
qc_analyze.py
-------------
Digs into the qc_report.csv that qc_sync.py already wrote. No audio is re-read,
so this runs in a second.

The summary printed by qc_sync.py leaves the two most diagnostic columns
unreported. This script surfaces them and separates three very different
situations that all look like "low sync_corr":

  A. THE MEASUREMENT FAILED, NOT THE SIGNAL.
     When the envelopes share nothing, the cross-correlation peak is flat and
     the argmax lands wherever noise is highest - often right at the edge of the
     +/-1.0 s search window. Any recording whose offset saturated at the boundary
     has an offset value that means nothing at all, and it drags the per-speaker
     mean and std with it. These have to be excluded before the offsets can be
     read as timing information.

  B. THE RECORDINGS ARE ALIGNED, THE LASER IS JUST NOISY.
     sync_corr_at_zero compares the two envelopes with no lag applied. If it is
     close to the free-lag sync_corr, then zero lag was already near-optimal and
     the labels do not need shifting - the correlation is low because the laser
     is noisy, not because it is misaligned.

  C. THE KEYWORD IS OR IS NOT PRESENT IN THE LASER.
     label_laser_energy_ratio is the laser's energy inside the annotated keyword
     interval divided by its median energy. Above ~1.0 means the word left a
     visible mark on the optical signal. Around 1.0 means the laser did not
     capture it, and no architecture will recover it.

Usage:
    python qc_analyze.py --qc qc_report.csv --meta metadata_clean.csv
"""

import argparse

import numpy as np
import pandas as pd

MAX_LAG_S = 1.0          # must match qc_sync.py
BOUNDARY_TOL = 0.05      # treat |offset| within this of MAX_LAG_S as saturated


def pct(x):
    return f"{100 * x:.1f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", default="qc_report.csv")
    ap.add_argument("--meta", default="metadata_clean.csv")
    ap.add_argument("--out", default="qc_analysis.csv")
    args = ap.parse_args()

    qc = pd.read_csv(args.qc, dtype={"recording_id": str, "speaker_id": str})
    meta = pd.read_csv(args.meta, dtype={"recording_id": str, "speaker_id": str})
    df = qc.merge(meta.drop(columns=["speaker_id"]), on="recording_id", how="left")
    df = df[df.status == "ok"].copy()

    # --- A. which offset estimates are meaningless -------------------------
    df["offset_saturated"] = df.offset_s.abs() >= (MAX_LAG_S - BOUNDARY_TOL)
    n_sat = int(df.offset_saturated.sum())
    print("=" * 74)
    print("A. DID THE OFFSET MEASUREMENT SUCCEED?")
    print("=" * 74)
    print(f"{n_sat}/{len(df)} ({pct(n_sat / len(df))}) offsets landed on the "
          f"+/-{MAX_LAG_S:.1f}s search boundary.")
    print("Those are failed searches, not measured delays. Excluding them:\n")

    clean = df[~df.offset_saturated]
    print(clean.groupby("speaker_id").offset_s.describe()
          [["count", "mean", "50%", "std"]].round(3).to_string())

    # Restrict further to recordings where the correlation is strong enough for
    # the peak location to be trustworthy at all.
    trust = df[(~df.offset_saturated) & (df.sync_corr >= 0.35)]
    print(f"\nRestricted to sync_corr >= 0.35 ({len(trust)} recordings) - the only "
          "offsets worth believing:\n")
    if len(trust):
        print(trust.groupby("speaker_id").offset_s.describe()
              [["count", "mean", "50%", "std"]].round(3).to_string())
        med = float(trust.offset_s.median())
        print(f"\nPooled median offset: {med:+.3f}s")
        if abs(med) < 0.05:
            print("-> mic and laser are aligned. The annotations do not need shifting.")
        else:
            print(f"-> a real delay of {med:+.3f}s; every label should be shifted by this.")

    # --- B. is zero lag already right? -------------------------------------
    print("\n" + "=" * 74)
    print("B. ALIGNMENT VS SIGNAL QUALITY")
    print("=" * 74)
    if "sync_corr_at_zero" in df:
        df["lag_gain"] = df.sync_corr - df.sync_corr_at_zero
        print(df.groupby("speaker_id")[["sync_corr", "sync_corr_at_zero", "lag_gain"]]
              .mean().round(3).to_string())
        gain = float(df.lag_gain.mean())
        print(f"\nMean improvement from allowing a lag: {gain:+.3f}")
        if gain < 0.05:
            print("-> shifting the tracks buys almost nothing. The recordings are")
            print("   already aligned and the low correlation is a signal-to-noise")
            print("   problem in the laser channel, not a timing problem.")
        else:
            print("-> a lag genuinely helps; timing is part of the problem.")

    # --- C. did the keyword survive into the laser? ------------------------
    print("\n" + "=" * 74)
    print("C. IS THE KEYWORD VISIBLE IN THE LASER SIGNAL?")
    print("=" * 74)
    kw = df[df.has_keyword == True]  # noqa: E712
    if "label_laser_energy_ratio" in kw and len(kw):
        print("laser energy inside the annotated word / median laser energy:\n")
        print(kw.groupby("speaker_id").label_laser_energy_ratio.describe()
              [["count", "mean", "50%", "std"]].round(3).to_string())
        print("\nsame, for the microphone (the control - this should be well above 1):\n")
        if "label_mic_energy_ratio" in kw:
            print(kw.groupby("speaker_id").label_mic_energy_ratio.describe()
                  [["count", "mean", "50%"]].round(3).to_string())
        frac = float((kw.label_laser_energy_ratio > 1.15).mean())
        print(f"\n{pct(frac)} of recordings show a clear energy bump in the laser "
              "at the annotated position.")

    # --- signal health, and how it splits by speaker -----------------------
    print("\n" + "=" * 74)
    print("D. LASER CHANNEL HEALTH BY SPEAKER")
    print("=" * 74)
    cols = [c for c in ["sync_corr", "laser_inband_ratio", "laser_peak",
                        "laser_clip_frac"] if c in df]
    print(df.groupby("speaker_id")[cols].mean().round(3).to_string())
    print("\nlaser_inband_ratio is the share of laser energy inside 100-1400 Hz.")
    print("Low values mean the recovered signal is dominated by content outside")
    print("the band that can carry speech - drift, or broadband tracking noise.")

    # --- does noise condition explain it? ----------------------------------
    if "noise_type" in df:
        print("\nsync_corr by noise condition:")
        print(df.groupby("noise_type").sync_corr.agg(["count", "mean"]).round(3).to_string())

    # --- ranked worst recordings ------------------------------------------
    df["quality_rank"] = df.sync_corr.rank(pct=True)
    df.sort_values("sync_corr").to_csv(args.out, index=False)
    print(f"\nwrote {args.out} (sorted worst-first, with quality_rank percentile)")

    good = df[df.sync_corr >= 0.35]
    print(f"\nIf you trained on only the {len(good)} recordings with sync_corr >= 0.35, "
          "the speaker breakdown would be:")
    print(good.groupby("speaker_id").size().to_string())


if __name__ == "__main__":
    main()
