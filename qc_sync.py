"""
qc_sync.py
----------
Validates the corpus before any model is trained. For every recording it:

  1. locates the laser WAV and the matching reference-microphone WAV
  2. band-limits both to the frequency range the laser can physically carry
     (100 - 1400 Hz, set by the ~2.9 kHz frame rate / Nyquist ~1467 Hz)
  3. computes an energy envelope for each and cross-correlates them to measure
     the mic -> laser time offset
  4. checks that the annotated keyword interval actually lands on speech energy
     in the microphone signal
  5. flags dead / silent / clipped laser takes

Why this matters: every label was placed by hand on the microphone track. If the
mic and laser clocks are offset by even 100 ms, the labels are systematically
wrong and no amount of modelling will fix it. This script measures that offset
instead of assuming it is zero.

Only numpy / scipy / pandas are required - WAVs are read with scipy.io.wavfile,
so there is no soundfile or librosa dependency.

Usage:
    # first, check that the file naming is being resolved correctly
    python qc_sync.py --laser-dir ./laser --mic-dir ./mic --meta metadata_clean.csv --list-only

    # then run the full pass
    python qc_sync.py --laser-dir ./laser --mic-dir ./mic --meta metadata_clean.csv \
                      --out qc_report.csv --plot-dir ./qc_plots --n-plots 12
"""

import argparse
import os
import re
import warnings
from glob import glob

import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal

# --- physical / analysis constants ----------------------------------------
F_LO = 100.0      # below this is building vibration, removed in matrix2sound.m
F_HI = 1400.0     # just under the laser Nyquist (~1467 Hz at 2934 fps)
ENV_RATE = 200.0  # envelope sample rate in Hz -> 5 ms resolution
ENV_LP = 25.0     # envelope smoothing cutoff in Hz
MAX_LAG_S = 1.0   # search window for the mic -> laser offset

# Candidate filename templates. {id} is substituted with several id formats.
LASER_PATTERNS = ["Laser_{id}.wav", "laser_{id}.wav", "{id}.wav", "matrix_{id}.wav"]
MIC_PATTERNS = ["microphone_{id}.wav", "Mic_{id}.wav", "mic_{id}.wav",
                "Audio_{id}.wav", "{id}.wav", "Recording_{id}.wav"]


# ---------------------------------------------------------------------------
# file resolution
# ---------------------------------------------------------------------------

def id_variants(rec_id):
    """'007' -> ['007', '7', '07']  so we match whatever convention was used."""
    s = str(rec_id)
    out = [s]
    stripped = s.lstrip("0") or "0"
    for v in (stripped, stripped.zfill(2), stripped.zfill(3)):
        if v not in out:
            out.append(v)
    return out


NUM_RE = re.compile(r"(\d+)")


def build_index(directory):
    """Index the wav files in a directory two ways: by exact name, and by the
    number embedded in the name.

    The numeric index is what makes this robust to naming conventions we have
    never seen. Whatever the prefix is - Mic_007, microphone007, rec_007_mic,
    007 - the last run of digits in the stem is taken as the recording id. Only
    if that is ambiguous do we need the explicit patterns.
    """
    paths = sorted(glob(os.path.join(directory, "*.wav")))
    by_name = {os.path.basename(p).lower(): p for p in paths}

    by_num = {}
    for p in paths:
        stem = os.path.splitext(os.path.basename(p))[0]
        nums = NUM_RE.findall(stem)
        if not nums:
            continue
        by_num.setdefault(int(nums[-1]), []).append(p)

    return {"paths": paths, "by_name": by_name, "by_num": by_num}


def resolve(directory, rec_id, patterns, index=None):
    """Find the file for one recording, by name pattern then by embedded number."""
    if index is None:
        index = build_index(directory)

    for pat in patterns:
        for v in id_variants(rec_id):
            hit = index["by_name"].get(pat.format(id=v).lower())
            if hit:
                return hit

    # Fallback: match on the number in the filename, whatever surrounds it.
    cands = index["by_num"].get(int(rec_id), [])
    if len(cands) == 1:
        return cands[0]
    return None


def describe_dir(label, directory, index, n=8):
    print(f"\n{label}: {len(index['paths'])} wav files in {directory}")
    for p in index["paths"][:n]:
        print(f"    {os.path.basename(p)}")
    if len(index["paths"]) > n:
        print(f"    ... and {len(index['paths']) - n} more")
    dupes = {k: v for k, v in index["by_num"].items() if len(v) > 1}
    if dupes:
        k = sorted(dupes)[0]
        print(f"  warning: {len(dupes)} numbers map to several files, e.g. {k} -> "
              f"{[os.path.basename(x) for x in dupes[k]]}")


# ---------------------------------------------------------------------------
# signal helpers
# ---------------------------------------------------------------------------

def read_wav(path):
    """Return (float64 mono signal in [-1, 1], sample_rate)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, x = wavfile.read(path)
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float64)
    return x, float(sr)


def bandpass(x, sr, f_lo=F_LO, f_hi=F_HI):
    """Band-limit to the range the laser can actually carry."""
    nyq = sr / 2.0
    hi = min(f_hi, nyq * 0.98)
    lo = min(f_lo, hi * 0.5)
    sos = signal.butter(4, [lo / nyq, hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x)


def envelope(x, sr, out_len):
    """Smoothed energy envelope resampled to `out_len` samples at ENV_RATE."""
    xb = bandpass(x, sr)
    p = xb ** 2
    sos = signal.butter(2, min(ENV_LP, sr / 2 * 0.9) / (sr / 2), btype="low", output="sos")
    p = np.maximum(signal.sosfiltfilt(sos, p), 0.0)
    env = np.sqrt(p)
    env = signal.resample(env, out_len)
    env = np.maximum(env, 0.0)
    # Zero-phase filtering rings at both edges; that transient is large enough
    # to dominate the z-normalisation and drag the cross-correlation peak.
    guard = int(0.25 * ENV_RATE)
    if out_len > 4 * guard:
        env[:guard] = np.median(env[guard:2 * guard])
        env[-guard:] = np.median(env[-2 * guard:-guard])
    return env


def znorm(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def best_lag(env_ref, env_tgt, max_lag):
    """Lag in envelope samples that best aligns env_tgt to env_ref, plus peak r."""
    a, b = znorm(env_ref), znorm(env_tgt)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    xc = signal.correlate(b, a, mode="full") / n
    lags = signal.correlation_lags(len(b), len(a), mode="full")
    keep = np.abs(lags) <= max_lag
    xc, lags = xc[keep], lags[keep]
    k = int(np.argmax(xc))
    return int(lags[k]), float(xc[k])


# ---------------------------------------------------------------------------
# per-recording analysis
# ---------------------------------------------------------------------------

def analyse(row, laser_path, mic_path):
    out = {
        "recording_id": row.recording_id,
        "speaker_id": row.speaker_id,
        "laser_file": os.path.basename(laser_path) if laser_path else "",
        "mic_file": os.path.basename(mic_path) if mic_path else "",
    }
    if laser_path is None or mic_path is None:
        out["status"] = "missing_file"
        return out

    laser, sr_l = read_wav(laser_path)
    mic, sr_m = read_wav(mic_path)

    dur_l, dur_m = len(laser) / sr_l, len(mic) / sr_m
    out.update(laser_sr=sr_l, mic_sr=sr_m,
               laser_dur_s=round(dur_l, 3), mic_dur_s=round(dur_m, 3),
               dur_mismatch_s=round(abs(dur_l - dur_m), 3))

    # --- laser health -----------------------------------------------------
    peak = float(np.max(np.abs(laser))) if len(laser) else 0.0
    out["laser_peak"] = round(peak, 4)
    out["laser_clip_frac"] = round(float(np.mean(np.abs(laser) > 0.999)), 5)

    lb = bandpass(laser, sr_l)
    tot = float(np.sum(laser ** 2)) + 1e-20
    out["laser_inband_ratio"] = round(float(np.sum(lb ** 2)) / tot, 4)

    if peak < 1e-6 or np.std(laser) < 1e-8:
        out["status"] = "dead_laser"
        return out

    # --- envelopes on a shared clock --------------------------------------
    n_env = int(round(min(dur_l, dur_m) * ENV_RATE))
    if n_env < int(ENV_RATE):  # under one second of overlap
        out["status"] = "too_short"
        return out

    env_l = envelope(laser[: int(min(dur_l, dur_m) * sr_l)], sr_l, n_env)
    env_m = envelope(mic[: int(min(dur_l, dur_m) * sr_m)], sr_m, n_env)

    lag, r = best_lag(env_m, env_l, int(MAX_LAG_S * ENV_RATE))
    out["offset_s"] = round(lag / ENV_RATE, 4)     # positive: laser lags the mic
    out["sync_corr"] = round(r, 4)
    out["sync_corr_at_zero"] = round(float(np.corrcoef(znorm(env_m), znorm(env_l))[0, 1]), 4)

    # --- does the label land on actual speech in the mic? -----------------
    if bool(row.has_keyword):
        i0 = int(row.start_s * ENV_RATE)
        i1 = int(row.end_s * ENV_RATE)
        i0, i1 = max(0, i0), min(n_env, max(i1, i0 + 1))
        in_win = float(env_m[i0:i1].mean())
        floor = float(np.percentile(env_m, 10))
        median = float(np.median(env_m))
        out["label_mic_energy_ratio"] = round(in_win / (median + 1e-12), 3)
        out["label_above_floor"] = bool(in_win > floor * 2.0)

        # same question of the laser track, after applying the measured offset
        j0, j1 = max(0, i0 + lag), min(n_env, max(i1 + lag, i0 + lag + 1))
        out["label_laser_energy_ratio"] = round(
            float(env_l[j0:j1].mean()) / (float(np.median(env_l)) + 1e-12), 3
        )

    out["status"] = "ok"
    return out


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="metadata_clean.csv")
    ap.add_argument("--laser-dir", required=True)
    ap.add_argument("--mic-dir", required=True)
    ap.add_argument("--out", default="qc_report.csv")
    ap.add_argument("--plot-dir", default=None)
    ap.add_argument("--n-plots", type=int, default=12)
    ap.add_argument("--list-only", action="store_true",
                    help="resolve filenames and stop, so naming can be verified first")
    args = ap.parse_args()

    meta = pd.read_csv(args.meta, dtype={"recording_id": str, "speaker_id": str})
    li, mi = build_index(args.laser_dir), build_index(args.mic_dir)
    describe_dir("laser", args.laser_dir, li)
    describe_dir("mic", args.mic_dir, mi)

    pairs = []
    for row in meta.itertuples():
        pairs.append((row,
                      resolve(args.laser_dir, row.recording_id, LASER_PATTERNS, li),
                      resolve(args.mic_dir, row.recording_id, MIC_PATTERNS, mi)))

    no_laser = [r.recording_id for r, l, _ in pairs if l is None]
    no_mic = [r.recording_id for r, _, m in pairs if m is None]
    n_ok = sum(1 for _, l, m in pairs if l is not None and m is not None)

    print(f"\nresolved {n_ok}/{len(pairs)} recordings")
    if no_laser:
        print(f"  no laser file ({len(no_laser)}): {', '.join(no_laser[:30])}")
    if no_mic:
        print(f"  no mic file ({len(no_mic)}): {', '.join(no_mic[:30])}")

    # Files on disk that no metadata row claims - usually a numbering mismatch.
    claimed = {p for _, l, m in pairs for p in (l, m) if p}
    orphan_l = [os.path.basename(p) for p in li["paths"] if p not in claimed]
    orphan_m = [os.path.basename(p) for p in mi["paths"] if p not in claimed]
    if orphan_l:
        print(f"  laser files not referenced by any row ({len(orphan_l)}): "
              f"{', '.join(orphan_l[:15])}")
    if orphan_m:
        print(f"  mic files not referenced by any row ({len(orphan_m)}): "
              f"{', '.join(orphan_m[:15])}")

    if args.list_only or n_ok == 0:
        for row, l, m in pairs[:10]:
            print(f"  {row.recording_id}: laser={os.path.basename(l) if l else 'NOT FOUND'}"
                  f"  mic={os.path.basename(m) if m else 'NOT FOUND'}")
        if n_ok == 0:
            print("\nNothing resolved. Check the sample filenames printed above and, "
                  "if the numbering itself differs between the two folders, add a "
                  "template to MIC_PATTERNS at the top of this file.")
        return
    pairs = [(r, l, m) for r, l, m in pairs if l is not None and m is not None]

    rows = []
    for k, (row, lp, mp) in enumerate(pairs, 1):
        try:
            rows.append(analyse(row, lp, mp))
        except Exception as exc:  # keep going, record the failure
            rows.append({"recording_id": row.recording_id,
                         "speaker_id": row.speaker_id,
                         "status": f"error: {exc}"})
        if k % 25 == 0:
            print(f"  {k}/{len(pairs)}")

    qc = pd.DataFrame(rows)
    qc.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}\n")

    print("status counts:")
    print(qc.status.value_counts().to_string(), "\n")

    ok = qc[qc.status == "ok"]
    if len(ok):
        print("mic -> laser offset by speaker (seconds):")
        print(ok.groupby("speaker_id").offset_s.describe()[
            ["count", "mean", "std", "min", "max"]].round(3).to_string(), "\n")
        print("sync correlation by speaker:")
        print(ok.groupby("speaker_id").sync_corr.describe()[
            ["count", "mean", "std", "min"]].round(3).to_string(), "\n")

        weak = ok[ok.sync_corr < 0.3]
        print(f"{len(weak)} recordings with sync_corr < 0.3 (suspect laser signal):")
        if len(weak):
            print("  " + ", ".join(weak.recording_id.tolist()[:40]))

        if "label_above_floor" in ok:
            bad = ok[(ok.label_above_floor == False)]
            print(f"\n{len(bad)} labels landing on near-silence in the mic:")
            if len(bad):
                print("  " + ", ".join(bad.recording_id.tolist()[:40]))

        big = ok[ok.offset_s.abs() > 0.05]
        print(f"\n{len(big)} recordings with |offset| > 50 ms")

    if args.plot_dir:
        make_plots(meta, pairs, qc, args.plot_dir, args.n_plots)


def make_plots(meta, pairs, qc, plot_dir, n_plots):
    """Envelope overlays for a sample of recordings, for eyeball verification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(plot_dir, exist_ok=True)
    ok_ids = set(qc[qc.status == "ok"].recording_id)
    chosen = [p for p in pairs if p[0].recording_id in ok_ids][:n_plots]

    for row, lp, mp in chosen:
        laser, sr_l = read_wav(lp)
        mic, sr_m = read_wav(mp)
        n_env = int(round(min(len(laser) / sr_l, len(mic) / sr_m) * ENV_RATE))
        env_l = znorm(envelope(laser, sr_l, n_env))
        env_m = znorm(envelope(mic, sr_m, n_env))
        t = np.arange(n_env) / ENV_RATE

        fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
        ax[0].plot(t, env_m, lw=0.8, label="microphone")
        ax[0].plot(t, env_l, lw=0.8, alpha=0.75, label="laser")
        ax[0].set_ylabel("envelope (z)")
        ax[0].legend(loc="upper right")
        ax[0].set_title(f"recording {row.recording_id}  speaker {row.speaker_id}  "
                        f"noise={row.noise_type}")

        f, tt, Sxx = signal.spectrogram(bandpass(laser, sr_l), sr_l,
                                        nperseg=256, noverlap=192)
        ax[1].pcolormesh(tt, f, 10 * np.log10(Sxx + 1e-14), shading="auto")
        ax[1].set_ylim(0, F_HI)
        ax[1].set_ylabel("Hz (laser)")
        ax[1].set_xlabel("time (s)")

        if bool(row.has_keyword):
            for a in ax:
                a.axvspan(row.start_s, row.end_s, color="red", alpha=0.25)

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, f"{row.recording_id}.png"), dpi=90)
        plt.close(fig)

    print(f"\nwrote {len(chosen)} plots to {plot_dir}")


if __name__ == "__main__":
    main()
