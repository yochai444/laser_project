"""
compare_variants.py
-------------------
Scores the six WAV variants produced by extract_variants.m against the
synchronised microphone recording of the same take, and ranks them.

The metric is the one that already told us something true about this corpus:
per-frequency correlation between the microphone's log-magnitude trajectory and
the laser's. The current extraction (X axis, 100 Hz highpass) sits at about 0.30
mean coherence. If another variant clears roughly 0.40, re-extracting all 369
recordings is worth the compute; if they are all within noise of each other, the
axis and cutoff hypotheses are dead and the 29-point gap lies elsewhere.

Usage:
    python compare_variants.py --variant-dir ./variants \
        --mic ./microphone/microphone_017.wav
"""

import argparse
import os
from glob import glob

import numpy as np
from scipy import signal
from scipy.io import wavfile

N_FFT = 512
HOP = 160
F_CEILING = 1600.0
SPEECH_BAND = (150.0, 700.0)   # the measured coherent band for this corpus
UPPER_BAND = (700.0, 1400.0)   # sampled but currently empty; the open question


def read_wav(path):
    sr, x = wavfile.read(path)
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    return x.astype(np.float64), float(sr)


def log_mag(x, sr):
    f, _, Z = signal.stft(x, sr, nperseg=N_FFT, noverlap=N_FFT - HOP,
                          window="hann", boundary=None, padded=False)
    keep = f <= F_CEILING
    return f[keep], np.log1p(np.abs(Z[keep]))


def per_bin_corr(A, B):
    n = min(A.shape[1], B.shape[1])
    A, B = A[:, :n], B[:, :n]
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1)) + 1e-20
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-dir", default="./variants")
    ap.add_argument("--mic", required=True, help="microphone wav for the same take")
    args = ap.parse_args()

    mic, sr_m = read_wav(args.mic)
    f_m, M = log_mag(mic, sr_m)

    paths = sorted(glob(os.path.join(args.variant_dir, "*.wav")))
    if not paths:
        raise SystemExit(f"no wav files in {args.variant_dir}")

    band = (f_m >= SPEECH_BAND[0]) & (f_m <= SPEECH_BAND[1])
    upper = (f_m >= UPPER_BAND[0]) & (f_m <= UPPER_BAND[1])
    noise = f_m > 1600.0  # above Nyquist for this camera; pure control

    results = []
    for p in paths:
        lz, sr_l = read_wav(p)
        f_l, L = log_mag(lz, sr_l)
        if L.shape[0] != M.shape[0]:
            print(f"  skipping {os.path.basename(p)}: "
                  f"{L.shape[0]} bins vs {M.shape[0]} - sample rates differ")
            continue
        r = per_bin_corr(M, L)
        results.append({
            "name": os.path.basename(p).replace("variant_", "").replace(".wav", ""),
            "r_band": float(np.mean(r[band])),
            "r_upper": float(np.mean(r[upper])) if upper.any() else float("nan"),
            "r_peak": float(np.max(r[band])),
            "peak_hz": float(f_l[band][int(np.argmax(r[band]))]),
            "r_noise": float(np.mean(r[noise])) if noise.any() else float("nan"),
            "profile": r,
            "freqs": f_l,
        })

    if not results:
        raise SystemExit("nothing comparable was produced")

    results.sort(key=lambda d: -d["r_band"])
    base = next((d for d in results if d["name"] in ("X_hp100", "baseline")), None)

    print(f"\nreference microphone: {os.path.basename(args.mic)}")
    print(f"coherence averaged over {SPEECH_BAND[0]:.0f}-{SPEECH_BAND[1]:.0f} Hz\n")
    print(f"{'variant':<14} {'150-700':>9} {'700-1400':>10} {'vs base':>9} "
          f"{'peak r':>8} {'at Hz':>7}")
    print("-" * 62)
    for d in results:
        delta = (f"{d['r_band'] - base['r_band']:+.3f}"
                 if base and d is not base else ("baseline" if d is base else "-"))
        print(f"{d['name']:<14} {d['r_band']:9.3f} {d['r_upper']:10.3f} "
              f"{delta:>9} {d['r_peak']:8.3f} {d['peak_hz']:7.0f}")

    best = results[0]
    best_up = max(results, key=lambda d: (d["r_upper"] if np.isfinite(d["r_upper"])
                                          else -1))
    print(f"\nbest in 150-700 Hz : {best['name']} at {best['r_band']:.3f}")
    print(f"best in 700-1400 Hz: {best_up['name']} at {best_up['r_upper']:.3f}")
    if base and np.isfinite(base["r_upper"]):
        gain_up = best_up["r_upper"] - base["r_upper"]
        print(f"\n700-1400 Hz is the band that was sampled but came back empty.")
        if gain_up < 0.03:
            print("No setting recovers it. That range is lost in the estimator "
                  "itself, not\nin any filtering step, and no parameter here "
                  "reaches it.")
        else:
            print(f"One setting lifts it by {gain_up:+.3f}. Confirm on two more "
                  "recordings\nfrom other speakers before re-extracting the "
                  "whole corpus.")
    if base:
        gain = best["r_band"] - base["r_band"]
        print(f"\n150-700 Hz, the band that already works: best is "
              f"{gain:+.3f} against the current extraction.")

    print("\ncoherence profile, 150-900 Hz:\n")
    freqs = results[0]["freqs"]
    show = np.where((freqs >= 150) & (freqs <= 900))[0][::2]
    print(f"{'Hz':>6} " + " ".join(f"{d['name']:>9}" for d in results))
    for i in show:
        print(f"{freqs[i]:6.0f} " + " ".join(f"{d['profile'][i]:9.3f}" for d in results))


if __name__ == "__main__":
    main()
