"""
band_coherence.py
-----------------
Measures, frequency bin by frequency bin, how well the laser signal tracks the
synchronised microphone. The answer tells you which part of the spectrum the
optical channel actually carries and which part is noise.

Why this matters: V3/V4 feed the model everything from 50 to 1500 Hz. The
spectrograms show laser energy concentrated below ~600 Hz, with what looks like
a uniform noise floor above it. Per-frequency CMVN then rescales those noise
bins to the same variance as the real signal bins, which actively amplifies
them. With only ~356 positives, that is a direct invitation to overfit noise.

Rather than guessing a new F_MAX, this measures it. For each frequency bin, the
log-magnitude trajectory over time is extracted from both channels and
correlated. A bin that carries speech shows the laser rising and falling with
the microphone. A bin that carries noise shows no relationship.

The output is a correlation-vs-frequency profile per speaker, and a suggested
F_MAX: the highest frequency at which coherence is still above the floor.

Usage:
    python band_coherence.py --laser-dir ./laser --mic-dir ./microphone \
        --meta metadata_clean.csv --out band_coherence.csv --plot band_coherence.png

    # quick pass while iterating
    python band_coherence.py ... --max-per-speaker 10
"""

import argparse
import os
from glob import glob

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

SR_EXPECTED = 16000
N_FFT = 512
HOP = 160
F_CEILING = 1600.0      # a little past the laser Nyquist, so the cliff is visible
COHERENCE_FLOOR = 0.10  # below this a bin is treated as carrying no speech


def read_wav(path):
    sr, x = wavfile.read(path)
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    return x.astype(np.float64), float(sr)


def log_mag(x, sr):
    """[F, T] log-magnitude spectrogram, restricted to the band of interest."""
    f, _, Z = signal.stft(x, sr, nperseg=N_FFT, noverlap=N_FFT - HOP,
                          window="hann", boundary=None, padded=False)
    keep = f <= F_CEILING
    return f[keep], np.log1p(np.abs(Z[keep]))


def per_bin_correlation(A, B):
    """Pearson r between matching rows of two [F, T] matrices."""
    n = min(A.shape[1], B.shape[1])
    A, B = A[:, :n], B[:, :n]
    A = A - A.mean(axis=1, keepdims=True)
    B = B - B.mean(axis=1, keepdims=True)
    num = (A * B).sum(axis=1)
    den = np.sqrt((A ** 2).sum(axis=1) * (B ** 2).sum(axis=1)) + 1e-20
    return num / den


def find_files(directory):
    idx = {}
    for p in sorted(glob(os.path.join(directory, "*.wav"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        nums = [c for c in stem.replace("_", " ").split() if c.isdigit()]
        if not nums:
            nums = ["".join(ch for ch in stem if ch.isdigit())]
        if nums and nums[-1]:
            idx[int(nums[-1])] = p
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="metadata_clean.csv")
    ap.add_argument("--laser-dir", required=True)
    ap.add_argument("--mic-dir", required=True)
    ap.add_argument("--out", default="band_coherence.csv")
    ap.add_argument("--plot", default="band_coherence.png")
    ap.add_argument("--max-per-speaker", type=int, default=0,
                    help="0 = use every recording")
    ap.add_argument("--offset-s", type=float, default=0.0,
                    help="shift the laser by this many seconds before comparing")
    args = ap.parse_args()

    meta = pd.read_csv(args.meta, dtype={"recording_id": str, "speaker_id": str})
    meta = meta[~meta.exclude]
    if args.max_per_speaker:
        meta = meta.groupby("speaker_id").head(args.max_per_speaker)

    li, mi = find_files(args.laser_dir), find_files(args.mic_dir)

    freqs = None
    records = []

    for k, row in enumerate(meta.itertuples(), 1):
        rid = int(row.recording_id)
        if rid not in li or rid not in mi:
            continue
        try:
            laser, sr_l = read_wav(li[rid])
            mic, sr_m = read_wav(mi[rid])
        except Exception:
            continue

        if args.offset_s:
            shift = int(round(args.offset_s * sr_l))
            laser = np.roll(laser, -shift)

        f_l, L = log_mag(laser, sr_l)
        f_m, M = log_mag(mic, sr_m)
        if L.shape[0] != M.shape[0]:
            continue

        freqs = f_l
        records.append({"recording_id": row.recording_id,
                        "speaker_id": row.speaker_id,
                        "r": per_bin_correlation(M, L)})
        if k % 50 == 0:
            print(f"  {k}/{len(meta)}")

    if not records:
        print("no recordings processed; check the directories")
        return

    R = np.vstack([rec["r"] for rec in records])
    spk = np.array([rec["speaker_id"] for rec in records])

    out = pd.DataFrame({"freq_hz": freqs,
                        "r_median_all": np.median(R, axis=0),
                        "r_mean_all": R.mean(axis=0)})
    for s in sorted(set(spk)):
        out[f"r_median_{s}"] = np.median(R[spk == s], axis=0)
    out.to_csv(args.out, index=False)

    print(f"\nprocessed {len(records)} recordings, wrote {args.out}\n")
    print("coherence between microphone and laser, by frequency:\n")
    print(f"{'Hz':>7} {'all':>7} " + " ".join(f"{s:>6}" for s in sorted(set(spk))))
    step = max(1, len(freqs) // 24)
    for i in range(0, len(freqs), step):
        line = f"{freqs[i]:7.0f} {np.median(R[:, i]):7.3f} "
        line += " ".join(f"{np.median(R[spk == s, i]):6.3f}" for s in sorted(set(spk)))
        bar = "#" * int(max(0, np.median(R[:, i])) * 40)
        print(line + "  " + bar)

    med = np.median(R, axis=0)
    usable = freqs[(med >= COHERENCE_FLOOR) & (freqs > 0)]
    if len(usable):
        print(f"\ncoherent band (median r >= {COHERENCE_FLOOR}): "
              f"{usable.min():.0f} - {usable.max():.0f} Hz")
        print(f"suggested F_MIN / F_MAX for the model: "
              f"{max(50.0, usable.min()):.0f} / {usable.max():.0f} Hz")
    else:
        print("\nno bin clears the coherence floor - check the offset and the pairing")

    print("\nper speaker, highest coherent frequency:")
    for s in sorted(set(spk)):
        m = np.median(R[spk == s], axis=0)
        u = freqs[(m >= COHERENCE_FLOOR) & (freqs > 0)]
        print(f"  {s}: {u.max():6.0f} Hz  (peak r {m.max():.3f} at {freqs[np.argmax(m)]:.0f} Hz)"
              if len(u) else f"  {s}: none")

    if args.plot:
        make_plot(freqs, R, spk, args.plot)


def make_plot(freqs, R, spk, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5))
    for s in sorted(set(spk)):
        ax.plot(freqs, np.median(R[spk == s], axis=0), lw=1.2, label=f"speaker {s}")
    ax.plot(freqs, np.median(R, axis=0), lw=2.5, color="black", label="all")
    ax.axhline(COHERENCE_FLOOR, ls="--", lw=0.8, color="grey")
    ax.axvline(1467, ls=":", lw=1.0, color="red")
    ax.text(1467, ax.get_ylim()[1], " laser Nyquist", va="top", fontsize=8, color="red")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("median mic-laser correlation")
    ax.set_title("Which frequencies does the optical channel actually carry?")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
