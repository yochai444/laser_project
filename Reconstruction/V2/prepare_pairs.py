"""
Stage 1 — pair QC and alignment.

Runs ONCE over the corpus and writes pairs.json containing, for every recording:
  - the estimated mic->optical delay in samples (band-limited GCC-PHAT)
  - a confidence score for that delay estimate
  - the peak magnitude-squared coherence in the speech band
  - the speaker id and keyword onset from the metadata

Nothing downstream recomputes alignment. The old dataset.py did a 2^20-point FFT
inside __getitem__, i.e. once per sample per epoch; that alone dominated training time.

Usage:
    python prepare_pairs.py --root dataset --out pairs.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, coherence

# Band over which the optical and acoustic channels are expected to agree.
# Below ~150 Hz the optical signal is dominated by slow surface drift, not speech;
# above ~1400 Hz it is above the optical Nyquist limit set by the 3 kHz frame rate.
BAND_LO_HZ = 150.0
BAND_HI_HZ = 1400.0
MAX_DELAY_SEC = 0.6          # covers the pre-handshake drift reported in the thesis
COHERENCE_NPERSEG = 4096


def read_wav(path, expect_sr=None):
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    if expect_sr is not None and sr != expect_sr:
        raise ValueError(f"{path}: expected {expect_sr} Hz, got {sr} Hz")
    return x, sr


def bandpass(x, sr, lo=BAND_LO_HZ, hi=BAND_HI_HZ):
    hi = min(hi, 0.45 * sr)
    sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
    return sosfiltfilt(sos, x)


def gcc_phat(ref, sig, sr, max_delay_sec=MAX_DELAY_SEC):
    """
    Band-limited GCC-PHAT delay estimate.

    PHAT whitening is used instead of plain cross-correlation because the two
    channels have very different spectral tilts; a raw cross-correlation is
    dominated by whichever channel has more low-frequency energy and, on
    low-coherence pairs, produces an essentially random peak.

    Returns (delay_samples, confidence) where confidence is peak / median(|cc|).
    A confidence near 1 means the peak is indistinguishable from noise.
    """
    n = min(len(ref), len(sig))
    ref, sig = ref[:n], sig[:n]
    n_fft = 1 << int(np.ceil(np.log2(2 * n)))

    R = np.fft.rfft(ref, n_fft)
    S = np.fft.rfft(sig, n_fft)
    cross = S * np.conj(R)
    mag = np.abs(cross)
    cross = np.divide(cross, mag, out=np.zeros_like(cross), where=mag > 1e-12)

    cc = np.fft.irfft(cross, n_fft)
    max_lag = int(max_delay_sec * sr)
    cc = np.concatenate([cc[-max_lag:], cc[:max_lag + 1]])

    peak = int(np.argmax(np.abs(cc)))
    delay = peak - max_lag
    conf = float(np.abs(cc[peak]) / (np.median(np.abs(cc)) + 1e-12))
    return delay, conf


def band_coherence(a, b, sr):
    nper = min(COHERENCE_NPERSEG, len(a))
    f, cxy = coherence(a, b, fs=sr, nperseg=nper)
    sel = (f >= BAND_LO_HZ) & (f <= BAND_HI_HZ)
    if not sel.any():
        return 0.0, 0.0, 0.0
    return float(cxy[sel].max()), float(np.median(cxy[sel])), float(f[sel][np.argmax(cxy[sel])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset", help="folder containing laser/ and microphone/")
    ap.add_argument("--metadata", default="Recordings Metadata.json")
    ap.add_argument("--out", default="pairs.json")
    args = ap.parse_args()

    root = Path(args.root)
    meta = json.load(open(args.metadata, encoding="utf-8"))

    records = {}
    laser_files = sorted((root / "laser").glob("Laser_*.wav"))
    if not laser_files:
        raise SystemExit(f"no Laser_*.wav under {root/'laser'}")

    for lf in laser_files:
        rid = lf.stem.split("_")[1]
        mf = root / "microphone" / f"microphone_{rid}.wav"
        if not mf.exists():
            print(f"[skip] {rid}: no matching microphone file")
            continue

        opt, sr = read_wav(lf)
        mic, sr2 = read_wav(mf, expect_sr=sr)

        n = min(len(opt), len(mic))
        opt, mic = opt[:n], mic[:n]

        opt_b = bandpass(opt, sr)
        mic_b = bandpass(mic, sr)

        delay, conf = gcc_phat(opt_b, mic_b, sr)
        msc_max, msc_med, msc_f = band_coherence(opt_b, mic_b, sr)

        m = meta.get(rid, {})
        records[rid] = {
            "laser": str(lf),
            "microphone": str(mf),
            "sample_rate": sr,
            "n_samples": int(n),
            "delay_samples": int(delay),
            "delay_ms": round(1000.0 * delay / sr, 2),
            "delay_confidence": round(conf, 2),
            "msc_max": round(msc_max, 4),
            "msc_median": round(msc_med, 4),
            "msc_peak_hz": round(msc_f, 1),
            "speaker": m.get("RecorderID"),
            "keyword_start": m.get("StartTime"),
            "keyword_duration": m.get("Duration"),
            "noise": (m.get("Noise") or "").strip(),
        }
        print(f"{rid}  msc={msc_max:.3f}@{msc_f:6.0f}Hz  delay={delay:+6d} ({conf:5.1f}x)")

    json.dump(records, open(args.out, "w"), indent=1)

    msc = np.array([r["msc_max"] for r in records.values()])
    print(f"\nwrote {args.out}: {len(records)} pairs")
    for t in (0.25, 0.35, 0.45, 0.55):
        print(f"  msc_max > {t:.2f}: {(msc > t).sum():3d} ({100 * (msc > t).mean():.0f}%)")


if __name__ == "__main__":
    main()
