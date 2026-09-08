"""
Microphone-only and optical-gate baselines.

This reproduces the ceiling measurement reported in the Experiments chapter. Any
optically guided enhancement result should be reported against these numbers,
because two of the three require no training at all and one of them requires no
optical channel at all.

Three methods are evaluated:

  1  SPECTRAL SUBTRACTION — microphone only, no training, no optical channel.
     The noise magnitude spectrum is estimated from the quietest 10% of frames
     and subtracted with an over-subtraction factor and a spectral floor.
     This is the baseline that matters: it is the honest comparison.

  2  OPTICAL GATE — the CEILING of optical guidance. The microphone is gated
     directly by the optical activity envelope, with no learning involved. No
     learned model using the optical channel for masking can exceed this by much,
     so if it is low, the limitation is the channel and not the architecture.

  3  MICROPHONE GATE — the same gating driven by a microphone-derived envelope.
     Reported as a reference only. It is CIRCULAR: the same microphone energy
     drives both the gate and the evaluation VAD, so it is an upper bound on the
     gating approach rather than a result. Do not quote it as a method.

Evaluation metric: SNR gain, defined as the change in level on speech frames
minus the change in level on silence frames, both referenced to a VAD derived
from the microphone in its cleanest band. A method that simply lowers the volume
scores zero.

Only recordings containing BOTH speech and real acoustic noise are used. Noise
added digitally to a recorded file was never present in the room and therefore was
never absent from the optical channel, so it cannot test the hypothesis.

Usage:
    python spectral_baselines.py --root dataset --metadata "Recordings Metadata.json"
    python spectral_baselines.py --root dataset --write-audio 250 252
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, stft, istft, resample_poly

# ------------------------------------------------------------------ parameters

WORK_SR = 16000
NPERSEG, NOVERLAP = 1024, 768
FRAME_MS, HOP_MS = 32.0, 8.0

OPT_BAND = (150.0, 1400.0)      # where the optical channel carries speech
MIC_BAND = (300.0, 3000.0)      # where the microphone is most reliable for VAD

OVERSUB = 1.5                   # spectral subtraction over-subtraction factor
SPEC_FLOOR = 0.1                # spectral floor, prevents musical noise
GATE_FLOOR_DB = -14.0           # attenuation applied where the gate is closed


# ---------------------------------------------------------------------- io

def load_mono(path, target_sr=WORK_SR):
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    if sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g)
    return x


def bandpass(x, lo, hi, sr=WORK_SR):
    hi = min(hi, 0.45 * sr)
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x)


def frame_energy_db(x, sr=WORK_SR):
    fr, hop = int(FRAME_MS * sr / 1000), int(HOP_MS * sr / 1000)
    n = 1 + max(0, (len(x) - fr) // hop)
    idx = np.arange(fr)[None, :] + hop * np.arange(n)[:, None]
    return 10 * np.log10(((x[idx] * np.hanning(fr)) ** 2).mean(axis=1) + 1e-12)


# ------------------------------------------------------------------- methods

def spectral_subtraction(mic, sr=WORK_SR, oversub=OVERSUB, floor=SPEC_FLOOR):
    """Boll-style magnitude spectral subtraction. Microphone only, no training."""
    f, t, Z = stft(mic, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    mag = np.abs(Z)

    frame_level = 20 * np.log10(mag.mean(axis=0) + 1e-12)
    quiet = frame_level < np.percentile(frame_level, 10)
    if quiet.sum() < 3:
        quiet = np.ones_like(frame_level, dtype=bool)
    noise = mag[:, quiet].mean(axis=1, keepdims=True)

    gain = np.clip((mag - oversub * noise) / (mag + 1e-12), floor, 1.0)
    _, y = istft(Z * gain, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return y


def envelope_gate(mic, score, sr=WORK_SR, floor_db=GATE_FLOOR_DB, smooth=7):
    """
    Gate the microphone with an arbitrary per-frame activity score. The score is
    smoothed, mapped to [0, 1] by its own 10th and 90th percentiles, and applied
    as a broadband gain. Deliberately the simplest thing that could work: if the
    score is informative, this will show it.
    """
    k = max(3, smooth | 1)
    a = np.convolve(np.pad(score, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:len(score)]
    lo, hi = np.percentile(a, 10), np.percentile(a, 90)
    g = np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    g = 10 ** (floor_db * (1 - g) / 20.0)

    f, t, Z = stft(mic, fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    gi = np.interp(np.arange(Z.shape[1]), np.linspace(0, Z.shape[1] - 1, len(g)), g)
    _, y = istft(Z * gi[None, :], fs=sr, nperseg=NPERSEG, noverlap=NOVERLAP)
    return y, g


# ---------------------------------------------------------------- evaluation

def mic_vad(mic, sr=WORK_SR):
    """Evaluation ground truth. Valid only where the microphone is trustworthy."""
    e = frame_energy_db(bandpass(mic, *MIC_BAND, sr=sr), sr)
    lo, hi = np.percentile(e, 5), np.percentile(e, 90)
    if hi - lo < 6.0:
        return None
    v = e > lo + 0.45 * (hi - lo)
    for _ in range(2):
        v = np.convolve(v.astype(float), np.ones(3) / 3, mode="same") > 0.5
    return v


def snr_gain(processed, mic, vad, sr=WORK_SR):
    """Change on speech frames minus change on silence frames, in dB."""
    em, ey = frame_energy_db(mic, sr), frame_energy_db(processed, sr)
    k = min(len(vad), len(em), len(ey))
    v, em, ey = vad[:k], em[:k], ey[:k]
    if not v.any() or v.all():
        return float("nan")
    return (ey[v].mean() - em[v].mean()) - (ey[~v].mean() - em[~v].mean())


# --------------------------------------------------------------------- main

def is_speech_and_noise(tag):
    t = (tag or "").strip().lower()
    if t in ("", "none"):
        return False                       # quiet recording
    if "only" in t or "no talking" in t or "no word" in t:
        return False                       # no speech to preserve
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--metadata", default="Recordings Metadata.json")
    ap.add_argument("--msc", default=None, help="optional pairs.json, to show coherence")
    ap.add_argument("--out", default="baselines.json")
    ap.add_argument("--write-audio", nargs="*", default=[])
    ap.add_argument("--audio-dir", default="baseline_audio")
    args = ap.parse_args()

    root = Path(args.root)
    meta = json.load(open(args.metadata, encoding="utf-8"))
    msc = {}
    if args.msc and os.path.exists(args.msc):
        msc = {k: v.get("msc_max") for k, v in json.load(open(args.msc, encoding="utf-8")).items()}

    want = set(args.write_audio)
    if want:
        os.makedirs(args.audio_dir, exist_ok=True)

    rows = []
    for rid, m in sorted(meta.items(), key=lambda z: int(z[0])):
        if not is_speech_and_noise(m.get("Noise")):
            continue
        mf = root / "microphone" / f"microphone_{rid}.wav"
        lf = root / "laser" / f"Laser_{rid}.wav"
        if not (mf.exists() and lf.exists()):
            continue

        mic, opt = load_mono(mf), load_mono(lf)
        n = min(len(mic), len(opt))
        mic, opt = mic[:n], opt[:n]

        vad = mic_vad(mic)
        if vad is None or vad.mean() < 0.05 or vad.mean() > 0.95:
            print(f"[skip] {rid}: no clear speech/silence contrast")
            continue

        ss = spectral_subtraction(mic)[:n]
        og, _ = envelope_gate(mic, frame_energy_db(bandpass(opt, *OPT_BAND)))
        mg, _ = envelope_gate(mic, frame_energy_db(bandpass(mic, *MIC_BAND)))

        row = dict(id=rid, speaker=m.get("RecorderID"), noise=(m.get("Noise") or "").strip(),
                   msc=msc.get(rid),
                   spectral_subtraction=snr_gain(ss, mic, vad),
                   optical_gate=snr_gain(og[:n], mic, vad),
                   mic_gate=snr_gain(mg[:n], mic, vad))
        rows.append(row)

        if rid in want:
            for name, sig in (("orig", mic), ("specsub", ss), ("optgate", og[:n])):
                sf.write(f"{args.audio_dir}/{name}_{rid}.wav",
                         sig / (np.abs(sig).max() + 1e-9), WORK_SR)

    if not rows:
        raise SystemExit("no recordings with both speech and real noise were found")

    json.dump(rows, open(args.out, "w"), indent=1)
    report(rows)


def report(rows):
    def col(k):
        return np.array([r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])])

    print("\n" + "=" * 78)
    print("SNR gain on recordings containing speech and REAL acoustic noise")
    print("=" * 78)
    print(f"{'id':>5} {'spk':>4} {'MSC':>6}  {'spec.sub':>9} {'opt-gate':>9} {'mic-gate':>9}   noise")
    for r in rows:
        m = f"{r['msc']:.3f}" if r.get("msc") is not None else "  —  "
        print(f"{r['id']:>5} {str(r['speaker']):>4} {m:>6}  "
              f"{r['spectral_subtraction']:>+9.2f} {r['optical_gate']:>+9.2f} "
              f"{r['mic_gate']:>+9.2f}   {r['noise']}")

    ss, og, mg = col("spectral_subtraction"), col("optical_gate"), col("mic_gate")
    print("-" * 78)
    print(f"{'MEAN':>5} {'':>4} {'':>6}  {ss.mean():>+9.2f} {og.mean():>+9.2f} {mg.mean():>+9.2f}")
    print(f"\n  n = {len(rows)} recordings")
    print(f"  spectral subtraction   {ss.mean():+.2f} dB   microphone only, no training")
    print(f"  optical gate           {og.mean():+.2f} dB   CEILING of optical guidance")
    print(f"  microphone gate        {mg.mean():+.2f} dB   circular, reference only")

    m = np.array([r["msc"] for r in rows if r.get("msc") is not None])
    if len(m) >= 3:
        o = np.array([r["optical_gate"] for r in rows if r.get("msc") is not None])
        print(f"\n  correlation between coherence and optical-gate gain: {np.corrcoef(m, o)[0,1]:.3f}")

    print("\n" + "=" * 78)
    if og.mean() < ss.mean():
        print(f"  The microphone-only method exceeds the optical ceiling by "
              f"{ss.mean() - og.mean():.2f} dB.")
        print("  No learned model using the optical channel for spectral masking can be")
        print("  expected to close that gap: the ceiling is a property of the channel,")
        print("  measured without any model at all. Report this alongside any optically")
        print("  guided result.")
    else:
        print("  The optical ceiling exceeds the microphone-only baseline. Optical")
        print("  guidance is worth pursuing; compare a trained model against both rows.")
    print("\n  Caveat: the microphone-gate column is circular — the same microphone")
    print("  energy drives both the gate and the evaluation VAD. It is an upper bound")
    print("  on gating, not a method. Only the spectral-subtraction column is a fair")
    print("  comparison.")


if __name__ == "__main__":
    main()
