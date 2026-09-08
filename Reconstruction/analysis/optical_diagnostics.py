"""
Optical channel diagnostics — three tests, no training, no GPU.

The proposed direction (use the optical channel as evidence about WHEN speech
occurs and at WHAT F0, while the microphone supplies the content) rests on two
assumptions that have not yet been measured:

    A1  the optical channel reliably indicates speech presence
    A2  the optical channel carries a usable F0 estimate

If A1 fails, masking cannot work. If A2 fails, harmonic extension cannot work.
Both are cheap to test directly. This script does that, and then applies the
simplest possible mask so the existing keyword detector can be run on the result.

Ground truth for speech presence comes from the microphone on QUIET recordings
only, where the microphone is trustworthy. Noisy recordings are used only in
test 3, where the question is what the mask does to them.

Usage:
    python optical_diagnostics.py --root dataset --metadata "Recordings Metadata.json"
    python optical_diagnostics.py --root dataset --write-masked 300 305 310

Outputs:
    diagnostics.json    per-recording numbers
    diagnostics.png     summary figure
    masked/*.wav        masked microphone audio for the recordings requested
"""

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfiltfilt, stft, istft, resample_poly

# ---------------------------------------------------------------- configuration

OPT_BAND = (150.0, 1400.0)     # where the optical channel is expected to carry speech
MIC_BAND = (300.0, 3000.0)     # where the microphone is most reliable for VAD ground truth
F0_RANGE = (70.0, 300.0)
FRAME_MS = 32.0
HOP_MS = 8.0
WORK_SR = 8000                 # enough for MIC_BAND, keeps the STFTs cheap


# ---------------------------------------------------------------- small helpers

def load_mono(path, target_sr):
    x, sr = sf.read(path, dtype="float64", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    if sr != target_sr:
        g = np.gcd(int(sr), int(target_sr))
        x = resample_poly(x, target_sr // g, sr // g)
    return x


def bandpass(x, sr, lo, hi):
    hi = min(hi, 0.45 * sr)
    return sosfiltfilt(butter(4, [lo, hi], btype="band", fs=sr, output="sos"), x)


def highpass(x, sr, f):
    return sosfiltfilt(butter(4, f, btype="high", fs=sr, output="sos"), x)


def frame_energy_db(x, sr, frame, hop):
    n = 1 + max(0, (len(x) - frame) // hop)
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx] * np.hanning(frame)[None, :]
    return 10 * np.log10((fr ** 2).mean(axis=1) + 1e-12), fr


def auc(scores, labels):
    """Rank-based ROC AUC. No sklearn dependency."""
    labels = np.asarray(labels).astype(bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mic_vad(mic, sr, frame, hop):
    """
    Energy VAD on the microphone, used as ground truth on quiet recordings only.
    Threshold sits between the noise floor (5th percentile) and the speech level
    (90th percentile), which is robust to how much silence a recording contains.
    """
    e, _ = frame_energy_db(bandpass(mic, sr, *MIC_BAND), sr, frame, hop)
    floor = np.percentile(e, 5)
    peak = np.percentile(e, 90)
    if peak - floor < 6.0:                       # no clear speech/silence contrast
        return None, e
    thr = floor + 0.45 * (peak - floor)
    v = e > thr
    # remove single-frame flicker
    for _ in range(2):
        v = np.convolve(v.astype(float), np.ones(3) / 3, mode="same") > 0.5
    return v, e


def optical_activity(opt, sr, frame, hop):
    """Speech-presence score from the optical channel: band energy, drift removed."""
    e, _ = frame_energy_db(bandpass(opt, sr, *OPT_BAND), sr, frame, hop)
    # subtract a slow baseline so the score is not dominated by session-level gain
    k = max(3, int(2.0 * sr / hop) | 1)
    base = np.convolve(np.pad(e, k // 2, mode="edge"), np.ones(k) / k, mode="valid")[:len(e)]
    return e - base


def estimate_f0(frames, sr, lo=F0_RANGE[0], hi=F0_RANGE[1]):
    """
    Autocorrelation F0 per frame, with a per-frame clarity score.
    Run on the high-passed signal, so for the optical channel the estimate comes
    from harmonic spacing rather than from the drift-dominated fundamental itself.
    """
    n = frames.shape[1]
    spec = np.fft.rfft(frames, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    ac0 = ac[:, :1].copy()
    ac0[ac0 <= 0] = 1e-12
    ac = ac / ac0

    lag_lo, lag_hi = int(sr / hi), min(int(sr / lo), n - 1)
    if lag_hi <= lag_lo:
        return np.full(len(frames), np.nan), np.zeros(len(frames))
    seg = ac[:, lag_lo:lag_hi]
    k = np.argmax(seg, axis=1)
    clarity = seg[np.arange(len(seg)), k]
    f0 = sr / (lag_lo + k)
    f0[clarity < 0.25] = np.nan
    return f0, clarity


def apply_mask(mic, sr, activity, hop, floor_db=-18.0, expand_frames=2):
    """
    Test 3, deliberately the simplest thing that could work: gate the microphone
    with the optical activity curve. No learning, no F0, no harmonic extension.
    The point is to find out whether the optical channel is informative at all.
    """
    a = activity.copy()
    if expand_frames:                          # dilate so onsets are not clipped
        k = 2 * expand_frames + 1
        a = np.convolve(np.pad(a, expand_frames, mode="edge"), np.ones(k), mode="valid")[:len(activity)]
        a = np.maximum(a, activity)
    lo, hi = np.percentile(a, 10), np.percentile(a, 90)
    g = np.clip((a - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    g = 10 ** (floor_db * (1 - g) / 20.0)      # 1.0 on speech, floor_db on silence

    nper = int(0.032 * sr)
    f, t, Z = stft(mic, fs=sr, nperseg=nper, noverlap=nper - hop)
    gi = np.interp(np.arange(Z.shape[1]), np.linspace(0, Z.shape[1] - 1, len(g)), g)
    _, y = istft(Z * gi[None, :], fs=sr, nperseg=nper, noverlap=nper - hop)
    return y[:len(mic)], g


# ---------------------------------------------------------------- main analysis

def analyse(rid, mic_path, opt_path, meta, sr, frame, hop):
    mic = load_mono(mic_path, sr)
    opt = load_mono(opt_path, sr)
    n = min(len(mic), len(opt))
    mic, opt = mic[:n], opt[:n]

    noise_tag = (meta.get("Noise") or "").strip().lower()
    quiet = noise_tag in ("", "none")

    act = optical_activity(opt, sr, frame, hop)
    vad, mic_e = mic_vad(mic, sr, frame, hop)

    row = {
        "id": rid,
        "speaker": meta.get("RecorderID"),
        "quiet": quiet,
        "noise": noise_tag,
        "keyword_start": meta.get("StartTime"),
    }

    # --- Test 1: does optical energy predict speech presence?
    if quiet and vad is not None and 0 < vad.sum() < len(vad):
        m = min(len(act), len(vad))
        row["vad_auc"] = auc(act[:m], vad[:m])
        rng = np.random.default_rng(int(rid) if rid.isdigit() else 0)
        row["vad_auc_null"] = auc(np.roll(act[:m], rng.integers(m // 4, 3 * m // 4)), vad[:m])
        row["speech_frac"] = float(vad[:m].mean())
    else:
        row["vad_auc"] = row["vad_auc_null"] = row["speech_frac"] = None

    # --- Test 2: does the optical channel give a usable F0?
    if quiet and vad is not None:
        _, mic_fr = frame_energy_db(bandpass(mic, sr, *MIC_BAND), sr, frame, hop)
        _, opt_fr = frame_energy_db(highpass(opt, sr, OPT_BAND[0]), sr, frame, hop)
        f0_mic, cl_mic = estimate_f0(mic_fr, sr)
        f0_opt, cl_opt = estimate_f0(opt_fr, sr)
        m = min(len(f0_mic), len(f0_opt), len(vad))
        ok = vad[:m] & np.isfinite(f0_mic[:m]) & np.isfinite(f0_opt[:m]) & (cl_mic[:m] > 0.4)
        if ok.sum() >= 20:
            a, b = f0_mic[:m][ok], f0_opt[:m][ok]
            rel = np.abs(b - a) / a
            row["f0_n"] = int(ok.sum())
            row["f0_mae_hz"] = float(np.median(np.abs(b - a)))
            row["f0_gross_error"] = float((rel > 0.20).mean())
            row["f0_octave_ok"] = float((np.minimum.reduce([
                np.abs(b - a), np.abs(2 * b - a), np.abs(b - 2 * a)]) / a < 0.20).mean())
            row["f0_median_mic"] = float(np.median(a))
        else:
            row["f0_n"] = int(ok.sum())
    else:
        row["f0_n"] = 0

    # --- Test 3: what does the simplest optical mask do to the microphone?
    y, g = apply_mask(mic, sr, act, hop)
    if vad is not None:
        m = min(len(vad), len(g))
        sp, si = vad[:m], ~vad[:m]
        if sp.any() and si.any():
            gi = g[:m]
            row["gain_on_speech_db"] = float(20 * np.log10(gi[sp].mean() + 1e-9))
            row["gain_on_silence_db"] = float(20 * np.log10(gi[si].mean() + 1e-9))
            row["mask_contrast_db"] = row["gain_on_speech_db"] - row["gain_on_silence_db"]
    return row, mic, y, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--metadata", default="Recordings Metadata.json")
    ap.add_argument("--out", default="diagnostics.json")
    ap.add_argument("--sr", type=int, default=WORK_SR)
    ap.add_argument("--write-masked", nargs="*", default=[],
                    help="recording ids to export as masked wav for the detector")
    ap.add_argument("--masked-dir", default="masked")
    args = ap.parse_args()

    root = Path(args.root)
    meta = json.load(open(args.metadata, encoding="utf-8"))
    frame = int(FRAME_MS * args.sr / 1000)
    hop = int(HOP_MS * args.sr / 1000)

    rows = []
    want = set(args.write_masked)
    if want:
        os.makedirs(args.masked_dir, exist_ok=True)

    for lf in sorted((root / "laser").glob("Laser_*.wav")):
        rid = lf.stem.split("_")[1]
        mf = root / "microphone" / f"microphone_{rid}.wav"
        if not mf.exists():
            continue
        try:
            row, mic, y, sr = analyse(rid, mf, lf, meta.get(rid, {}), args.sr, frame, hop)
        except Exception as exc:
            print(f"[warn] {rid}: {exc}")
            continue
        rows.append(row)
        if rid in want:
            sf.write(f"{args.masked_dir}/masked_{rid}.wav", y / (np.abs(y).max() + 1e-9), sr)
            sf.write(f"{args.masked_dir}/orig_{rid}.wav", mic / (np.abs(mic).max() + 1e-9), sr)
        print(f"{rid} spk{row['speaker']} "
              f"AUC={row.get('vad_auc') if row.get('vad_auc') is None else round(row['vad_auc'],3)} "
              f"F0gross={row.get('f0_gross_error')} "
              f"contrast={row.get('mask_contrast_db') and round(row['mask_contrast_db'],1)}")

    json.dump(rows, open(args.out, "w"), indent=1)
    report(rows)


def report(rows):
    def col(rs, key):
        v = [r[key] for r in rs if r.get(key) is not None and np.isfinite(r[key])]
        return np.array(v) if v else np.array([])

    print("\n" + "=" * 78)
    print("TEST 1 — does optical energy predict speech presence? (quiet recordings)")
    print("=" * 78)
    print(f"{'speaker':>8} {'n':>4} {'AUC':>8} {'null':>8} {'AUC>0.8':>9}")
    by = defaultdict(list)
    for r in rows:
        by[str(r["speaker"])].append(r)
    for s in sorted(by):
        a, z = col(by[s], "vad_auc"), col(by[s], "vad_auc_null")
        if len(a):
            print(f"{s:>8} {len(a):>4} {a.mean():>8.3f} {z.mean():>8.3f} {(a > 0.8).mean():>8.0%}")
    a = col(rows, "vad_auc")
    if len(a):
        print(f"{'ALL':>8} {len(a):>4} {a.mean():>8.3f} {col(rows,'vad_auc_null').mean():>8.3f} "
              f"{(a > 0.8).mean():>8.0%}")
        print("\n  Read: null should sit near 0.500. If the real AUC is not clearly above it,")
        print("  the optical channel does not indicate speech presence and masking cannot work.")

    print("\n" + "=" * 78)
    print("TEST 2 — is the optical F0 usable?")
    print("=" * 78)
    print(f"{'speaker':>8} {'n':>5} {'MAE Hz':>8} {'gross':>8} {'w/octave':>9} {'F0 mic':>8}")
    for s in sorted(by):
        g = col(by[s], "f0_gross_error")
        if len(g):
            print(f"{s:>8} {len(g):>5} {col(by[s],'f0_mae_hz').mean():>8.1f} "
                  f"{g.mean():>7.0%} {col(by[s],'f0_octave_ok').mean():>8.0%} "
                  f"{col(by[s],'f0_median_mic').mean():>8.0f}")
    g = col(rows, "f0_gross_error")
    if len(g):
        print(f"{'ALL':>8} {len(g):>5} {col(rows,'f0_mae_hz').mean():>8.1f} "
              f"{g.mean():>7.0%} {col(rows,'f0_octave_ok').mean():>8.0%}")
        print("\n  Read: gross error is the fraction of frames off by more than 20%.")
        print("  'w/octave' allows octave confusions, which are trivially correctable.")
        print("  Below roughly 30% gross error, harmonic extension is worth building.")

    print("\n" + "=" * 78)
    print("TEST 3 — what does the simplest optical mask do?")
    print("=" * 78)
    c = col(rows, "mask_contrast_db")
    if len(c):
        print(f"  mask contrast (speech gain - silence gain): {c.mean():+.1f} dB "
              f"[{np.percentile(c,25):+.1f}, {np.percentile(c,75):+.1f}]")
        print(f"  gain applied on speech frames : {col(rows,'gain_on_speech_db').mean():+.1f} dB")
        print(f"  gain applied on silence frames: {col(rows,'gain_on_silence_db').mean():+.1f} dB")
        print("\n  Read: contrast near 0 dB means the mask is flat and does nothing.")
        print("  Run the keyword detector on masked/*.wav and compare against 0.898.")

    print("\n" + "=" * 78)
    print("DECISION")
    print("=" * 78)
    a = col(rows, "vad_auc")
    g = col(rows, "f0_gross_error")
    if len(a) and a.mean() > 0.80:
        print("  Test 1 passes. Optical masking is viable.")
        if len(g) and g.mean() < 0.30:
            print("  Test 2 passes. Build the harmonic-extension model.")
        else:
            print("  Test 2 fails. Stay with presence-only masking; do not build")
            print("  harmonic extension on an F0 estimate this noisy.")
    elif len(a) and a.mean() > 0.65:
        print("  Test 1 is marginal. Check the per-speaker table: if speakers 06 and 07")
        print("  pass while 01-04 do not, the limitation is acquisition, not method,")
        print("  and that is itself a reportable finding.")
    else:
        print("  Test 1 fails. The optical channel does not indicate speech presence")
        print("  reliably. Report this as a channel characterisation result and do not")
        print("  invest in the masking architecture.")


if __name__ == "__main__":
    main()
