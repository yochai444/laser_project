"""
Does the optical -> microphone model actually reconstruct the extension band,
or does it emit an average speaker?

Three tests. The first says whether anything was learned; the second and third
say whether what was learned is speaker-specific or a single memorised spectrum.
The third is the one that matters most, and it needs no labels at all.

    1  Spectral distance, split at 1.5 kHz, against four baselines.
       The decisive baseline is GLOBAL MEAN: the average log-spectrum of the
       training microphones, i.e. a model that ignores its input entirely. If
       the model does not beat that above 1.5 kHz, it has learned nothing there.

    2  Speaker identification from the extension band alone.
       Nearest speaker centroid over the 1.5-4 kHz long-term average spectrum.
       Chance is 1/n_speakers. If real microphone audio identifies the speaker
       and the model output does not, the output is speaker-agnostic.

    3  Output diversity.
       Mean pairwise correlation between utterance spectra. If the model's
       outputs resemble each other much more than the true microphones resemble
       each other, the model has collapsed onto one answer.

Usage:
    python laser2mic_eval.py --ckpt runs/laser2mic/best.pt --pairs pairs.json
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

from model import OpticalUNet

SPLIT_HZ = 1500.0
LOW_BAND = (150.0, SPLIT_HZ)
EXT_BAND = (SPLIT_HZ, 4000.0)
N_FFT, HOP = 512, 128


def load(path, sr):
    x, s = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    if s != sr:
        t = torchaudio.transforms.Resample(s, sr)(t)
    return t


def rms(x):
    return x / (x.pow(2).mean().sqrt() + 1e-8)


def logspec(x, sr):
    s = torch.stft(x.squeeze(0), n_fft=N_FFT, hop_length=HOP,
                   window=torch.hann_window(N_FFT), return_complex=True)
    return 20 * torch.log10(s.abs().clamp_min(1e-7)).numpy()


def band_mask(sr, lo, hi):
    f = np.linspace(0, sr / 2, N_FFT // 2 + 1)
    return (f >= lo) & (f < hi)


def lsd(a, b, mask):
    """Mean-removed log-spectral distance, so an overall gain offset is not
    counted as an error."""
    A, B = a[mask], b[mask]
    return float(np.sqrt((((A - A.mean()) - (B - B.mean())) ** 2).mean()))


def ltas(spec, mask):
    v = spec[mask].mean(axis=1)
    return v - v.mean()


@torch.no_grad()
def run_model(model, x, sr, chunk_sec=2.0):
    chunk = int(chunk_sec * sr)
    n = x.shape[-1]
    if n <= chunk:
        return model(x.unsqueeze(0)).squeeze(0)
    hop, out, wsum = chunk // 2, torch.zeros_like(x), torch.zeros_like(x)
    win = torch.hann_window(chunk).unsqueeze(0)
    for s in list(range(0, n - chunk + 1, hop)) + [n - chunk]:
        out[:, s:s + chunk] += model(x[:, s:s + chunk].unsqueeze(0)).squeeze(0) * win
        wsum[:, s:s + chunk] += win
    return out / wsum.clamp_min(1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pairs", default="pairs.json")
    ap.add_argument("--splits", default=None, help="splits.json from training")
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--out", default="hallucination_report.json")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    model = OpticalUNet(in_channels=1)
    model.load_state_dict(ck["model"])
    model.eval()

    recs = json.load(open(args.pairs, encoding="utf-8"))
    splits_path = args.splits or Path(args.ckpt).parent / "splits.json"
    splits = json.load(open(splits_path, encoding="utf-8"))
    train_ids, test_ids = set(splits["train"]), set(splits["test"] or splits["val"])

    lo_m, ex_m = band_mask(args.sr, *LOW_BAND), band_mask(args.sr, *EXT_BAND)

    # global mean log-spectrum of the TRAINING microphones = the "ignore the input" model
    acc, cnt = None, 0
    for rid in train_ids:
        if rid not in recs:
            continue
        s = logspec(rms(AF.highpass_biquad(load(recs[rid]["microphone"], args.sr), args.sr, 100.0)), args.sr)
        acc = s.mean(axis=1) if acc is None else acc + s.mean(axis=1)
        cnt += 1
    global_mean = (acc / max(cnt, 1))[:, None]

    rows, spk_ref, out_ltas, ref_ltas = [], defaultdict(list), [], []

    for rid in sorted(test_ids, key=lambda z: int(z)):
        r = recs.get(rid)
        if not r:
            continue
        mic = rms(AF.highpass_biquad(load(r["microphone"], args.sr), args.sr, 100.0))
        opt = rms(AF.highpass_biquad(load(r["laser"], args.sr), args.sr, 150.0))
        n = min(mic.shape[-1], opt.shape[-1])
        mic, opt = mic[:, :n], opt[:, :n]

        y = rms(run_model(model, opt, args.sr))
        S_y, S_m, S_o = logspec(y, args.sr), logspec(mic, args.sr), logspec(opt, args.sr)
        k = min(S_y.shape[1], S_m.shape[1], S_o.shape[1])
        S_y, S_m, S_o = S_y[:, :k], S_m[:, :k], S_o[:, :k]
        G = np.repeat(global_mean, k, axis=1)

        rows.append({
            "id": rid, "speaker": r.get("speaker"), "msc": r.get("msc_max"),
            "lsd_low_model": lsd(S_y, S_m, lo_m),
            "lsd_low_optical": lsd(S_o, S_m, lo_m),
            "lsd_ext_model": lsd(S_y, S_m, ex_m),
            "lsd_ext_optical": lsd(S_o, S_m, ex_m),
            "lsd_ext_globalmean": lsd(G, S_m, ex_m),
        })
        out_ltas.append(ltas(S_y, ex_m))
        ref_ltas.append(ltas(S_m, ex_m))

    for rid in train_ids:
        r = recs.get(rid)
        if not r:
            continue
        s = logspec(rms(AF.highpass_biquad(load(r["microphone"], args.sr), args.sr, 100.0)), args.sr)
        spk_ref[str(r.get("speaker"))].append(ltas(s, ex_m))

    report(rows, spk_ref, out_ltas, ref_ltas, recs, test_ids, ex_m, args)
    json.dump(rows, open(args.out, "w"), indent=1)


def report(rows, spk_ref, out_ltas, ref_ltas, recs, test_ids, ex_m, args):
    def col(k):
        return np.array([r[k] for r in rows if r.get(k) is not None])

    print("\n" + "=" * 72)
    print("TEST 1 — spectral distance to the true microphone (lower is better)")
    print("=" * 72)
    print(f"{'':<26}{'150-1500 Hz':>14}{'1500-4000 Hz':>16}")
    print(f"{'model output':<26}{col('lsd_low_model').mean():>14.2f}{col('lsd_ext_model').mean():>16.2f}")
    print(f"{'optical input (copy)':<26}{col('lsd_low_optical').mean():>14.2f}{col('lsd_ext_optical').mean():>16.2f}")
    print(f"{'global mean spectrum':<26}{'-':>14}{col('lsd_ext_globalmean').mean():>16.2f}")
    gap = col('lsd_ext_globalmean').mean() - col('lsd_ext_model').mean()
    print(f"\n  model beats the input-ignoring baseline above 1.5 kHz by {gap:+.2f} dB")
    print("  Under about +0.5 dB, the extension band is not being reconstructed.")

    print("\n" + "=" * 72)
    print("TEST 2 — can the speaker be identified from 1.5-4 kHz alone?")
    print("=" * 72)
    speakers = sorted(spk_ref)
    if speakers:
        cent = {s: np.mean(spk_ref[s], axis=0) for s in speakers}

        def nn_id(v):
            return min(speakers, key=lambda s: np.linalg.norm(v - cent[s]))

        truth = [str(recs[r["id"]].get("speaker")) for r in rows]
        held = set(truth)
        note = " (test speaker unseen in training — see note below)" if not (held & set(speakers)) else ""
        acc_out = np.mean([nn_id(v) == t for v, t in zip(out_ltas, truth)])
        acc_ref = np.mean([nn_id(v) == t for v, t in zip(ref_ltas, truth)])
        print(f"  from real microphone audio : {acc_ref:.1%}")
        print(f"  from model output          : {acc_out:.1%}")
        print(f"  chance                     : {1/len(speakers):.1%}{note}")
        print("\n  If the reference row is high and the model row is near chance, the")
        print("  output carries no speaker-specific information above 1.5 kHz.")
        if note:
            print("  With a fully held-out speaker, both rows will be low by construction;")
            print("  rerun with --splits pointing at a within-speaker split to read this test.")

    print("\n" + "=" * 72)
    print("TEST 3 — output diversity (the collapse test)")
    print("=" * 72)

    def mean_pairwise_corr(vs):
        if len(vs) < 2:
            return float("nan")
        M = np.array(vs)
        C = np.corrcoef(M)
        iu = np.triu_indices(len(vs), 1)
        return float(np.nanmean(C[iu]))

    c_out, c_ref = mean_pairwise_corr(out_ltas), mean_pairwise_corr(ref_ltas)
    print(f"  mean pairwise correlation between utterances, 1.5-4 kHz")
    print(f"    real microphones : {c_ref:.3f}")
    print(f"    model outputs    : {c_out:.3f}")
    print(f"    excess           : {c_out - c_ref:+.3f}")
    print("\n  Excess above about +0.15 means the outputs are more alike than the")
    print("  real recordings are: the model is emitting one spectrum regardless")
    print("  of what it was given.")

    print("\n" + "=" * 72)
    print("VERDICT")
    print("=" * 72)
    if gap < 0.5:
        print("  The extension band is not reconstructed. The optical channel does not")
        print("  support bandwidth extension on this corpus. Report it as a limit of")
        print("  the channel and do not build the shared-latent architecture on top.")
    elif c_out - c_ref > 0.15:
        print("  Something is reconstructed, but the outputs are near-identical across")
        print("  utterances — a speaker-averaged hallucination, not reconstruction.")
        print("  A latent consistency loss built on this would propagate the average.")
    else:
        print("  The extension band is reconstructed with utterance-specific detail.")
        print("  The shared-decoder architecture is worth building.")
    print("\n  In all three cases, also listen. These metrics do not measure intelligibility.")


if __name__ == "__main__":
    main()
