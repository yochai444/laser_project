"""
Stage 6 — inference and figures.

Changes from the previous version, and why:

  colour      Both panels now share vmin/vmax. The old plot called imshow without
              them, so each panel auto-scaled to its own range (-100..+40 dB for the
              input, -80..+40 for the output). Twenty decibels of the apparent
              "cleaning" was the colour map, not the model.

  panels      Three panels: microphone input, model output, and the optical
              recording. Without the third panel there is no way to tell whether the
              model approached the target or collapsed onto something else.

  axes        Frequency axis in Hz, time axis in seconds. "Frequency Bins" is not
              interpretable and hides the fact that the usable band is narrow.

  chunking    Long files are processed in overlapping windows with a Hann cross-fade,
              matching the 2 s chunks used in training. The old script pushed a full
              20 s file through a model that had only ever seen 2 s.

Usage:
    python infer_and_plot.py --ckpt runs/optical_unet/best.pt \
        --mic dataset/microphone/microphone_300.wav \
        --opt dataset/laser/Laser_300.wav --out fig_300
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

from model import OpticalUNet

DB_FLOOR, DB_CEIL = -80.0, 10.0


def load(path, target_sr):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    if sr != target_sr:
        t = torchaudio.transforms.Resample(sr, target_sr)(t)
    return t


def prep(x, sr, highpass):
    if highpass > 0:
        x = AF.highpass_biquad(x, sr, highpass)
    return x / (x.pow(2).mean().sqrt() + 1e-8)


@torch.no_grad()
def enhance(model, mic, sr, chunk_sec=2.0, overlap=0.5):
    chunk = int(chunk_sec * sr)
    hop = int(chunk * (1 - overlap))
    n = mic.shape[-1]
    if n <= chunk:
        return model(mic.unsqueeze(0)).squeeze(0)

    out = torch.zeros_like(mic)
    wsum = torch.zeros_like(mic)
    win = torch.hann_window(chunk).unsqueeze(0)
    for s in range(0, n - chunk + 1, hop):
        seg = mic[:, s:s + chunk].unsqueeze(0)
        y = model(seg).squeeze(0)
        out[:, s:s + chunk] += y * win
        wsum[:, s:s + chunk] += win
    tail = n - chunk
    if tail % hop:
        seg = mic[:, -chunk:].unsqueeze(0)
        y = model(seg).squeeze(0)
        out[:, -chunk:] += y * win
        wsum[:, -chunk:] += win
    return out / wsum.clamp_min(1e-6)


def spec_db(x, n_fft, hop):
    s = torch.stft(x.squeeze(0), n_fft=n_fft, hop_length=hop,
                   window=torch.hann_window(n_fft), return_complex=True)
    mag = s.abs().clamp_min(1e-7)
    mag = mag / mag.max()
    return 20 * torch.log10(mag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mic", required=True)
    ap.add_argument("--opt", required=True)
    ap.add_argument("--out", default="comparison")
    ap.add_argument("--sr", type=int, default=4000)
    ap.add_argument("--highpass", type=float, default=150.0)
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--ablate-input", action="store_true",
                    help="feed zeros instead of the microphone, as a sanity check")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    model = OpticalUNet(in_channels=1)
    model.load_state_dict(ck["model"])
    model.eval()

    mic = prep(load(args.mic, args.sr), args.sr, args.highpass)
    opt = prep(load(args.opt, args.sr), args.sr, args.highpass)
    n = min(mic.shape[-1], opt.shape[-1])
    mic, opt = mic[:, :n], opt[:, :n]

    src = torch.zeros_like(mic) if args.ablate_input else mic
    enh = enhance(model, src, args.sr)
    enh = enh / (enh.pow(2).mean().sqrt() + 1e-8)

    sf.write(f"{args.out}.wav", enh.squeeze(0).numpy(), args.sr)

    panels = [("Microphone input", mic), ("Model output", enh), ("Optical reference (target)", opt)]
    fig, axs = plt.subplots(3, 1, figsize=(11, 10), sharex=True, sharey=True)
    extent = [0, n / args.sr, 0, args.sr / 2]

    for ax, (title, sig) in zip(axs, panels):
        im = ax.imshow(spec_db(sig, args.n_fft, args.hop).numpy(),
                       origin="lower", aspect="auto", cmap="magma",
                       vmin=DB_FLOOR, vmax=DB_CEIL, extent=extent)
        ax.set_title(title)
        ax.set_ylabel("Frequency (Hz)")
        ax.axhline(150, color="w", lw=0.6, ls="--", alpha=0.5)
        ax.axhline(1400, color="w", lw=0.6, ls="--", alpha=0.5)

    axs[-1].set_xlabel("Time (s)")
    fig.colorbar(im, ax=axs, format="%+2.0f dB", fraction=0.03)
    fig.suptitle("Dashed lines mark the band used for training and evaluation", fontsize=9, y=0.995)
    fig.savefig(f"{args.out}.png", dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}.png and {args.out}.wav")


if __name__ == "__main__":
    main()
