"""
V3 inference: noisy microphone -> enhanced microphone.
Laser is NOT used.

Example:
    python infer_v3.py --ckpt runs/optical_guided_v3/best.pt ^
        --mic microphone/microphone_300.wav --out enhanced_300.wav
"""

import argparse
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

from model_v3 import OpticalGuidedEnhancer


def load(path, target_sr):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x[:, 0]
    t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    if sr != target_sr:
        t = torchaudio.transforms.Resample(sr, target_sr)(t)
    return t


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/optical_guided_v3/best.pt")
    ap.add_argument("--mic", required=True)
    ap.add_argument("--out", default="enhanced_v3.wav")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck.get("args", {})

    sr = int(cfg.get("sr", 16000))
    highpass = float(cfg.get("highpass", 80.0))
    n_fft = int(cfg.get("n_fft", 512))
    hop = int(cfg.get("hop", 128))
    win = int(cfg.get("win", 512))

    model = OpticalGuidedEnhancer(n_fft=n_fft, hop=hop, win=win)
    model.load_state_dict(ck["model"])
    model.eval()

    mic = load(args.mic, sr)
    if highpass > 0:
        mic = AF.highpass_biquad(mic, sr, highpass)

    # Match clean-target scale convention used in training.
    scale = mic.pow(2).mean().sqrt().clamp_min(1e-8)
    x = mic / scale

    out = model(x.unsqueeze(0) if x.ndim == 2 else x)
    enh = out["enhanced"]
    if enh.ndim == 3:
        enh = enh.squeeze(0)

    # Restore approximately the input microphone RMS.
    enh = enh * scale

    # Only prevent clipping; do not normalize to unit RMS.
    peak = enh.abs().max().item()
    if peak > 0.98:
        enh = enh * (0.98 / peak)

    sf.write(args.out, enh.squeeze().cpu().numpy(), sr)
    print(f"wrote {args.out} ({sr} Hz)")


if __name__ == "__main__":
    main()
