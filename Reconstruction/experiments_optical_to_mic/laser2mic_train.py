"""
Experiment: optical -> clean microphone (the left branch of the diagram).

Input  : the optically reconstructed waveform (band-limited to ~1.5 kHz)
Target : the simultaneously recorded microphone signal (full band)

The model must therefore do two things at once: reproduce what the optical
channel already contains below 1.5 kHz, and INVENT everything above it. The
second part is the whole question. With 137 usable pairs from 7 speakers, the
failure mode to expect is a speaker-averaged hallucination that sounds the same
regardless of who spoke. laser2mic_eval.py exists specifically to detect that.

Includes the fix for the loss blow-up seen in the previous run (32 of 120 epochs
had train_loss above 1e5): near-silent target chunks drove the spectral
convergence denominator to ~1e-5 while the numerator stayed O(1).

Usage:
    python laser2mic_train.py --pairs pairs.json --val-speakers 05 --test-speakers 06
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import DataLoader, Dataset

from model import OpticalUNet

OPT_HP_HZ = 150.0      # remove optical drift
MIC_HP_HZ = 100.0      # remove microphone DC / rumble only
SPLIT_HZ = 1500.0      # optical Nyquist: below = copy, above = invent


# --------------------------------------------------------------------- dataset

class Laser2MicDataset(Dataset):
    def __init__(self, pairs, sr=8000, chunk_sec=2.0, train=True, keyword_bias=0.3):
        self.pairs, self.sr = list(pairs), sr
        self.chunk = int(chunk_sec * sr)
        self.train, self.keyword_bias = train, keyword_bias
        self._rs = {}

    def _load(self, path):
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x[:, 0]
        t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
        if sr != self.sr:
            if (sr, self.sr) not in self._rs:
                self._rs[(sr, self.sr)] = torchaudio.transforms.Resample(sr, self.sr)
            t = self._rs[(sr, self.sr)](t)
        return t

    @staticmethod
    def _rms(x):
        return x / (x.pow(2).mean().sqrt() + 1e-8)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx, _depth=0):
        rec = self.pairs[idx]
        mic, opt = self._load(rec["microphone"]), self._load(rec["laser"])

        d = int(round(rec["delay_samples"] * self.sr / rec["sample_rate"]))
        if d > 0:
            mic, opt = mic[:, d:], opt[:, : opt.shape[-1] - d]
        elif d < 0:
            mic, opt = mic[:, : mic.shape[-1] + d], opt[:, -d:]
        n = min(mic.shape[-1], opt.shape[-1])
        mic, opt = mic[:, :n], opt[:, :n]

        if n < self.chunk:
            pad = self.chunk - n
            mic, opt, start = nn.functional.pad(mic, (0, pad)), nn.functional.pad(opt, (0, pad)), 0
        elif self.train:
            ks = rec.get("keyword_start")
            if self.keyword_bias > 0 and ks not in (None, -1) and random.random() < self.keyword_bias:
                start = min(max(0, int(ks * self.sr) - self.chunk // 2), n - self.chunk)
            else:
                start = random.randint(0, n - self.chunk)
        else:
            start = max(0, (n - self.chunk) // 2)

        mic, opt = mic[:, start:start + self.chunk], opt[:, start:start + self.chunk]

        opt = AF.highpass_biquad(opt, self.sr, OPT_HP_HZ)
        mic = AF.highpass_biquad(mic, self.sr, MIC_HP_HZ)

        # Reject a near-silent target rather than letting it destabilise the loss.
        if mic.pow(2).mean().sqrt() < 1e-5 or opt.pow(2).mean().sqrt() < 1e-5:
            if _depth < 8:
                return self.__getitem__(random.randrange(len(self.pairs)), _depth + 1)
            mic = torch.zeros_like(mic)

        opt, mic = self._rms(opt), self._rms(mic)
        w = float(rec.get("msc_max", 1.0))
        return opt, mic, torch.tensor(w, dtype=torch.float32)


# ------------------------------------------------------------------------ loss

class BandLoss(nn.Module):
    """Multi-resolution STFT loss with an extra term on the extension band,
    so the model cannot minimise the objective by only getting <1.5 kHz right."""

    def __init__(self, sr=8000, configs=((512, 128), (256, 64), (1024, 256)),
                 ext_weight=2.0, l1_weight=1.0):
        super().__init__()
        self.sr, self.ext_weight, self.l1_weight = sr, ext_weight, l1_weight
        self.cfg = configs
        for i, (n_fft, _) in enumerate(configs):
            self.register_buffer(f"w{i}", torch.hann_window(n_fft))
            f = torch.linspace(0, sr / 2, n_fft // 2 + 1)
            self.register_buffer(f"ext{i}", f >= SPLIT_HZ)

    def _mag(self, x, i):
        n_fft, hop = self.cfg[i]
        s = torch.stft(x.squeeze(1), n_fft=n_fft, hop_length=hop,
                       window=getattr(self, f"w{i}"), return_complex=True)
        return s.abs().clamp_min(1e-7)

    @staticmethod
    def _terms(p, t):
        num = torch.linalg.norm((t - p).flatten(1), dim=1)
        ref = torch.linalg.norm(t.flatten(1), dim=1)
        den = ref.clamp_min(1e-2 * ref.median().clamp_min(1e-4))
        sc = (num / den).clamp_max(10.0)
        mag = (torch.log(p) - torch.log(t)).abs().flatten(1).mean(1)
        return sc + mag

    def forward(self, pred, target, weight=None):
        total = 0.0
        for i in range(len(self.cfg)):
            p, t = self._mag(pred, i), self._mag(target, i)
            per = self._terms(p, t)
            ext = getattr(self, f"ext{i}")
            if ext.any():
                per = per + self.ext_weight * self._terms(p[:, ext, :], t[:, ext, :])
            total = total + per
        total = total / len(self.cfg)
        total = total + self.l1_weight * (pred - target).abs().flatten(1).mean(1)

        if weight is not None:
            w = weight.to(total.device).clamp_min(1e-3)
            return (total * w).sum() / w.sum()
        return total.mean()


# ----------------------------------------------------------------------- train

def epoch(model, loader, crit, dev, opt=None):
    model.train(opt is not None)
    tot, n = 0.0, 0
    for x, y, w in loader:
        x, y, w = x.to(dev), y.to(dev), w.to(dev)
        with torch.set_grad_enabled(opt is not None):
            loss = crit(model(x), y, weight=w)
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tot += loss.item() * x.size(0)
        n += x.size(0)
    return tot / max(n, 1)


def main():
    from dataset import load_pairs, speaker_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.json")
    ap.add_argument("--val-speakers", nargs="+", default=["05"])
    ap.add_argument("--test-speakers", nargs="+", default=["06"])
    ap.add_argument("--min-msc", type=float, default=0.35)
    ap.add_argument("--sr", type=int, default=8000)
    ap.add_argument("--chunk-sec", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default="runs/laser2mic")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs, min_msc=args.min_msc)
    tr, va, te = speaker_split(pairs, args.val_speakers, args.test_speakers)
    print(f"pairs after QC: {len(pairs)}   train {len(tr)}  val {len(va)}  test {len(te)}")
    if not tr or not va:
        raise SystemExit("empty split")

    mk = lambda p, t: DataLoader(
        Laser2MicDataset(p, args.sr, args.chunk_sec, train=t),
        batch_size=args.batch_size, shuffle=t, num_workers=args.workers, drop_last=t)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpticalUNet(in_channels=1).to(dev)
    crit = BandLoss(sr=args.sr).to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda e: (e + 1) / args.warmup if e < args.warmup
                                              else 0.5 * (1 + math.cos(math.pi * (e - args.warmup) /
                                                                       max(1, args.epochs - args.warmup))))

    tl, vl = mk(tr, True), mk(va, False)
    best, hist = float("inf"), []
    for e in range(args.epochs):
        a = epoch(model, tl, crit, dev, optim)
        b = epoch(model, vl, crit, dev)
        sched.step()
        hist.append({"epoch": e + 1, "train": a, "val": b})
        tag = ""
        if b < best:
            best, tag = b, "  <- best"
            torch.save({"model": model.state_dict(), "args": vars(args)}, out / "best.pt")
        print(f"epoch {e+1:3d}/{args.epochs}  train {a:.4f}  val {b:.4f}{tag}")
        if a > 100:
            print("   [warn] training loss is diverging — inspect the batch, do not ignore this")

    json.dump(hist, open(out / "history.json", "w"), indent=1)
    json.dump({"train": [p["id"] for p in tr], "val": [p["id"] for p in va],
               "test": [p["id"] for p in te]}, open(out / "splits.json", "w"), indent=1)
    print(f"\ndone. now run:  python laser2mic_eval.py --ckpt {out}/best.pt --pairs {args.pairs}")


if __name__ == "__main__":
    main()
