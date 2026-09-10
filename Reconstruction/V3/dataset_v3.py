"""
V3 dataset: microphone speech enhancement with optical auxiliary supervision.

Core idea
---------
- Clean microphone recording = final target.
- A noisy microphone input is created on the fly from the clean microphone.
- The synchronized laser signal is NOT the reconstruction target.
- The laser is returned only as auxiliary supervision during training.
- At inference, only the microphone is required.

The dataset uses pairs.json created by prepare_pairs.py.
"""

import json
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF
from torch.utils.data import Dataset


def _is_clean_noise_label(label):
    s = (label or "").strip().lower()
    return s in ("", "none", "no noise", "clean")


def _is_noise_only_label(label):
    s = (label or "").strip().lower()
    return "only" in s and "noise" in s


def load_v3_pairs(pairs_json, min_delay_conf=3.0, min_msc=0.0):
    """
    Clean-source recordings only.

    The microphone is the clean target, so low optical coherence does not invalidate
    the speech target. min_msc is therefore optional and defaults to 0.0.
    Optical reliability is handled later through per-example weighting.
    """
    recs = json.load(open(pairs_json, encoding="utf-8"))
    out = []
    for rid, r in recs.items():
        if r.get("delay_confidence", 0.0) < min_delay_conf:
            continue
        if r.get("msc_max", 0.0) < min_msc:
            continue
        if not _is_clean_noise_label(r.get("noise")):
            continue
        rr = dict(r)
        rr["id"] = rid
        out.append(rr)
    return out


def speaker_split(pairs, val_speakers, test_speakers):
    val_speakers = set(map(str, val_speakers))
    test_speakers = set(map(str, test_speakers))
    tr, va, te = [], [], []
    for p in pairs:
        s = str(p.get("speaker"))
        (te if s in test_speakers else va if s in val_speakers else tr).append(p)
    return tr, va, te


def noise_only_paths(pairs_json):
    recs = json.load(open(pairs_json, encoding="utf-8"))
    return [
        r["microphone"] for r in recs.values()
        if _is_noise_only_label(r.get("noise"))
    ]


class V3EnhancementDataset(Dataset):
    def __init__(
        self,
        pairs,
        target_sr=16000,
        chunk_sec=2.0,
        highpass_hz=80.0,
        train=True,
        real_noise_paths=None,
        real_noise_prob=0.5,
        gaussian_prob=0.5,
        snr_range=(0.0, 20.0),
        keyword_bias=0.30,
    ):
        self.pairs = list(pairs)
        self.sr = int(target_sr)
        self.chunk = int(round(chunk_sec * target_sr))
        self.highpass_hz = float(highpass_hz)
        self.train = bool(train)
        self.real_noise_prob = float(real_noise_prob) if real_noise_paths else 0.0
        self.gaussian_prob = float(gaussian_prob)
        self.snr_range = tuple(map(float, snr_range))
        self.keyword_bias = float(keyword_bias)

        self._resamplers = {}
        self.real_noise = []
        for p in (real_noise_paths or []):
            try:
                x, sr = sf.read(p, dtype="float32", always_2d=False)
                if x.ndim > 1:
                    x = x[:, 0]
                t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
                self.real_noise.append(self._resample(t, sr))
            except Exception as e:
                print(f"[dataset_v3] skipping noise file {p}: {e}")

        if self.real_noise:
            print(f"[dataset_v3] loaded {len(self.real_noise)} real noise-only recordings")

    def __len__(self):
        return len(self.pairs)

    def _resample(self, x, sr):
        if sr == self.sr:
            return x
        key = (sr, self.sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(sr, self.sr)
        return self._resamplers[key](x)

    def _load(self, path):
        x, sr = sf.read(path, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x[:, 0]
        t = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
        return self._resample(t, sr)

    @staticmethod
    def _remove_dc(x):
        return x - x.mean(dim=-1, keepdim=True)

    def _highpass(self, x):
        if self.highpass_hz <= 0:
            return x
        return AF.highpass_biquad(x, self.sr, self.highpass_hz)

    @staticmethod
    def _rms(x, eps=1e-8):
        return x.pow(2).mean().sqrt().clamp_min(eps)

    def _apply_delay(self, mic, opt, delay_samples_native, native_sr):
        delay = int(round(delay_samples_native * self.sr / native_sr))
        if delay > 0:
            mic = mic[:, delay:]
            opt = opt[:, : max(0, opt.shape[-1] - delay)]
        elif delay < 0:
            d = -delay
            mic = mic[:, : max(0, mic.shape[-1] - d)]
            opt = opt[:, d:]
        n = min(mic.shape[-1], opt.shape[-1])
        return mic[:, :n], opt[:, :n]

    def _choose_start(self, total, rec):
        if total <= self.chunk:
            return 0
        if not self.train:
            return max(0, (total - self.chunk) // 2)

        ks = rec.get("keyword_start")
        if (
            self.keyword_bias > 0
            and ks not in (None, -1)
            and random.random() < self.keyword_bias
        ):
            centre = int(float(ks) * self.sr)
            lo = centre - self.chunk // 2
            return min(max(0, lo), total - self.chunk)

        return random.randint(0, total - self.chunk)

    def _take_noise_chunk(self, noise):
        if noise.shape[-1] >= self.chunk:
            if self.train:
                s = random.randint(0, noise.shape[-1] - self.chunk)
            else:
                s = max(0, (noise.shape[-1] - self.chunk) // 2)
            return noise[:, s:s + self.chunk]
        return torch.nn.functional.pad(noise, (0, self.chunk - noise.shape[-1]))

    def _mix_at_snr(self, clean, noise, snr_db):
        clean_rms = self._rms(clean)
        noise = noise - noise.mean(dim=-1, keepdim=True)
        noise_rms = self._rms(noise)
        scale = clean_rms / (noise_rms * (10.0 ** (snr_db / 20.0)))
        return clean + noise * scale

    def _make_noisy(self, clean):
        if not self.train:
            # Deterministic evaluation corruption: Gaussian noise at 10 dB.
            g = torch.randn_like(clean)
            return self._mix_at_snr(clean, g, 10.0), 10.0, "gaussian"

        use_real = self.real_noise and random.random() < self.real_noise_prob
        if use_real:
            nz = self._take_noise_chunk(random.choice(self.real_noise))
            kind = "real"
        else:
            nz = torch.randn_like(clean)
            kind = "gaussian"

        snr = random.uniform(*self.snr_range)
        noisy = self._mix_at_snr(clean, nz, snr)

        # Optional second, weaker Gaussian component for robustness.
        if kind == "real" and random.random() < self.gaussian_prob:
            g = torch.randn_like(clean)
            noisy = self._mix_at_snr(noisy, g, random.uniform(15.0, 30.0))

        return noisy, snr, kind

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        native_sr = int(rec["sample_rate"])

        clean = self._load(rec["microphone"])
        optical = self._load(rec["laser"])

        clean, optical = self._apply_delay(
            clean, optical, rec.get("delay_samples", 0), native_sr
        )

        clean = self._highpass(self._remove_dc(clean))
        optical = self._highpass(self._remove_dc(optical))

        total = min(clean.shape[-1], optical.shape[-1])
        clean, optical = clean[:, :total], optical[:, :total]

        start = self._choose_start(total, rec)
        clean = clean[:, start:start + self.chunk]
        optical = optical[:, start:start + self.chunk]

        if clean.shape[-1] < self.chunk:
            pad = self.chunk - clean.shape[-1]
            clean = torch.nn.functional.pad(clean, (0, pad))
            optical = torch.nn.functional.pad(optical, (0, pad))

        # One scale for the clean microphone target.
        # No peak normalization: preserve waveform shape.
        scale = self._rms(clean)
        clean = clean / scale

        # Optical is only an auxiliary guide, so normalize independently.
        optical = optical / self._rms(optical)

        noisy, snr_db, noise_kind = self._make_noisy(clean)

        # Guard against pathological files.
        for x in (noisy, clean, optical):
            if not torch.isfinite(x).all():
                x.zero_()

        optical_weight = float(np.clip(rec.get("msc_max", 0.0), 0.0, 1.0))

        return {
            "noisy": noisy.float(),
            "clean": clean.float(),
            "optical": optical.float(),
            "optical_weight": torch.tensor(optical_weight, dtype=torch.float32),
            "snr_db": torch.tensor(float(snr_db), dtype=torch.float32),
            "id": rec["id"],
        }
