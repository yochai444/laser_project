"""
Stage 2 — dataset.

Changes from the previous version, and why:

  target      The optical signal is now the TARGET, not a second input channel.
              This is what the thesis claims the system does, and it is what makes
              the deployment claim ("no laser at inference") true. The old code fed
              (noisy_mic, optical) in and regressed onto clean_mic, which is sensor
              fusion and requires the laser at inference.

  input       The microphone recording is used as-is. No synthetic noise is added.
              The old code built its noisy input by mixing white noise from a single
              file (microphone_177.wav, speaker 04, "White Noise Only") at a random
              SNR. That is exactly the synthetic-mixing construction the literature
              review is written against, and it means the model only ever saw one
              noise realisation. Optional augmentation from a POOL of real noise-only
              recordings is available via --noise-pool, off by default.

  highpass    The optical target is high-passed at 150 Hz. In this corpus roughly 90%
              of the raw optical energy sits below 134 Hz and is slow surface drift,
              not speech. Without the high-pass, an L1 or STFT loss is dominated by
              drift and the model has almost no gradient signal from speech.

  gain        Both channels are normalised to unit RMS after filtering, not to peak.
              The two modalities differ in RMS by roughly a factor of 20 here, and
              peak normalisation is driven by single-sample transients.

  alignment   Read from pairs.json. Never recomputed per item.
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

BAND_LO_HZ = 150.0


class OpticalPairDataset(Dataset):
    def __init__(
        self,
        pairs,                     # list of dicts from pairs.json
        target_sr=4000,
        chunk_sec=2.0,
        highpass_hz=BAND_LO_HZ,
        noise_pool=None,           # list of wav paths of REAL noise-only recordings
        noise_prob=0.0,
        snr_range=(0.0, 15.0),
        train=True,
        keyword_bias=0.0,          # prob. of centring the chunk on the keyword
    ):
        self.pairs = list(pairs)
        self.sr = target_sr
        self.chunk = int(chunk_sec * target_sr)
        self.highpass_hz = highpass_hz
        self.noise_prob = noise_prob if noise_pool else 0.0
        self.snr_range = snr_range
        self.train = train
        self.keyword_bias = keyword_bias

        self._resamplers = {}
        self.noise = []
        for p in (noise_pool or []):
            w, sr = sf.read(p, dtype="float32", always_2d=False)
            if w.ndim > 1:
                w = w[:, 0]
            t = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0)
            self.noise.append(self._resample(t, sr))
        if noise_pool:
            print(f"[dataset] loaded {len(self.noise)} real noise recordings")

    # ---------- helpers ----------

    def _resample(self, wav, sr):
        if sr == self.sr:
            return wav
        key = (sr, self.sr)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(sr, self.sr)
        return self._resamplers[key](wav)

    def _load(self, path, sr_hint=None):
        w, sr = sf.read(path, dtype="float32", always_2d=False)
        if w.ndim > 1:
            w = w[:, 0]
        t = torch.from_numpy(np.ascontiguousarray(w)).unsqueeze(0)
        return self._resample(t, sr)

    @staticmethod
    def _rms_norm(x, eps=1e-8):
        return x / (x.pow(2).mean().sqrt() + eps)

    def _highpass(self, x):
        if self.highpass_hz <= 0:
            return x
        return AF.highpass_biquad(x, self.sr, self.highpass_hz)

    def _apply_delay(self, mic, opt, delay):
        """delay > 0 means the mic lags the optical signal."""
        if delay > 0:
            mic, opt = mic[:, delay:], opt[:, : opt.shape[-1] - delay]
        elif delay < 0:
            d = -delay
            mic, opt = mic[:, : mic.shape[-1] - d], opt[:, d:]
        n = min(mic.shape[-1], opt.shape[-1])
        return mic[:, :n], opt[:, :n]

    def _add_real_noise(self, mic):
        if not self.noise or random.random() > self.noise_prob:
            return mic
        nz = random.choice(self.noise)
        if nz.shape[-1] > self.chunk:
            s = random.randint(0, nz.shape[-1] - self.chunk)
            nz = nz[:, s: s + self.chunk]
        else:
            nz = torch.nn.functional.pad(nz, (0, self.chunk - nz.shape[-1]))
        snr = random.uniform(*self.snr_range)
        s_rms = mic.pow(2).mean().sqrt() + 1e-8
        n_rms = nz.pow(2).mean().sqrt() + 1e-8
        return mic + nz * (s_rms / n_rms) / (10 ** (snr / 20.0))

    # ---------- protocol ----------

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rec = self.pairs[idx]
        native_sr = rec["sample_rate"]

        mic = self._load(rec["microphone"])
        opt = self._load(rec["laser"])

        delay = int(round(rec["delay_samples"] * self.sr / native_sr))
        mic, opt = self._apply_delay(mic, opt, delay)

        total = mic.shape[-1]
        if total < self.chunk:
            pad = self.chunk - total
            mic = torch.nn.functional.pad(mic, (0, pad))
            opt = torch.nn.functional.pad(opt, (0, pad))
            start = 0
        elif self.train:
            start = None
            ks = rec.get("keyword_start")
            if self.keyword_bias > 0 and ks not in (None, -1) and random.random() < self.keyword_bias:
                centre = int(ks * self.sr)
                lo = max(0, centre - self.chunk // 2)
                start = min(lo, total - self.chunk)
            if start is None:
                start = random.randint(0, total - self.chunk)
        else:
            start = max(0, (total - self.chunk) // 2)

        mic = mic[:, start: start + self.chunk]
        opt = opt[:, start: start + self.chunk]

        mic = self._add_real_noise(mic)

        # High-pass BOTH so input and target live in the same band, then RMS-normalise.
        mic = self._rms_norm(self._highpass(mic))
        opt = self._rms_norm(self._highpass(opt))

        # Reject a near-silent target and resample rather than let it
        # destabilise the spectral-convergence term (see loss.py).
        if opt.pow(2).mean().sqrt() < 1e-5 and len(self.pairs) > 1:
            return self.__getitem__(random.randrange(len(self.pairs)))

        # Guard against the rare all-zero chunk producing NaNs downstream.
        if not torch.isfinite(mic).all():
            mic = torch.zeros_like(mic)
        if not torch.isfinite(opt).all():
            opt = torch.zeros_like(opt)

        weight = float(rec.get("msc_max", 1.0))
        return mic, opt, torch.tensor(weight, dtype=torch.float32)


# ---------- pair selection and speaker-disjoint splits ----------

def load_pairs(pairs_json, min_msc=0.35, min_delay_conf=3.0, exclude_noise_only=True):
    recs = json.load(open(pairs_json, encoding="utf-8"))
    out = []
    for rid, r in recs.items():
        if r.get("msc_max", 0) < min_msc:
            continue
        if r.get("delay_confidence", 0) < min_delay_conf:
            continue
        if exclude_noise_only and "only" in (r.get("noise") or "").lower():
            continue
        r = dict(r, id=rid)
        out.append(r)
    return out


def speaker_split(pairs, val_speakers, test_speakers):
    """
    Split by SPEAKER, not by recording. Splitting recordings randomly puts the same
    speaker on both sides and produces a validation score that does not predict
    performance on anyone new.
    """
    val_speakers = set(map(str, val_speakers))
    test_speakers = set(map(str, test_speakers))
    tr, va, te = [], [], []
    for p in pairs:
        s = str(p.get("speaker"))
        (te if s in test_speakers else va if s in val_speakers else tr).append(p)
    return tr, va, te


def noise_only_paths(pairs_json, root="dataset"):
    """Real noise-only recordings, for optional augmentation."""
    recs = json.load(open(pairs_json, encoding="utf-8"))
    return [r["microphone"] for r in recs.values()
            if "only" in (r.get("noise") or "").lower()]
