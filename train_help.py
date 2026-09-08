#!/usr/bin/env python3
"""
train_help.py
====================
Fifth-generation HELP keyword localizer for the optical microphone corpus.
Built for Apple Silicon (MPS), falls back to CUDA or CPU.

WHY THIS IS NOT V3 WITH DIFFERENT HYPERPARAMETERS
-------------------------------------------------
Rescoring V3's own predictions showed it losing to a constant-time guess on all
seven folds: 52.8% mean top-1 localization against an 85.6% baseline that
ignores the audio entirely. Three causes were identified, and V5 addresses each.

1. BANDWIDTH. Measured microphone-to-laser coherence peaks at 190-310 Hz and
   collapses past 700 Hz. V3 fed the model 50-1500 Hz, so roughly 61% of its
   input bins carried no speech - and per-frequency CMVN then rescaled that
   noise to the same variance as the signal, actively amplifying it. V5 uses
   156-688 Hz, the measured band, at twice the frequency resolution.

2. OBJECTIVE MISMATCH. V3 trained a binary classifier on independent 1 s
   windows, selected checkpoints by window-level Average Precision, and decoded
   with a threshold. But the corpus is offline with exactly one keyword per
   recording, so the real question is "which moment in THIS recording", a
   comparison among the windows of one file. Window-level AP measures ranking
   across the whole corpus and is blind to that. V5 optimizes the actual
   question: a softmax over the time axis of one recording, trained with cross
   entropy against the annotated position.

3. THE CLOCK. The keyword always falls between 8.7 s and 13.1 s, which is why a
   fixed guess scores 85.6%. Two properties keep V5 from exploiting it. The
   network is fully convolutional with weights shared across time, so it has no
   representation of absolute position and *cannot* encode "answer at 11 s".
   And training crops are placed so the word lands uniformly anywhere inside
   them, which removes the regularity from the data as well.

The 3 recordings whose keyword V3 silently labelled negative (185, 293, 299 -
their annotated duration exceeds 2 s, so no 1 s window could reach the 50%
coverage threshold) are handled correctly here, because labels are per frame.

USAGE
-----
    # sanity pass first: 2 speakers, few recordings, 3 epochs, a couple of minutes
    python train_help_v5_mac.py --laser-dir ./laser --mic-dir ./microphone \
        --meta metadata_clean.csv --smoke-test

    # one fold, comparable to V3's test_07_val_01
    python train_help_v5_mac.py --laser-dir ./laser --mic-dir ./microphone \
        --meta metadata_clean.csv --test-speaker 07 --val-speaker 01

    # full leave-one-speaker-out
    python train_help_v5_mac.py --laser-dir ./laser --mic-dir ./microphone \
        --meta metadata_clean.csv --test-speaker all

Each fold writes test_predictions_v5.csv in the same schema V3 used, so
rescore_topk.py runs on it unchanged and the two are directly comparable.
"""

import argparse
import json
import math
import os
import random
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from torch.utils.data import DataLoader, Dataset

from freq_warp import random_warp

# ---------------------------------------------------------------------------
# Signal constants. F_MIN/F_MAX come from band_coherence.py on this corpus.
# ---------------------------------------------------------------------------
SR = 16000
N_FFT = 1024          # 15.6 Hz resolution; 512 was too coarse for a 530 Hz band
HOP = 160             # 10 ms -> 100 frames/s
WIN_LENGTH = 640      # 40 ms
F_MIN = 150.0         # measured coherent band; override with --f-min/--f-max
F_MAX = 700.0

TIME_POOL = 2         # network output stride: 100 fps / 2 = 50 fps = 20 ms
OUT_FPS = (SR / HOP) / TIME_POOL

CROP_SEC = 8.0        # training crop; the word sits uniformly anywhere inside
CROP_MARGIN = 0.4     # keep the word this far from the crop edges
REC_SEC = 20.0

HIT_TOL = 0.5         # peak within this many seconds of the word centre = hit


def ce_floor(meta):
    """Lowest cross entropy achievable on this data.

    The soft target is a Gaussian, and cross entropy against it cannot go below
    the target's own entropy. Printing this alongside the training loss is what
    separates "the model has not learned" from "the model has fit the training
    data and the problem is generalisation" - two situations that call for
    opposite fixes.
    """
    kw = meta[meta.has_keyword & np.isfinite(meta.start_s)]
    if not len(kw):
        return float("nan")
    sigma = np.maximum(0.5 * kw.duration_s.to_numpy(), 0.12) * OUT_FPS
    return float(np.mean(np.log(sigma) + 0.5 * math.log(2 * math.pi * math.e)))


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer=None):
    if prefer:
        return torch.device(prefer)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# audio loading
# ---------------------------------------------------------------------------

def index_wavs(directory):
    """Map recording id -> path, keyed by the filename tail.

    Keys are strings, not integers. The laser corpus uses ids like "001", but
    the LibriSpeech corpus built by prep_librispeech.py uses "ls00000" for
    positives and "lsn00000" for negatives. Parsing out the digits would map
    both of those to 0 and silently collide, and int("ls00000") would raise
    outright. Each file is registered under its tail and, when that tail is
    numeric, under the unpadded number as well, so "Laser_001.wav" answers to
    both "001" and "1".
    """
    idx = {}
    for p in sorted(glob(os.path.join(directory, "*.wav"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        tail = stem.split("_")[-1]
        idx.setdefault(tail, p)
        if tail.isdigit():
            idx.setdefault(str(int(tail)), p)
    return idx


def lookup(idx, rid):
    rid = str(rid)
    if rid in idx:
        return idx[rid]
    if rid.isdigit() and str(int(rid)) in idx:
        return idx[str(int(rid))]
    return None


def read_wav(path):
    sr, x = wavfile.read(path)
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / float(np.iinfo(x.dtype).max)
    else:
        x = x.astype(np.float32)
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}")
    n = int(REC_SEC * SR)
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    return x[:n]


class AudioCache:
    """Holds every waveform in RAM. 369 x 20 s x 16 kHz x float32 = 472 MB per
    channel, which a Mac handles comfortably and which removes disk I/O as a
    bottleneck - V3's lru_cache(128) was thrashing against ~254 train files."""

    def __init__(self, directory, recording_ids, label=""):
        idx = index_wavs(directory)
        self.data = {}
        missing = []
        for rid in recording_ids:
            path = lookup(idx, rid)
            if path is None:
                missing.append(rid)
                continue
            try:
                self.data[rid] = read_wav(path)
            except Exception:
                missing.append(rid)
        mb = sum(a.nbytes for a in self.data.values()) / 1e6
        print(f"  cached {len(self.data)} {label} recordings ({mb:.0f} MB)"
              + (f", {len(missing)} unavailable" if missing else ""))
        self.missing = set(missing)

    def __contains__(self, rid):
        return rid in self.data

    def get(self, rid):
        return self.data[rid]


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def sliding_cmvn(x, win_frames):
    """Per-frequency mean/variance normalisation over a moving window.

    Normalising over the whole segment makes the features depend on the segment
    length: training sees 8 s crops and inference sees 20 s recordings, so the
    same audio produced different inputs in the two regimes and the weights were
    being applied to a distribution they were never trained on. A fixed-width
    moving window gives identical statistics either way.

    x: [F, T] -> [F, T]. Computed with cumulative sums, so cost is O(T) rather
    than O(T * win).
    """
    f_bins, t = x.shape
    w = min(win_frames | 1, t if t % 2 else t - 1)   # odd, and never longer than t
    if w < 3:
        mu = x.mean(dim=1, keepdim=True)
        sd = x.std(dim=1, keepdim=True).clamp_min(1e-4)
        return (x - mu) / sd

    pad = (w - 1) // 2
    xp = F.pad(x.unsqueeze(0), (pad, pad), mode="reflect").squeeze(0)

    zeros = torch.zeros(f_bins, 1, dtype=x.dtype)
    c1 = torch.cat([zeros, xp.cumsum(dim=1)], dim=1)
    c2 = torch.cat([zeros, (xp * xp).cumsum(dim=1)], dim=1)

    mean = (c1[:, w:] - c1[:, :-w]) / w
    mean2 = (c2[:, w:] - c2[:, :-w]) / w
    std = (mean2 - mean * mean).clamp_min(1e-8).sqrt().clamp_min(1e-4)
    return (x - mean) / std


def mel_filterbank(f_min, f_max, n_bands, sr=SR, n_fft=N_FFT):
    """Triangular filters on a mel scale, returned as [n_bands, n_fft//2+1].

    This exists so bandwidth can be changed without changing anything else.
    Taking raw FFT bins over 50-7500 Hz gives 477 input rows against 35 for
    150-700 Hz, which inflates the temporal stack from 0.34M to 2.37M
    parameters. A wider model on the same 268 recordings would memorise more, so
    a worse result could not be attributed to bandwidth rather than capacity.
    Fixing the number of bands holds capacity and speed constant and varies only
    the information reaching the network.
    """
    def to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def from_mel(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    edges = from_mel(np.linspace(to_mel(f_min), to_mel(f_max), n_bands + 2))

    fb = np.zeros((n_bands, len(freqs)), dtype=np.float32)
    for i in range(n_bands):
        lo, mid, hi = edges[i], edges[i + 1], edges[i + 2]
        rise = (freqs >= lo) & (freqs <= mid)
        fall = (freqs > mid) & (freqs <= hi)
        if mid > lo:
            fb[i, rise] = (freqs[rise] - lo) / (mid - lo)
        if hi > mid:
            fb[i, fall] = (hi - freqs[fall]) / (hi - mid)
        s = fb[i].sum()
        if s > 0:
            fb[i] /= s          # unit area, so band width does not set the level
    return torch.from_numpy(fb)


class Features:
    """Log-magnitude STFT restricted to a frequency range, normalised with a
    moving window so that 8 s crops and 20 s recordings look alike.

    With use_filterbank the range is summarised by a fixed number of mel bands,
    which keeps the model identical across bandwidth settings.
    """

    def __init__(self, norm_win_sec=3.0, f_min=F_MIN, f_max=F_MAX,
                 use_filterbank=False, n_bands=35):
        self.norm_win = int(round(norm_win_sec * SR / HOP))
        self.window = torch.hann_window(WIN_LENGTH)
        self.use_filterbank = use_filterbank

        freqs = torch.fft.rfftfreq(N_FFT, d=1.0 / SR)
        if use_filterbank:
            self.fb = mel_filterbank(f_min, f_max, n_bands)
            self.bins = None
            self.n_bins = n_bands
            self.f_lo, self.f_hi = float(f_min), float(f_max)
        else:
            self.bins = torch.where((freqs >= f_min) & (freqs <= f_max))[0]
            self.n_bins = len(self.bins)
            self.f_lo = float(freqs[self.bins[0]])
            self.f_hi = float(freqs[self.bins[-1]])

    def __call__(self, wav):
        """wav: 1-D float tensor -> [1, F, T]"""
        spec = torch.stft(wav, n_fft=N_FFT, hop_length=HOP, win_length=WIN_LENGTH,
                          window=self.window, center=True, return_complex=True)
        mag = spec.abs()
        mag = self.fb @ mag if self.use_filterbank else mag[self.bins, :]
        lg = torch.log1p(mag * 100.0)
        return sliding_cmvn(lg, self.norm_win).unsqueeze(0)


# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------

def n_out_frames(n_samples):
    """Frames the network emits for a waveform of this length."""
    return (1 + n_samples // HOP) // TIME_POOL


class LocalizerDataset(Dataset):
    """One item = one random crop of one recording, plus per-frame targets.

    The crop placement is the part that matters. For a recording that contains
    the keyword, the crop start is drawn so that the word can land anywhere
    inside the crop with only a small edge margin. Across an epoch the word's
    position within the input is close to uniform, so "always answer near the
    middle" is not a winning strategy for the network to discover.
    """

    def __init__(self, meta, cache, train=True, augment=True, repeats=1,
                 feat=None, warp_pct=0.0):
        # Every item is a fresh random crop, so visiting each recording several
        # times per epoch multiplies the effective training set at no I/O cost.
        base = meta.reset_index(drop=True)
        self.rows = (pd.concat([base] * repeats, ignore_index=True)
                     if repeats > 1 else base)
        self.cache = cache
        self.train = train
        self.augment = augment and train
        # The feature extractor is passed in, never built here. Building it
        # with defaults meant the dataset and the model could disagree about how
        # many frequency rows exist - editing F_MIN/F_MAX at the top of the file
        # changed the dataset's view while the model still followed the command
        # line flags, and the mismatch only surfaced as a shape error deep inside
        # the first forward pass.
        self.feat = feat if feat is not None else Features()
        self.warp_pct = warp_pct
        self.crop_samples = int(CROP_SEC * SR)

    def __len__(self):
        return len(self.rows)

    # -- crop placement ----------------------------------------------------
    def pick_crop(self, row, speed=1.0):
        """Start time of the input window, in seconds.

        The window read from disk spans CROP_SEC * speed seconds, because
        resampling compresses or stretches it to CROP_SEC afterwards. Bounds are
        computed in that input domain; using CROP_SEC directly would let the
        window run past the end of the recording at speed > 1 and silently pad
        the tail with zeros.
        """
        win = CROP_SEC * speed
        margin = CROP_MARGIN * speed
        max_start = max(0.0, REC_SEC - win)

        if not row.has_keyword or not np.isfinite(row.start_s):
            return random.uniform(0.0, max_start)

        # window must contain [start_s, end_s] with margin on both sides
        lo = max(0.0, row.end_s + margin - win)
        hi = min(max_start, row.start_s - margin)
        if hi <= lo:
            return float(np.clip(0.5 * (row.start_s + row.end_s) - win / 2,
                                 0.0, max_start))
        return random.uniform(lo, hi)

    # -- augmentation ------------------------------------------------------
    def speed_perturb(self, wav, factor, out_len):
        """Resample so that output sample i reads input sample i * factor.

        out_len is passed in rather than inferred, because the caller needs a
        fixed-size crop and the input deliberately has a different length. The
        earlier version derived the output length from the input and then
        truncated to it, which returned a short crop whenever factor < 1 and
        broke batching. The mapping here is the one the label arithmetic below
        assumes: a word at input offset d appears at output offset d / factor.
        """
        idx = np.arange(out_len, dtype=np.float32) * factor
        src = np.arange(len(wav), dtype=np.float32)
        return np.interp(idx, src, wav).astype(np.float32)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        full = self.cache.get(row.recording_id)

        # speed first: the crop bounds depend on how much input is needed
        speed = 1.0
        if self.augment and random.random() < 0.5:
            speed = random.uniform(0.92, 1.08)
        crop_start = self.pick_crop(row, speed)

        # Enough input samples for the resampler to reach the last output
        # sample: index (crop_samples - 1) * speed, plus a margin for interp.
        need = int(math.ceil(self.crop_samples * speed)) + 2
        beg = int(crop_start * SR)
        seg = full[beg: beg + need]
        if len(seg) < need:
            seg = np.pad(seg, (0, need - len(seg)))

        if speed != 1.0:
            seg = self.speed_perturb(seg, speed, self.crop_samples)
        else:
            seg = seg[: self.crop_samples]

        # A silent length mismatch here is what broke batching before, so make
        # it impossible rather than merely unlikely.
        assert len(seg) == self.crop_samples, (
            f"crop is {len(seg)} samples, expected {self.crop_samples} "
            f"(speed={speed:.3f})")

        wav = torch.from_numpy(np.ascontiguousarray(seg))

        if self.augment:
            if random.random() < 0.5:
                wav = wav * random.uniform(0.7, 1.4)
            if random.random() < 0.4:
                rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
                snr = random.uniform(12.0, 32.0)
                wav = wav + torch.randn_like(wav) * (rms / (10 ** (snr / 20)))

        x = self.feat(wav)

        if self.augment and self.warp_pct > 0:
            # Vocal tract length perturbation. The measured coherence peak sits
            # at 188-312 Hz, so the fundamental is the strongest surviving
            # feature - and there are five voices to learn it from.
            x = random_warp(x, self.feat, self.warp_pct)

        if self.augment:
            _, nf, nt = x.shape
            if random.random() < 0.3 and nf > 8:
                w = random.randint(1, 3)
                s = random.randint(0, nf - w)
                x[:, s:s + w, :] = 0.0
            if random.random() < 0.3 and nt > 40:
                w = random.randint(3, 15)
                s = random.randint(0, nt - w)
                x[:, :, s:s + w] = 0.0

        T = n_out_frames(self.crop_samples)
        soft = torch.zeros(T)
        frame = torch.zeros(T)
        has_kw = bool(row.has_keyword) and np.isfinite(row.start_s)

        if has_kw:
            # positions inside the crop, in seconds, after speed perturbation
            s_rel = (row.start_s - crop_start) / speed
            e_rel = (row.end_s - crop_start) / speed
            centre = 0.5 * (s_rel + e_rel)

            if 0.0 <= centre <= CROP_SEC:
                t = (torch.arange(T, dtype=torch.float32) + 0.5) / OUT_FPS
                sigma = max(0.5 * (e_rel - s_rel), 0.12)
                soft = torch.exp(-0.5 * ((t - centre) / sigma) ** 2)
                soft = soft / soft.sum().clamp_min(1e-8)
                frame = ((t >= s_rel - 0.05) & (t <= e_rel + 0.05)).float()
            else:
                has_kw = False  # word fell outside the crop; treat as negative

        return x, soft, frame, torch.tensor(float(has_kw))


def collate(batch):
    xs, softs, frames, has = zip(*batch)
    return (torch.stack(xs), torch.stack(softs),
            torch.stack(frames), torch.stack(has))


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, c, drop):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
            nn.ReLU(inplace=True), nn.Dropout2d(drop),
            nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.b(x))


class HelpLocalizerV5(nn.Module):
    """Fully convolutional: [B,1,F,T] -> one logit per output time step.

    Frequency is pooled away; time is pooled exactly once, to 50 fps. Nothing in
    the architecture depends on absolute position, so the network cannot learn
    the recording schedule even if that would lower the loss. The dilated 1-D
    stack at the end gives each output frame about 1.3 s of context, enough to
    see a whole word plus its surroundings.
    """

    def __init__(self, n_bins, width=32, drop=0.15):
        super().__init__()
        c1, c2, c3 = width, width * 2, width * 3

        self.blocks = nn.Sequential(
            nn.Conv2d(1, c1, 3, padding=1, bias=False), nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),                       # frequency only
            ResBlock(c1, drop * 0.6),

            nn.Conv2d(c1, c2, 3, padding=1, bias=False), nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, TIME_POOL)),               # the one time pool
            ResBlock(c2, drop),

            nn.Conv2d(c2, c3, 3, padding=1, bias=False), nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            ResBlock(c3, drop),
        )

        f_out = n_bins
        for _ in range(3):
            f_out //= 2
        self.f_out = max(f_out, 1)

        ch = c3 * self.f_out
        self.temporal = nn.Sequential(
            nn.Conv1d(ch, 128, 3, padding=1, dilation=1, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=8, dilation=8, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, 3, padding=16, dilation=16, bias=False),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Dropout(drop * 2),
            nn.Conv1d(128, 1, 1),
        )

    def forward(self, x):
        h = self.blocks(x)                   # [B, C, F', T']
        b, c, f, t = h.shape
        h = h.reshape(b, c * f, t)
        return self.temporal(h).squeeze(1)   # [B, T']


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------

def localization_loss(logits, soft, frame, has_kw, bce_weight=0.3, pos_weight=8.0):
    """Softmax-over-time cross entropy, plus a per-frame detection term.

    The cross entropy is the objective that matches evaluation: it asks the
    network to put its probability mass on the right moment of this recording,
    which is exactly the top-1 question. The BCE term keeps the raw scores
    calibrated so a recording with no keyword produces low values everywhere -
    a softmax alone always sums to one and could never express absence.
    """
    logp = F.log_softmax(logits, dim=1)
    ce_per = -(soft * logp).sum(dim=1)
    denom = has_kw.sum().clamp_min(1.0)
    ce = (ce_per * has_kw).sum() / denom

    bce = F.binary_cross_entropy_with_logits(
        logits, frame,
        pos_weight=torch.tensor(pos_weight, device=logits.device))

    return ce + bce_weight * bce, ce.detach(), bce.detach()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def score_recordings(model, meta, cache, device, feat):
    """Run the model over each full 20 s recording; return peak time and curve."""
    model.eval()
    out = []
    for row in meta.itertuples():
        if row.recording_id not in cache:
            continue
        wav = torch.from_numpy(cache.get(row.recording_id))
        x = feat(wav).unsqueeze(0).to(device)
        logits = model(x).squeeze(0).float().cpu()
        t = (torch.arange(len(logits), dtype=torch.float32) + 0.5) / OUT_FPS
        k = int(torch.argmax(logits))
        out.append({
            "recording_id": row.recording_id,
            "speaker_id": row.speaker_id,
            "has_keyword": bool(row.has_keyword),
            "help_start": row.start_s,
            "help_end": row.end_s,
            "peak_time": float(t[k]),
            "peak_logit": float(logits[k]),
            "logits": logits.numpy(),
            "times": t.numpy(),
        })
    return out


def validation_ce(scored):
    """Cross entropy of the full-recording prediction against the true position.

    top-1 accuracy over ~49 recordings moves in 2% steps and swung 20 points
    between epochs, which is far too noisy to select a checkpoint on - doing so
    picked epoch 1 in the first run. This is the same objective the model is
    trained with, measured on held-out speakers, and it varies smoothly.
    """
    vals = []
    for s_ in scored:
        if not (s_["has_keyword"] and np.isfinite(s_["help_start"])):
            continue
        logits = torch.as_tensor(s_["logits"], dtype=torch.float32)
        t = torch.as_tensor(s_["times"], dtype=torch.float32)
        centre = 0.5 * (s_["help_start"] + s_["help_end"])
        sigma = max(0.5 * (s_["help_end"] - s_["help_start"]), 0.12)
        soft = torch.exp(-0.5 * ((t - centre) / sigma) ** 2)
        soft = soft / soft.sum().clamp_min(1e-8)
        vals.append(float(-(soft * F.log_softmax(logits, dim=0)).sum()))
    return float(np.mean(vals)) if vals else float("nan")


def localization_accuracy(scored, tol=HIT_TOL):
    kw = [s for s in scored if s["has_keyword"] and np.isfinite(s["help_start"])]
    if not kw:
        return float("nan"), 0
    ok = sum(1 for s in kw
             if abs(s["peak_time"] - 0.5 * (s["help_start"] + s["help_end"])) <= tol)
    return ok / len(kw), len(kw)


def fixed_time_baseline(fit_meta, eval_scored, tol=HIT_TOL):
    """Best constant answer learned from one speaker set, applied to another.

    Fitting on the training speakers and scoring on the test speaker is the
    honest version of this control: it is what a model that only learned the
    experiment's schedule would achieve.
    """
    fit = fit_meta[fit_meta.has_keyword & np.isfinite(fit_meta.start_s)]
    if not len(fit):
        return float("nan"), float("nan")
    centres = (0.5 * (fit.start_s + fit.end_s)).to_numpy()
    grid = np.arange(0.0, REC_SEC, 0.02)
    best_t = grid[int(np.argmax([(np.abs(centres - g) <= tol).sum() for g in grid]))]

    kw = [s for s in eval_scored if s["has_keyword"] and np.isfinite(s["help_start"])]
    if not kw:
        return float("nan"), float(best_t)
    hits = sum(1 for s in kw
               if abs(best_t - 0.5 * (s["help_start"] + s["help_end"])) <= tol)
    return hits / len(kw), float(best_t)


def write_v3_style_predictions(scored, path, window=1.0, stride=0.5):
    """Re-express frame scores as 1 s windows so rescore_topk.py can read them."""
    rows = []
    for s in scored:
        t, lg = s["times"], s["logits"]
        prob = 1.0 / (1.0 + np.exp(-lg))
        ws = 0.0
        while ws + window <= REC_SEC + 1e-9:
            m = (t >= ws) & (t < ws + window)
            p = float(prob[m].max()) if m.any() else 0.0
            hs, he = s["help_start"], s["help_end"]
            if np.isfinite(hs):
                ov = max(0.0, min(ws + window, he) - max(ws, hs))
                cov = ov / (he - hs) if he > hs else 0.0
            else:
                cov = 0.0
            rows.append({
                "recording_id": s["recording_id"],
                "recorder_id": s["speaker_id"],
                "recording_duration": REC_SEC,
                "window_start": ws, "window_end": ws + window,
                "help_start": hs, "help_end": he,
                "coverage": cov,
                "label": int(cov >= 0.5),
                "true_label": int(cov >= 0.5),
                "probability": p,
            })
            ws += stride
    pd.DataFrame(rows).to_csv(path, index=False)


# ---------------------------------------------------------------------------
# one fold
# ---------------------------------------------------------------------------

def run_fold(meta, laser, mic, test_spk, val_spk, args, device, out_root):
    # --channel mic is the control experiment: same pipeline, clean audio. If it
    # also plateaus near 45%, the ceiling is this model or this formulation, not
    # the optical channel - and no amount of better signal extraction from the
    # .mat files would help.
    data = mic if args.channel == "mic" else laser
    fold = f"test_{test_spk}_val_{val_spk}"
    out = Path(out_root) / fold
    out.mkdir(parents=True, exist_ok=True)

    te = meta[meta.speaker_id == test_spk]
    va = meta[meta.speaker_id == val_spk]
    tr = meta[~meta.speaker_id.isin([test_spk, val_spk])]

    print("\n" + "=" * 72)
    print(f"FOLD {fold}   train {sorted(tr.speaker_id.unique())}  "
          f"({len(tr)} rec)  val {val_spk} ({len(va)})  test {test_spk} ({len(te)})")
    print("=" * 72)

    feat = Features(norm_win_sec=args.norm_win, f_min=args.f_min,
                    f_max=args.f_max, use_filterbank=args.filterbank,
                    n_bands=args.n_bands)
    model = HelpLocalizerV5(feat.n_bins, width=args.width, drop=args.dropout).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    floor = ce_floor(tr)

    if args.init_from:
        # Stage 2 of the two-stage plan: start from weights learned on a corpus
        # with many speakers. Training here uses 5 voices and the model memorises
        # them - best epoch was 1, 1, 3, 8, 4, 15, 2 across the seven folds while
        # training cross entropy sat 0.06 above its floor. Weights that already
        # encode what the word sounds like across many speakers give the fine
        # tune something general to adapt rather than something to memorise.
        ck = torch.load(args.init_from, map_location="cpu")
        if ck.get("n_bins") != feat.n_bins:
            raise SystemExit(
                f"{args.init_from} was trained with {ck.get('n_bins')} frequency "
                f"bins but this run uses {feat.n_bins}. The feature settings "
                "(--f-min/--f-max/--filterbank/--n-bands) must match.")
        model.load_state_dict(ck["model_state_dict"])
        print(f"initialised from {args.init_from}")
    print(f"channel {args.channel}, band {feat.f_lo:.0f}-{feat.f_hi:.0f} Hz, "
          f"{feat.n_bins} bins, {n_par/1000:.0f}k parameters, {OUT_FPS:.0f} fps out")
    print(f"cross-entropy floor for this training set: {floor:.3f} "
          f"(uniform over an 8 s crop would be {math.log(n_out_frames(int(CROP_SEC*SR))):.3f})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)

    def loader_for(df, cache, train):
        ds = LocalizerDataset(df, cache, train=train, augment=train,
                              repeats=args.repeats if train else 1, feat=feat,
                              warp_pct=args.warp if train else 0.0)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=train,
                          num_workers=args.num_workers, collate_fn=collate,
                          drop_last=train and len(ds) > args.batch_size)

    # -- optional pretraining on the synchronised microphone -----------------
    # The feature extractor already restricts everything to 156-688 Hz, so mic
    # and laser features live in the same space; no extra filtering is needed.
    # This is the only stage that uses more data than the laser corpus offers.
    if args.pretrain_epochs > 0 and mic is not None and args.channel != "mic":
        print(f"\npretraining {args.pretrain_epochs} epochs on the microphone channel")
        pre = loader_for(tr, mic, True)
        for ep in range(1, args.pretrain_epochs + 1):
            ce, bce = train_epoch(model, pre, opt, device, args)
            print(f"  pretrain {ep:02d}/{args.pretrain_epochs}  "
                  f"ce={ce:.3f} (floor {floor:.3f})  bce={bce:.3f}")
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.5

    train_loader = loader_for(tr, data, True)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = float("inf")          # selection is on validation CE, lower is better
    best_acc, best_ep = 0.0, 0
    best_state = None
    history = []

    for ep in range(1, args.epochs + 1):
        tr_ce, tr_bce = train_epoch(model, train_loader, opt, device, args)
        sched.step()

        val_scored = score_recordings(model, va, data, device, feat)
        vce = validation_ce(val_scored)
        vacc, vn = localization_accuracy(val_scored)
        vbase, vt = fixed_time_baseline(tr, val_scored)

        gap = vce - tr_ce
        print(f"  epoch {ep:02d}/{args.epochs}  train_ce={tr_ce:.3f} "
              f"(floor {floor:.3f}, gap {tr_ce - floor:+.3f})  "
              f"val_ce={vce:.3f}  generalization_gap={gap:+.3f}  "
              f"val_top1={vacc:.1%} (base {vbase:.1%})")
        history.append({"epoch": ep, "train_ce": tr_ce, "train_bce": tr_bce,
                        "val_ce": vce, "val_top1": vacc, "val_baseline": vbase,
                        "ce_floor": floor})

        if vce < best:
            best, best_acc, best_ep = vce, vacc, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    print(f"\n  selected epoch {best_ep} (val_ce {best:.3f}, val_top1 {best_acc:.1%})")
    if best_state:
        model.load_state_dict(best_state)
    torch.save({"model_state_dict": model.state_dict(),
                "n_bins": feat.n_bins, "width": args.width},
               out / "best_help_v5.pt")

    test_scored = score_recordings(model, te, data, device, feat)
    tacc, tn = localization_accuracy(test_scored)
    tbase, tt = fixed_time_baseline(tr, test_scored)
    err_red = (tacc - tbase) / (1 - tbase) if tbase < 1 else float("nan")

    print(f"\n  TEST speaker {test_spk}: top-1 {tacc:.1%} over {tn} recordings")
    print(f"  fixed-time baseline (fitted on train speakers): {tbase:.1%} at t={tt:.1f}s")
    print(f"  errors removed vs baseline: {err_red:+.1%}")

    write_v3_style_predictions(test_scored, out / "test_predictions_v5.csv")
    write_v3_style_predictions(
        score_recordings(model, va, data, device, feat),
        out / "validation_predictions_v5.csv")

    pd.DataFrame([{k: v for k, v in s.items() if k not in ("logits", "times")}
                  for s in test_scored]).to_csv(out / "test_peaks_v5.csv", index=False)

    result = {"fold": fold, "test_speaker": test_spk, "val_speaker": val_spk,
              "test_top1": tacc, "test_baseline": tbase, "baseline_time": tt,
              "errors_removed": err_red, "val_ce_best": best,
              "val_top1_at_best": best_acc, "best_epoch": best_ep,
              "ce_floor": floor, "channel": args.channel,
              "n_test": tn, "history": history}
    with open(out / "final_results_v5.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def train_epoch(model, loader, opt, device, args):
    model.train()
    tot_ce, tot_bce, n = 0.0, 0.0, 0
    for x, soft, frame, has in loader:
        x, soft = x.to(device), soft.to(device)
        frame, has = frame.to(device), has.to(device)

        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss, ce, bce = localization_loss(logits, soft, frame, has,
                                          bce_weight=args.bce_weight)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        b = x.size(0)
        tot_ce += ce.item() * b
        tot_bce += bce.item() * b
        n += b
    return tot_ce / max(n, 1), tot_bce / max(n, 1)


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="metadata_clean.csv")
    ap.add_argument("--laser-dir", required=True)
    ap.add_argument("--mic-dir", default=None)
    ap.add_argument("--output-dir", default="outputs/help_v5")
    ap.add_argument("--test-speaker", default="07", help="'all' for full LOSO")
    ap.add_argument("--val-speaker", default=None)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--pretrain-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=4,
                    help="random crops drawn per recording per epoch")
    ap.add_argument("--bce-weight", type=float, default=0.3)
    ap.add_argument("--channel", choices=["laser", "mic"], default="laser",
                    help="'mic' runs the control experiment on clean audio")
    ap.add_argument("--norm-win", type=float, default=3.0,
                    help="seconds of context for the moving CMVN window")
    ap.add_argument("--f-min", type=float, default=F_MIN)
    ap.add_argument("--f-max", type=float, default=F_MAX)
    ap.add_argument("--filterbank", action="store_true",
                    help="summarise the band with a fixed number of mel bands, "
                         "so model size stays constant when bandwidth changes")
    ap.add_argument("--n-bands", type=int, default=35)
    ap.add_argument("--warp", type=float, default=0.0,
                    help="frequency warp range in percent, e.g. 15. 0 disables.")
    ap.add_argument("--init-from", default=None,
                    help="checkpoint to start from, e.g. weights pretrained on "
                         "LibriSpeech. Feature settings must match.")
    ap.add_argument("--val-fraction", type=float, default=0.0,
                    help="hold out this fraction of speakers for validation "
                         "instead of naming one. Use on corpora with many "
                         "speakers, where naming a single speaker is pointless.")
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true",
                    help="tiny run to prove the pipeline works before committing")
    args = ap.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    print(f"device: {device}")

    meta = pd.read_csv(args.meta, dtype={"recording_id": str, "speaker_id": str})
    meta = meta[~meta.exclude].reset_index(drop=True)

    if args.smoke_test:
        args.epochs = 3
        args.pretrain_epochs = 1
        args.repeats = 2
        meta = meta.groupby("speaker_id").head(12).reset_index(drop=True)
        print(f"SMOKE TEST: {len(meta)} recordings, {args.epochs} epochs")

    print("loading audio into memory")
    laser = AudioCache(args.laser_dir, meta.recording_id.tolist(), "laser")
    need_mic = args.mic_dir and (args.pretrain_epochs > 0 or args.channel == "mic")
    mic = (AudioCache(args.mic_dir, meta.recording_id.tolist(), "mic")
           if need_mic else None)
    if args.channel == "mic" and mic is None:
        raise SystemExit("--channel mic requires --mic-dir")

    keep = (mic if args.channel == "mic" else laser).data.keys()
    meta = meta[meta.recording_id.isin(keep)].reset_index(drop=True)
    print(f"{len(meta)} usable recordings, "
          f"{int(meta.has_keyword.sum())} with the keyword")

    speakers = sorted(meta.speaker_id.unique())

    if args.val_fraction > 0:
        # Many-speaker mode: one run, with a random slice of speakers held out.
        # Leave-one-speaker-out over 129 speakers would mean 129 trainings and
        # would measure nothing useful - this corpus is for pretraining, not for
        # evaluation.
        rng = random.Random(args.seed)
        shuffled = speakers[:]
        rng.shuffle(shuffled)
        n_hold = max(1, int(len(shuffled) * args.val_fraction))
        held = set(shuffled[:n_hold])
        meta = meta.copy()
        meta["speaker_id"] = np.where(meta.speaker_id.isin(held), "__val", "__train")
        print(f"holding out {n_hold} of {len(speakers)} speakers for validation")
        folds = [("__val", "__val")]
    elif args.test_speaker == "all":
        folds = [(s, speakers[(i + 1) % len(speakers)])
                 for i, s in enumerate(speakers)]
    else:
        v = args.val_speaker or speakers[
            (speakers.index(args.test_speaker) + 1) % len(speakers)]
        folds = [(args.test_speaker, v)]

    results = []
    for test_spk, val_spk in folds:
        results.append(run_fold(meta, laser, mic, test_spk, val_spk,
                                args, device, args.output_dir))

    if len(results) > 1:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"}
                           for r in results])
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(args.output_dir) / "cross_validation_summary_v5.csv", index=False)
        print("\n" + "=" * 72)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 72)
        print(df[["fold", "test_top1", "test_baseline", "errors_removed"]]
              .to_string(index=False))
        print(f"\nmean top-1 {df.test_top1.mean():.1%} "
              f"vs baseline {df.test_baseline.mean():.1%}")


if __name__ == "__main__":
    main()
