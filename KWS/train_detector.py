#!/usr/bin/env python3
"""
train_detector.py
=====================
Trains a detector for "was the word spoken at all", as opposed to the localizer,
which answers "where in this recording is it".

WHY A SEPARATE MODEL
--------------------
V5 is trained with a softmax over the time axis. That normalisation makes the
scores sum to one within a recording, so the output can only express WHERE the
best moment is - it is mathematically incapable of expressing that no moment is
good. The per-frame BCE term carried weight 0.3 and was never what checkpoints
were selected on.

detection_check.py measured the cost: AUC 0.604 for presence, against 0.500 for
a coin, with five independent framings of the measurement agreeing. That number
describes a model asked a question it was never trained on. It does not tell us
whether the task is possible - only that V5 does not solve it.

WHAT IS DIFFERENT HERE
----------------------
  OBJECTIVE. No softmax over time. Each crop gets one score and one binary
  label, trained with BCE. The model is free to output low everywhere.

  NEGATIVES ARE ABUNDANT. Every positive recording holds ~18 s of ordinary book
  reading outside the keyword. Sampled as 3 s crops that is about six negatives
  per recording, redrawn at fresh offsets every epoch - on the order of 1600
  distinct negative crops per epoch from 268 recordings, with no new audio.

  HARD NEGATIVE MINING. The failure mode is a moment of ordinary reading that
  resembles the keyword. In the 156-688 Hz band the /h/ and /p/ are gone, so
  "held", "hell" and "help" are nearly identical. After a warmup, negative crops
  are drawn from wherever the model currently scores highest, concentrating
  training on exactly those confusions.

  POOLING THAT MATCHES THE DECISION. The recording-level score pools frame
  scores with log-sum-exp rather than a bare max, so one freak frame no longer
  decides the answer. Sustained evidence does, which is what a real keyword
  produces and a chance resemblance does not.

Evaluation is recording-level AUC on a speaker the model never saw, against two
negative sets: the same recordings with the keyword excised, and the 13
genuinely keyword-free recordings. The bar to beat is 0.604.

Usage:
    python train_detector_mac.py --laser-dir ./laser --meta metadata_clean.csv \
        --smoke-test

    python train_detector_mac.py --laser-dir ./laser --meta metadata_clean.csv
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from freq_warp import random_warp

from train_help_v5_mac import (SR, REC_SEC, OUT_FPS, AudioCache, Features,
                               HelpLocalizerV5, get_device, n_out_frames,
                               seed_all)

DET_CROP_SEC = 3.0     # long enough for context, short enough for many negatives
GUARD_S = 0.7          # exceeds half the 1.26 s receptive field
CROSSFADE_S = 0.02
EVAL_STRIDE = 0.5


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class Detector(nn.Module):
    """The V5 backbone, with the softmax replaced by a pooling head.

    The backbone still emits one logit per 20 ms frame. What changes is how
    those become a decision: log-sum-exp with temperature tau interpolates
    between the mean (tau large) and the max (tau small). At tau = 1 a recording
    needs several agreeing frames to score highly, which is the difference
    between a spoken word and a chance resemblance.
    """

    def __init__(self, n_bins, width=32, drop=0.15, tau=1.0):
        super().__init__()
        self.backbone = HelpLocalizerV5(n_bins, width=width, drop=drop)
        self.tau = tau

    def frame_logits(self, x):
        return self.backbone(x)

    def pool(self, logits):
        n = logits.shape[1]
        return self.tau * (torch.logsumexp(logits / self.tau, dim=1)
                           - math.log(n))

    def forward(self, x):
        fl = self.frame_logits(x)
        return self.pool(fl), fl


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def negative_spans(row):
    """Regions of a recording that provably contain no keyword."""
    if not row.has_keyword or not np.isfinite(row.start_s):
        return [(0.0, REC_SEC)]
    spans = []
    if row.start_s - GUARD_S > DET_CROP_SEC:
        spans.append((0.0, row.start_s - GUARD_S))
    if REC_SEC - (row.end_s + GUARD_S) > DET_CROP_SEC:
        spans.append((row.end_s + GUARD_S, REC_SEC))
    return spans


def candidate_negative_starts(row, stride=0.5):
    starts = []
    for lo, hi in negative_spans(row):
        s = lo
        while s + DET_CROP_SEC <= hi + 1e-9:
            starts.append(round(s, 3))
            s += stride
    return starts


class CropDataset(Dataset):
    """One item = a 3 s crop with a binary label.

    Positive crops place the word at a random position inside the crop, so the
    model cannot key on the word sitting in the middle. Negative crops come from
    `neg_index`, which the trainer replaces with mined hard negatives once the
    warmup is over.
    """

    def __init__(self, meta, cache, neg_index, neg_per_pos=3, augment=True,
                 warp_pct=0.0):
        self.meta = meta.reset_index(drop=True)
        self.cache = cache
        self.neg_index = neg_index          # {recording_id: [start_s, ...]}
        self.neg_per_pos = neg_per_pos
        self.augment = augment
        self.warp_pct = warp_pct
        self.feat = Features()
        self.n_samples = int(DET_CROP_SEC * SR)

        self.pos_rows = [r for r in self.meta.itertuples()
                         if r.has_keyword and np.isfinite(r.start_s)]
        self.neg_rows = [r for r in self.meta.itertuples()
                         if candidate_negative_starts(r)]
        self.items = self._build()

    def _build(self):
        items = [(r.recording_id, None, 1.0) for r in self.pos_rows]
        for _ in range(self.neg_per_pos):
            for r in self.neg_rows:
                items.append((r.recording_id, "neg", 0.0))
        return items

    def resample(self):
        """Called between epochs so crop offsets are redrawn."""
        self.items = self._build()

    def __len__(self):
        return len(self.items)

    def _positive_start(self, row):
        # The word may sit anywhere inside the crop, edges excluded.
        lo = max(0.0, row.end_s + 0.15 - DET_CROP_SEC)
        hi = min(REC_SEC - DET_CROP_SEC, row.start_s - 0.15)
        if hi <= lo:
            return float(np.clip(0.5 * (row.start_s + row.end_s) - DET_CROP_SEC / 2,
                                 0.0, REC_SEC - DET_CROP_SEC))
        return random.uniform(lo, hi)

    def __getitem__(self, i):
        rid, kind, label = self.items[i]
        row = self.meta[self.meta.recording_id == rid].iloc[0]
        full = self.cache.get(rid)

        if kind is None:
            start = self._positive_start(row)
        else:
            opts = self.neg_index.get(rid) or candidate_negative_starts(row)
            if not opts:
                start, label = self._positive_start(row), 1.0
            else:
                start = random.choice(opts)

        beg = int(start * SR)
        seg = full[beg: beg + self.n_samples]
        if len(seg) < self.n_samples:
            seg = np.pad(seg, (0, self.n_samples - len(seg)))
        wav = torch.from_numpy(np.ascontiguousarray(seg))

        if self.augment:
            wav = wav * random.uniform(0.7, 1.4)
            if random.random() < 0.4:
                rms = wav.pow(2).mean().sqrt().clamp_min(1e-8)
                snr = random.uniform(12.0, 32.0)
                wav = wav + torch.randn_like(wav) * (rms / (10 ** (snr / 20)))

        x = self.feat(wav)

        if self.augment and self.warp_pct > 0:
            x = random_warp(x, self.feat, self.warp_pct)

        if self.augment:
            _, nf, nt = x.shape
            if random.random() < 0.3 and nf > 8:
                w = random.randint(1, 3)
                s = random.randint(0, nf - w)
                x[:, s:s + w, :] = 0.0
            if random.random() < 0.3 and nt > 30:
                w = random.randint(3, 12)
                s = random.randint(0, nt - w)
                x[:, :, s:s + w] = 0.0

        return x, torch.tensor(label, dtype=torch.float32)


# ---------------------------------------------------------------------------
# hard negative mining
# ---------------------------------------------------------------------------

@torch.no_grad()
def mine_hard_negatives(model, meta, cache, device, feat, keep_frac=0.25,
                        max_per_rec=6):
    """Rebuild the negative pool from wherever the model currently scores high.

    Uniformly sampled negatives quickly become trivial: most of the reading
    sounds nothing like the keyword, so they stop producing gradient. Keeping
    the top-scoring candidates puts the training set exactly where the errors
    are.
    """
    model.eval()
    index = {}
    for row in meta.itertuples():
        starts = candidate_negative_starts(row)
        if not starts:
            continue
        n = int(DET_CROP_SEC * SR)
        full = cache.get(row.recording_id)
        batch = []
        for s in starts:
            b = int(s * SR)
            seg = full[b: b + n]
            if len(seg) < n:
                seg = np.pad(seg, (0, n - len(seg)))
            batch.append(feat(torch.from_numpy(np.ascontiguousarray(seg))))
        x = torch.stack(batch).to(device)
        scores, _ = model(x)
        scores = scores.float().cpu().numpy()
        order = np.argsort(-scores)
        keep = max(1, min(max_per_rec, int(len(starts) * keep_frac)))
        index[row.recording_id] = [starts[k] for k in order[:keep]]
    return index


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def excise(wav, start_s, end_s, guard=GUARD_S):
    a = max(0, int((start_s - guard) * SR))
    b = min(len(wav), int((end_s + guard) * SR))
    if b <= a:
        return wav.copy()
    left, right = wav[:a], wav[b:]
    n = int(CROSSFADE_S * SR)
    if len(left) < n or len(right) < n:
        return np.concatenate([left, right])
    ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    join = left[-n:] * (1 - ramp) + right[:n] * ramp
    return np.concatenate([left[:-n], join, right[n:]])


@torch.no_grad()
def recording_score(model, wav, feat, device, stride=EVAL_STRIDE):
    """Slide the crop over a whole recording; return the pooled score."""
    n = int(DET_CROP_SEC * SR)
    starts = np.arange(0.0, max(len(wav) / SR - DET_CROP_SEC, 0.0) + 1e-9, stride)
    if not len(starts):
        starts = np.array([0.0])
    batch = []
    for s in starts:
        b = int(s * SR)
        seg = wav[b: b + n]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        batch.append(feat(torch.from_numpy(np.ascontiguousarray(seg))))
    x = torch.stack(batch).to(device)
    scores, _ = model(x)
    s = scores.float().cpu().numpy()
    return {"max": float(s.max()),
            "top3": float(np.sort(s)[-3:].mean()) if len(s) >= 3 else float(s.max()),
            "lse": float(np.log(np.exp(s - s.max()).mean()) + s.max())}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def metrics_at(pos, neg, thr):
    """Recall, specificity and balanced accuracy at a given threshold.

    recall      share of keyword recordings the detector fires on
    specificity share of keyword-free recordings it correctly stays silent on
    balanced    the mean of the two, which is what "accuracy" should mean when
                the two classes are not equally common

    AUC alone hides which of the two errors dominates. A detector that fires on
    everything and one that fires on nothing can both look mediocre by AUC, but
    they fail in opposite directions and call for opposite fixes.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    recall = float((pos >= thr).mean()) if len(pos) else float("nan")
    spec = float((neg < thr).mean()) if len(neg) else float("nan")
    return {"recall": recall, "specificity": spec,
            "balanced": 0.5 * (recall + spec)}


def best_balanced_threshold(pos, neg):
    """Threshold maximising balanced accuracy on the data given."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    cand = np.unique(np.concatenate([pos, neg]))
    best, bt = -1.0, float(cand[0])
    for t in cand:
        v = 0.5 * ((pos >= t).mean() + (neg < t).mean())
        if v > best:
            best, bt = v, float(t)
    return bt, best


def eer_and_acc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    cand = np.unique(np.concatenate([pos, neg]))
    bacc, bt = 0.0, cand[0]
    for t in cand:
        v = 0.5 * ((pos >= t).mean() + (neg < t).mean())
        if v > bacc:
            bacc, bt = v, t
    gap, et = 1.0, cand[0]
    for t in cand:
        d = abs((pos < t).mean() - (neg >= t).mean())
        if d < gap:
            gap, et = d, t
    return bacc, float(bt), 0.5 * ((pos < et).mean() + (neg >= et).mean())


@torch.no_grad()
def evaluate(model, meta, cache, device, feat):
    """Recording-level scores for positives, excised negatives, real negatives."""
    model.eval()
    pos, exc, real = {"max": [], "top3": [], "lse": []}, \
                     {"max": [], "top3": [], "lse": []}, \
                     {"max": [], "top3": [], "lse": []}
    for row in meta.itertuples():
        if row.recording_id not in cache:
            continue
        wav = cache.get(row.recording_id)
        s = recording_score(model, wav, feat, device)
        if row.has_keyword and np.isfinite(row.start_s):
            for k in pos:
                pos[k].append(s[k])
            s2 = recording_score(model, excise(wav, row.start_s, row.end_s),
                                 feat, device)
            for k in exc:
                exc[k].append(s2[k])
        else:
            for k in real:
                real[k].append(s[k])
    return pos, exc, real


# ---------------------------------------------------------------------------

def run_fold(meta, cache, test_spk, val_spk, args, device, out_root):
    fold = f"test_{test_spk}_val_{val_spk}"
    out = Path(out_root) / fold
    out.mkdir(parents=True, exist_ok=True)

    te = meta[meta.speaker_id == test_spk]
    va = meta[meta.speaker_id == val_spk]
    tr = meta[~meta.speaker_id.isin([test_spk, val_spk])]

    print("\n" + "=" * 72)
    print(f"FOLD {fold}   train {sorted(tr.speaker_id.unique())} ({len(tr)} rec)  "
          f"val {val_spk} ({len(va)})  test {test_spk} ({len(te)})")
    print("=" * 72)

    feat = Features()
    model = Detector(feat.n_bins, width=args.width, drop=args.dropout,
                     tau=args.tau).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    n_neg = sum(len(candidate_negative_starts(r)) for r in tr.itertuples())
    print(f"{n_par/1000:.0f}k parameters, crop {DET_CROP_SEC:.0f}s, "
          f"{n_out_frames(int(DET_CROP_SEC*SR))} frames per crop")
    print(f"{len(tr)} recordings -> {n_neg} candidate negative crop positions")

    neg_index = {r.recording_id: candidate_negative_starts(r)
                 for r in tr.itertuples()}
    ds = CropDataset(tr, cache, neg_index, neg_per_pos=args.neg_per_pos,
                     warp_pct=args.warp)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best, best_state, best_ep, history = -1.0, None, 0, []
    best_thr = 0.0

    for ep in range(1, args.epochs + 1):
        if ep > args.mine_after and (ep - args.mine_after - 1) % args.mine_every == 0:
            neg_index = mine_hard_negatives(model, tr, cache, device, feat,
                                            keep_frac=args.mine_keep)
            ds.neg_index = neg_index
        ds.resample()
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, drop_last=True)

        model.train()
        tot, n, correct = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            score, _ = model(x)
            loss = F.binary_cross_entropy_with_logits(score, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item() * x.size(0)
            correct += ((score > 0).float() == y).sum().item()
            n += x.size(0)
        sched.step()

        vp, vx, vr = evaluate(model, va, cache, device, feat)
        v_auc = auc(vp["lse"], vx["lse"])
        v_thr, _ = best_balanced_threshold(vp["lse"], vx["lse"])
        vm = metrics_at(vp["lse"], vx["lse"], v_thr)
        mined = "mined" if ep > args.mine_after else "uniform"
        print(f"  epoch {ep:02d}/{args.epochs}  loss={tot/max(n,1):.4f}  "
              f"crop_acc={correct/max(n,1):.1%}  {mined:<7}  "
              f"val_AUC={v_auc:.3f}  bal={vm['balanced']:.1%}  "
              f"recall={vm['recall']:.1%}  spec={vm['specificity']:.1%}")
        history.append({"epoch": ep, "loss": tot / max(n, 1),
                        "crop_acc": correct / max(n, 1), "val_auc": v_auc,
                        "val_threshold": v_thr, **{f"val_{k}": v
                                                   for k, v in vm.items()}})

        score = v_auc if args.select_by == "auc" else vm["balanced"]
        if score > best:
            best, best_ep, best_thr = score, ep, v_thr
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n  selected epoch {best_ep} (val {args.select_by} {best:.3f}, "
          f"threshold {best_thr:+.3f})")

    tp, tx, trn = evaluate(model, te, cache, device, feat)
    result = {"fold": fold, "test_speaker": test_spk, "val_speaker": val_spk,
              "best_epoch": best_ep, "val_auc": best, "history": history}

    result["val_threshold"] = best_thr

    print(f"\n  TEST speaker {test_spk} - recording-level detection")
    print(f"  {'pooling':<8} {'AUC':>7} {'bal.acc':>9} {'recall':>8} "
          f"{'spec':>7} {'EER':>7}   negatives")
    for pool in ("max", "top3", "lse"):
        a = auc(tp[pool], tx[pool])
        bacc, thr, eer = eer_and_acc(tp[pool], tx[pool])
        m = metrics_at(tp[pool], tx[pool], thr)
        print(f"  {pool:<8} {a:7.3f} {bacc:9.1%} {m['recall']:8.1%} "
              f"{m['specificity']:7.1%} {eer:7.1%}   {len(tx[pool])} excised")
        result[f"auc_{pool}_excised"] = a
        result[f"bacc_{pool}_excised"] = bacc
        result[f"eer_{pool}_excised"] = eer
        result[f"recall_{pool}_excised"] = m["recall"]
        result[f"spec_{pool}_excised"] = m["specificity"]

    # The row above tunes the threshold on the test set, so it is an upper
    # bound rather than a measurement. Repeating it at the threshold chosen on
    # the validation speaker is the number that transfers to new recordings.
    mv = metrics_at(tp["lse"], tx["lse"], best_thr)
    print(f"\n  at the threshold picked on the validation speaker "
          f"({best_thr:+.3f}), which is the honest operating point:")
    print(f"    balanced {mv['balanced']:.1%}   recall {mv['recall']:.1%}   "
          f"specificity {mv['specificity']:.1%}")
    result.update({f"held_{k}": v for k, v in mv.items()})

    if len(trn["lse"]) >= 5:
        a = auc(tp["lse"], trn["lse"])
        mr = metrics_at(tp["lse"], trn["lse"], best_thr)
        print(f"\n  against the {len(trn['lse'])} genuinely keyword-free "
              f"recordings: AUC {a:.3f}, specificity {mr['specificity']:.1%}")
        print("    (too few to measure with; the detector also saw them in "
              "training)")
        result["auc_lse_real"] = a
        result["spec_real"] = mr["specificity"]

    pd.DataFrame({"score_lse_positive": tp["lse"]}).to_csv(
        out / "test_scores_positive.csv", index=False)
    pd.DataFrame({"score_lse_excised": tx["lse"]}).to_csv(
        out / "test_scores_negative.csv", index=False)
    torch.save({"model_state_dict": model.state_dict(), "n_bins": feat.n_bins,
                "width": args.width, "tau": args.tau}, out / "detector.pt")
    with open(out / "final_results_detector.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="metadata_clean.csv")
    ap.add_argument("--laser-dir", required=True)
    ap.add_argument("--output-dir", default="outputs/detector")
    ap.add_argument("--test-speaker", default="all")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--tau", type=float, default=1.0,
                    help="pooling temperature; smaller behaves more like a max")
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--mine-after", type=int, default=5,
                    help="epochs of uniform negatives before mining starts")
    ap.add_argument("--mine-every", type=int, default=3)
    ap.add_argument("--mine-keep", type=float, default=0.25)
    ap.add_argument("--select-by", choices=["auc", "balanced"], default="auc",
                    help="checkpoint criterion; auc is threshold-free and less "
                         "noisy, balanced matches the reported headline")
    ap.add_argument("--warp", type=float, default=0.0,
                    help="frequency warp range in percent, e.g. 15. 0 disables.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    seed_all(args.seed)
    device = get_device(args.device)
    print(f"device: {device}")

    meta = pd.read_csv(args.meta, dtype={"recording_id": str, "speaker_id": str})
    meta = meta[~meta.exclude].reset_index(drop=True)

    if args.smoke_test:
        args.epochs, args.mine_after, args.test_speaker = 4, 2, "07"
        meta = meta.groupby("speaker_id").head(12).reset_index(drop=True)
        print(f"SMOKE TEST: {len(meta)} recordings")

    print("loading audio into memory")
    cache = AudioCache(args.laser_dir, meta.recording_id.tolist(), "laser")
    meta = meta[meta.recording_id.isin(cache.data.keys())].reset_index(drop=True)
    print(f"{len(meta)} recordings, {int(meta.has_keyword.sum())} with the keyword, "
          f"{int((~meta.has_keyword).sum())} genuinely without")

    speakers = sorted(meta.speaker_id.unique())
    wanted = (speakers if args.test_speaker == "all"
              else [s.strip() for s in args.test_speaker.split(",") if s.strip()])
    folds = [(s, speakers[(speakers.index(s) + 1) % len(speakers)])
             for s in wanted if s in speakers]

    results = [run_fold(meta, cache, t, v, args, device, args.output_dir)
               for t, v in folds]

    if results:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"}
                           for r in results])
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(args.output_dir) / "detector_summary.csv", index=False)
        print("\n" + "=" * 72)
        print("SUMMARY - recording-level detection AUC")
        print("=" * 72)
        cols = [c for c in ["fold", "auc_lse_excised", "held_balanced",
                            "held_recall", "held_specificity"] if c in df]
        print(df[cols].to_string(index=False))
        print(f"\nmean AUC          {df.auc_lse_excised.mean():.3f}")
        if "held_balanced" in df:
            print(f"mean balanced     {df.held_balanced.mean():.1%}")
            print(f"mean recall       {df.held_recall.mean():.1%}   "
                  f"(share of HELP recordings caught)")
            print(f"mean specificity  {df.held_specificity.mean():.1%}   "
                  f"(share of keyword-free ones correctly rejected)")
        print("\nAll three use the threshold chosen on the validation speaker, "
              "not on the test\nspeaker. The V5 localizer scored AUC 0.604 on "
              "this question; 0.500 is a coin.")


if __name__ == "__main__":
    main()
