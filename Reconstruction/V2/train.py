"""
Stage 5 — training.

Changes from the previous version, and why:

  splits      Speaker-disjoint train/val/test. The old script used random_split over
              recordings, so the same speakers appeared on both sides and the
              validation loss did not measure generalisation to a new talker. It also
              had no test set at all, and selected the best checkpoint on the same
              validation set it then reported.

  QC gate     Only pairs whose optical and acoustic channels are actually coherent are
              used. In this corpus only about a third of the 369 pairs clear
              msc_max > 0.35, and they are heavily concentrated in speakers 06 and 07.
              Training on the rest asks the model to regress onto a target that does
              not contain the input's speech.

  weighting   Remaining pairs are weighted by their coherence, so a marginal pair
              contributes proportionally less than a clean one.

  scheduling  Cosine schedule with warmup, and AdamW. The old script used a constant
              1e-4 for 50 epochs.

Usage:
    python train.py --pairs pairs.json --val-speakers 05 --test-speakers 06
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import OpticalPairDataset, load_pairs, speaker_split, noise_only_paths
from loss import MultiResBandLoss, si_sdr
from model import OpticalUNet


def build_loader(pairs, args, train, noise_pool=None):
    ds = OpticalPairDataset(
        pairs,
        target_sr=args.sr,
        chunk_sec=args.chunk_sec,
        highpass_hz=args.highpass,
        noise_pool=noise_pool if train else None,
        noise_prob=args.noise_prob if train else 0.0,
        train=train,
        keyword_bias=args.keyword_bias if train else 0.0,
    )
    return DataLoader(
        ds, batch_size=args.batch_size, shuffle=train,
        num_workers=args.workers, pin_memory=True, drop_last=train,
    )


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    tot_loss, tot_sdr, n = 0.0, 0.0, 0

    for mic, opt, w in loader:
        mic, opt, w = mic.to(device), opt.to(device), w.to(device)
        with torch.set_grad_enabled(train):
            pred = model(mic)
            loss = criterion(pred, opt, weight=w)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        with torch.no_grad():
            tot_sdr += si_sdr(pred.squeeze(1), opt.squeeze(1)).mean().item() * mic.size(0)
        tot_loss += loss.item() * mic.size(0)
        n += mic.size(0)

    return tot_loss / max(n, 1), tot_sdr / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.json")
    ap.add_argument("--val-speakers", nargs="+", default=["05"])
    ap.add_argument("--test-speakers", nargs="+", default=["06"])
    ap.add_argument("--min-msc", type=float, default=0.35)
    ap.add_argument("--min-delay-conf", type=float, default=3.0)
    ap.add_argument("--sr", type=int, default=4000)
    ap.add_argument("--chunk-sec", type=float, default=2.0)
    ap.add_argument("--highpass", type=float, default=150.0)
    ap.add_argument("--band", type=float, nargs=2, default=[150.0, 1400.0])
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--noise-prob", type=float, default=0.0,
                    help="probability of mixing a REAL noise-only recording into the input")
    ap.add_argument("--keyword-bias", type=float, default=0.3)
    ap.add_argument("--out", default="runs/optical_unet")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs(args.pairs, min_msc=args.min_msc, min_delay_conf=args.min_delay_conf)
    tr, va, te = speaker_split(pairs, args.val_speakers, args.test_speakers)

    print(f"usable pairs after QC: {len(pairs)}")
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}")
    for name, subset in (("train", tr), ("val", va), ("test", te)):
        spk = sorted({str(p['speaker']) for p in subset})
        print(f"  {name:5s} speakers: {spk}")
    if not tr or not va:
        raise SystemExit("empty train or val split — check --val-speakers / --test-speakers")

    noise_pool = noise_only_paths(args.pairs) if args.noise_prob > 0 else None

    train_loader = build_loader(tr, args, True, noise_pool)
    val_loader = build_loader(va, args, False)
    test_loader = build_loader(te, args, False) if te else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpticalUNet(in_channels=1).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M   device: {device}")

    criterion = MultiResBandLoss(sample_rate=args.sr, band=tuple(args.band)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    def lr_at(ep):
        if ep < args.warmup:
            return (ep + 1) / args.warmup
        p = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1 + math.cos(math.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    best = float("inf")
    history = []
    for ep in range(args.epochs):
        tr_loss, tr_sdr = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_sdr = run_epoch(model, val_loader, criterion, device)
        sched.step()

        history.append(dict(epoch=ep + 1, train_loss=tr_loss, val_loss=va_loss,
                            train_sdr=tr_sdr, val_sdr=va_sdr))
        flag = ""
        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "args": vars(args)}, out / "best.pt")
            flag = "  <- best"
        print(f"epoch {ep+1:3d}/{args.epochs}  train {tr_loss:.4f} (SI-SDR {tr_sdr:+.2f} dB)"
              f"  val {va_loss:.4f} (SI-SDR {va_sdr:+.2f} dB){flag}")

    json.dump(history, open(out / "history.json", "w"), indent=1)

    if test_loader:
        model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
        te_loss, te_sdr = run_epoch(model, test_loader, criterion, device)
        print(f"\nHELD-OUT TEST (speakers {args.test_speakers}): "
              f"loss {te_loss:.4f}  SI-SDR {te_sdr:+.2f} dB")
        json.dump({"test_loss": te_loss, "test_si_sdr": te_sdr},
                  open(out / "test.json", "w"), indent=1)


if __name__ == "__main__":
    main()
