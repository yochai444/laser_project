"""
V3 training: optical-guided microphone speech enhancement.

Final task:
    noisy microphone -> enhanced/clean microphone

Laser role:
    auxiliary supervision during training only.

Recommended first run:
    python train_v3.py --pairs pairs.json --val-speakers 05 --test-speakers 06

Ablation without laser:
    python train_v3.py --pairs pairs.json --val-speakers 05 --test-speakers 06 ^
        --lambda-optical 0 --out runs/v3_no_optical
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_v3 import (
    V3EnhancementDataset,
    load_v3_pairs,
    speaker_split,
    noise_only_paths,
)
from model_v3 import OpticalGuidedEnhancer
from loss_v3 import V3Loss, si_sdr


def build_loader(pairs, args, train, noise_paths=None):
    ds = V3EnhancementDataset(
        pairs,
        target_sr=args.sr,
        chunk_sec=args.chunk_sec,
        highpass_hz=args.highpass,
        train=train,
        real_noise_paths=noise_paths if train else None,
        real_noise_prob=args.real_noise_prob if train else 0.0,
        gaussian_prob=args.gaussian_prob if train else 0.0,
        snr_range=tuple(args.snr_range),
        keyword_bias=args.keyword_bias if train else 0.0,
    )
    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


@torch.no_grad()
def stft_corr(model, pred, clean):
    P = torch.log1p(model.stft(pred).abs())
    S = torch.log1p(model.stft(clean).abs())
    a = P.flatten(1) - P.flatten(1).mean(1, keepdim=True)
    b = S.flatten(1) - S.flatten(1).mean(1, keepdim=True)
    den = torch.sqrt((a.square().sum(1) * b.square().sum(1)).clamp_min(1e-8))
    return ((a * b).sum(1) / den).mean()


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)

    sums = {
        "loss": 0.0,
        "wave": 0.0,
        "spec": 0.0,
        "mask": 0.0,
        "optical": 0.0,
        "before_sdr": 0.0,
        "after_sdr": 0.0,
        "before_stft": 0.0,
        "after_stft": 0.0,
    }
    n = 0

    for batch in loader:
        noisy = batch["noisy"].to(device)
        clean = batch["clean"].to(device)
        optical = batch["optical"].to(device)
        ow = batch["optical_weight"].to(device)

        with torch.set_grad_enabled(train):
            out = model(noisy)
            losses = criterion(out, noisy, clean, optical, ow)
            loss = losses["total"]

        if not torch.isfinite(loss):
            print("[warning] non-finite loss; batch skipped")
            continue

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        bs = noisy.size(0)
        with torch.no_grad():
            before_sdr = si_sdr(noisy.squeeze(1), clean.squeeze(1)).mean()
            after_sdr = si_sdr(out["enhanced"].squeeze(1), clean.squeeze(1)).mean()
            before_stft = stft_corr(model, noisy, clean)
            after_stft = stft_corr(model, out["enhanced"], clean)

        sums["loss"] += loss.item() * bs
        for k in ("wave", "spec", "mask", "optical"):
            sums[k] += float(losses[k].item()) * bs
        sums["before_sdr"] += float(before_sdr.item()) * bs
        sums["after_sdr"] += float(after_sdr.item()) * bs
        sums["before_stft"] += float(before_stft.item()) * bs
        sums["after_stft"] += float(after_stft.item()) * bs
        n += bs

    if n == 0:
        return {k: float("nan") for k in sums}
    return {k: v / n for k, v in sums.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="pairs.json")
    ap.add_argument("--val-speakers", nargs="+", default=["05"])
    ap.add_argument("--test-speakers", nargs="+", default=["06"])
    ap.add_argument("--min-delay-conf", type=float, default=3.0)
    ap.add_argument("--min-msc", type=float, default=0.0)

    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--chunk-sec", type=float, default=2.0)
    ap.add_argument("--highpass", type=float, default=80.0)
    ap.add_argument("--n-fft", type=int, default=512)
    ap.add_argument("--hop", type=int, default=128)
    ap.add_argument("--win", type=int, default=512)

    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--snr-range", type=float, nargs=2, default=[0.0, 20.0])
    ap.add_argument("--real-noise-prob", type=float, default=0.5)
    ap.add_argument("--gaussian-prob", type=float, default=0.5)
    ap.add_argument("--keyword-bias", type=float, default=0.30)

    ap.add_argument("--lambda-wave", type=float, default=0.5)
    ap.add_argument("--lambda-spec", type=float, default=1.0)
    ap.add_argument("--lambda-mask", type=float, default=0.5)
    ap.add_argument("--lambda-optical", type=float, default=0.25)

    ap.add_argument("--out", default="runs/optical_guided_v3")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    pairs = load_v3_pairs(
        args.pairs,
        min_delay_conf=args.min_delay_conf,
        min_msc=args.min_msc,
    )
    tr, va, te = speaker_split(pairs, args.val_speakers, args.test_speakers)

    print(f"clean-source pairs: {len(pairs)}")
    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}")
    for name, subset in (("train", tr), ("val", va), ("test", te)):
        speakers = sorted({str(x.get('speaker')) for x in subset})
        print(f"  {name:5s} speakers: {speakers}")

    if not tr or not va:
        raise SystemExit("empty train or val split; check speakers / clean labels")

    noise_paths = noise_only_paths(args.pairs)
    if noise_paths:
        print(f"real noise-only recordings available: {len(noise_paths)}")
    else:
        print("no real noise-only recordings found; training will use Gaussian noise")

    train_loader = build_loader(tr, args, True, noise_paths)
    val_loader = build_loader(va, args, False)
    test_loader = build_loader(te, args, False) if te else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OpticalGuidedEnhancer(
        n_fft=args.n_fft,
        hop=args.hop,
        win=args.win,
    ).to(device)

    criterion = V3Loss(
        sample_rate=args.sr,
        n_fft=args.n_fft,
        hop=args.hop,
        win=args.win,
        lambda_wave=args.lambda_wave,
        lambda_spec=args.lambda_spec,
        lambda_mask=args.lambda_mask,
        lambda_optical=args.lambda_optical,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_lambda(ep):
        if ep < args.warmup:
            return (ep + 1) / max(1, args.warmup)
        p = (ep - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(
        f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M"
        f"   device: {device}"
    )

    history = []
    best = float("inf")

    for ep in range(args.epochs):
        trm = run_epoch(model, train_loader, criterion, device, optimizer)
        vam = run_epoch(model, val_loader, criterion, device)
        scheduler.step()

        row = {
            "epoch": ep + 1,
            "train": trm,
            "val": vam,
        }
        history.append(row)
        json.dump(history, open(outdir / "history.json", "w"), indent=1)

        flag = ""
        if vam["loss"] < best:
            best = vam["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                },
                outdir / "best.pt",
            )
            flag = "  <- best"

        print(
            f"epoch {ep+1:3d}/{args.epochs}"
            f"  train {trm['loss']:.4f}"
            f"  val {vam['loss']:.4f}"
            f"  SI-SDR {vam['before_sdr']:+.2f}->{vam['after_sdr']:+.2f} dB"
            f"  STFT {vam['before_stft']:+.3f}->{vam['after_stft']:+.3f}"
            f"  opt {vam['optical']:.4f}{flag}"
        )

    if test_loader:
        ck = torch.load(outdir / "best.pt", map_location=device)
        model.load_state_dict(ck["model"])
        tem = run_epoch(model, test_loader, criterion, device)

        result = {
            "speakers": args.test_speakers,
            "metrics": tem,
            "si_sdr_improvement_db": tem["after_sdr"] - tem["before_sdr"],
            "stft_corr_improvement": tem["after_stft"] - tem["before_stft"],
        }
        json.dump(result, open(outdir / "test.json", "w"), indent=1)

        print("\nHELD-OUT TEST")
        print(f"  speakers: {args.test_speakers}")
        print(f"  SI-SDR:   {tem['before_sdr']:+.2f} -> {tem['after_sdr']:+.2f} dB"
              f"  (Δ {tem['after_sdr'] - tem['before_sdr']:+.2f} dB)")
        print(f"  STFT corr:{tem['before_stft']:+.3f} -> {tem['after_stft']:+.3f}"
              f"  (Δ {tem['after_stft'] - tem['before_stft']:+.3f})")


if __name__ == "__main__":
    main()
