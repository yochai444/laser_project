# V3 — Optical-Guided Speech Enhancement

## Goal
Final deployment:
`noisy microphone -> enhanced microphone`

The laser is used only during training as auxiliary speech-activity supervision.
It is NOT needed at inference.

## Files
- `dataset_v3.py` — chooses clean microphone recordings, aligns paired laser,
  creates noisy inputs on the fly.
- `model_v3.py` — 2D U-Net predicting a time-frequency mask.
- `loss_v3.py` — clean-microphone reconstruction + ideal-ratio-mask loss +
  optical activity guidance.
- `train_v3.py` — speaker-disjoint training / validation / test.
- `infer_v3.py` — microphone-only inference.

## First run

Put these files next to `pairs.json` and your `laser/` and `microphone/` folders.

Install:
```bash
pip install torch torchaudio soundfile numpy
```

Train:
```bash
python train_v3.py --pairs pairs.json --val-speakers 05 --test-speakers 06
```

Outputs:
```text
runs/optical_guided_v3/
    best.pt
    history.json
    test.json
```

## Critical ablation experiment
Train the exact same model WITHOUT optical guidance:
```bash
python train_v3.py --pairs pairs.json --val-speakers 05 --test-speakers 06 --lambda-optical 0 --out runs/v3_no_optical
```

Then compare `runs/optical_guided_v3/test.json` with `runs/v3_no_optical/test.json`.
That directly tests whether the optical signal added value.

## Inference
```bash
python infer_v3.py --ckpt runs/optical_guided_v3/best.pt --mic microphone/microphone_300.wav --out enhanced_300_v3.wav
```

No laser file is required at inference.
