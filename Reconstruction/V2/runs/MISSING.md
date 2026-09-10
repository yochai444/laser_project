# V3 — source code not included

This folder holds only the artefacts produced by your V3 runs:

- `v3_best.pt` — trained checkpoint
- `history.json` — 120 epochs of training and validation loss
- `test.json` — held-out test result (`test_si_sdr: -24.17`)

**The training code is not here.** `train_v3.py` was run on your machine and never
shared, so it is not in this package and has not been reconstructed. A
plausible-looking substitute would be worse than an obvious gap: anyone reading the
report would assume it was the code that produced the results, and it would not be.

To complete the package, copy in your own `train_v3.py`, the model definition it
imports, and any dataset or loss files specific to V3.

## What is known about V3 from the results

- Input includes both the microphone and the optical channel; the ablation was run
  as `--lambda-optical 0`, which suggests the optical guidance enters through a
  weighted loss term rather than only through the input.
- The laser is required at inference.
- Full bandwidth is preserved in the output: on recording 250 the applied gain was
  −0.05 dB in 150–700 Hz and −4.33 dB in 5000–8000 Hz, with a standard deviation of
  4.03 dB across time and frequency — a genuine time–frequency mask, not a level change.
- Effective SNR gain +4.80 dB on that recording, with full bandwidth retained.

## Two measurements to keep with the code

The applied gain correlated **0.741** with a microphone-derived activity curve and
**0.157** with the optical activity curve. Together with the ablation
(+8.19 dB without the optical channel against +8.09 dB with it), this indicates the
model is performing microphone-based masking and the optical input contributes
nothing measurable.

`history.json` also shows `train_loss` above 10⁵ in 32 of 120 epochs. The cause and
the fix are described in the top-level README; if V3 reuses the original `loss.py`,
apply the same change before drawing conclusions from a rerun.
