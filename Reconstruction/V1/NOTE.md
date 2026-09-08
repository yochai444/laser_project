# V1 — original code, unmodified

These files are exactly as uploaded. Nothing has been changed, including the
issues described in the top-level README:

- the optical channel enters as an input, not as a supervisory target, so the
  laser is required at inference
- the noisy signal is synthesised by adding white noise from a single file
  (`microphone_177.wav`, speaker 04, "White Noise Only") at a random SNR
- alignment is recomputed inside `__getitem__` with a 2^20-point FFT, once per
  sample per epoch
- `random_split` places the same speakers on both sides of the split
- `imshow` in `inference.py` is called without `vmin`/`vmax`, so the two
  spectrogram panels are scaled independently and are not comparable by eye

They are preserved unchanged so the reported V1 results remain reproducible.
`unet_model_best.pt` is the corresponding checkpoint.

First-layer weight analysis of that checkpoint: the optical channel has an L2
norm of 2.428 against 2.234 for the microphone channel and dominates in 25 of
32 filters, so the model was using the optical input — for a task whose
supervision was synthetic.
