#!/usr/bin/env python3
"""
predict.py
==========
Takes a WAV from the optical receiver and returns the probability that the word
"help" was spoken in it.

WHAT IT DOES
------------
Loads every per-fold detector produced by train_detector_mac.py and runs all of
them. Each fold was trained without one speaker, so no single model has seen
every speaking style; averaging them is both more accurate and more stable than
picking one arbitrarily.

CALIBRATION IS THE PART THAT MATTERS
------------------------------------
A model's raw output is a score, not a probability. A score of +2.1 does not
mean anything on its own. Each fold's score is therefore mapped to a probability
with a logistic fit on that fold's held-out test scores - the recordings from
the speaker that model never saw. That makes "0.78" mean roughly what it says,
rather than being a number that merely sorts correctly.

HONEST ACCURACY
---------------
Measured across seven held-out speakers, balanced accuracy is about 78%, ranging
from 93% for the best speaker to 61% for the worst. AUC averages 0.81. Those
figures come from negatives synthesised by cutting the keyword out of real
recordings; only 13 genuinely keyword-free recordings exist, too few to measure
against. The number this tool prints should be read as a strong hint, not a
verdict.

USAGE
-----
    python predict.py recording.wav
    python predict.py folder_of_wavs/ --csv results.csv
    python predict.py --serve            # drag-and-drop page at localhost:8000
"""

import argparse
import io
import json
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
from scipy import signal as sps
from scipy.io import wavfile
from scipy.optimize import minimize

from train_detector_mac import (DET_CROP_SEC, EVAL_STRIDE, Detector,
                                recording_score)
from train_help_v5_mac import SR, OUT_FPS, Features, get_device


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def fit_platt(pos_scores, neg_scores):
    """Logistic map from score to probability, fitted on held-out scores.

    Platt scaling: p = sigmoid(a * s + b), with a and b chosen to minimise
    negative log likelihood on data the model was never trained on. Fitting on
    training scores would produce badly overconfident probabilities, because the
    model separates its own training data far better than anything new.
    """
    s = np.concatenate([np.asarray(pos_scores, float),
                        np.asarray(neg_scores, float)])
    y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])

    # Mild smoothing of the targets keeps the fit from running to infinity when
    # the two score sets happen to be perfectly separable.
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    t = np.where(y > 0, (n_pos + 1) / (n_pos + 2), 1.0 / (n_neg + 2))

    def nll(p):
        a, b = p
        z = a * s + b
        return float(np.mean(np.logaddexp(0, z) - t * z))

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 4000})
    a, b = res.x
    if a <= 0:      # a decreasing fit would invert the meaning of the score
        a, b = 1e-3, 0.0
    return float(a), float(b)


def load_ensemble(model_dir, device):
    """Load every fold's detector together with its own calibration.

    model_dir may be several directories separated by commas. Training the same
    folds under different --seed values and pooling them here averages away the
    run-to-run noise that comes with memorising 268 recordings, without needing
    any new data.
    """
    members = []
    dirs = [d.strip() for d in str(model_dir).split(",") if d.strip()]
    cks = [c for d in dirs for c in sorted(glob(os.path.join(d, "*", "detector.pt")))]
    for ck in cks:
        d = Path(ck).parent
        pos_f, neg_f = d / "test_scores_positive.csv", d / "test_scores_negative.csv"
        if not (pos_f.exists() and neg_f.exists()):
            print(f"  skipping {d.name}: no held-out scores to calibrate on")
            continue

        ckpt = torch.load(ck, map_location="cpu")
        model = Detector(ckpt["n_bins"], width=ckpt.get("width", 32),
                         tau=ckpt.get("tau", 1.0))
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        import pandas as pd
        pos = pd.read_csv(pos_f).iloc[:, 0].to_numpy()
        neg = pd.read_csv(neg_f).iloc[:, 0].to_numpy()
        a, b = fit_platt(pos, neg)
        tag = f"{d.parent.name}/{d.name}" if len(dirs) > 1 else d.name
        members.append({"name": d.name, "tag": tag, "model": model, "a": a, "b": b})
        print(f"  {tag}: calibrated on {len(pos)} positive / {len(neg)} "
              f"negative held-out recordings  (a={a:.2f}, b={b:.2f})")

    if not members:
        raise SystemExit(f"no usable detectors under {model_dir}")
    return members


# ---------------------------------------------------------------------------
# audio
# ---------------------------------------------------------------------------

def load_audio(path_or_bytes):
    if isinstance(path_or_bytes, (bytes, bytearray)):
        sr, x = wavfile.read(io.BytesIO(path_or_bytes))
    else:
        sr, x = wavfile.read(path_or_bytes)
    x = np.asarray(x)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    x = x.astype(np.float32)

    notes = []
    if sr != SR:
        # The models were trained at 16 kHz. Resampling keeps them applicable,
        # though a file that was never at 16 kHz may not have come from the same
        # extraction pipeline, which matters more than the rate itself.
        g = np.gcd(int(sr), SR)
        x = sps.resample_poly(x, SR // g, int(sr) // g).astype(np.float32)
        notes.append(f"resampled from {sr} Hz to {SR} Hz")

    dur = len(x) / SR
    if dur < DET_CROP_SEC:
        x = np.pad(x, (0, int(DET_CROP_SEC * SR) - len(x)))
        notes.append(f"padded from {dur:.1f}s to {DET_CROP_SEC:.0f}s")
    return x, dur, notes


@torch.no_grad()
def predict_one(wav, members, feat, device):
    """Average the calibrated probability across folds; also locate the peak."""
    probs, raw = [], []
    best_frames, best_p = None, -1.0

    for m in members:
        s = recording_score(m["model"], wav, feat, device)["lse"]
        p = 1.0 / (1.0 + np.exp(-(m["a"] * s + m["b"])))
        probs.append(float(p))
        raw.append(float(s))
        if p > best_p:
            best_p = p
            best_frames = m["model"]

    # Where the most confident member thinks it heard something.
    n = int(DET_CROP_SEC * SR)
    starts = np.arange(0.0, max(len(wav) / SR - DET_CROP_SEC, 0.0) + 1e-9,
                       EVAL_STRIDE)
    if not len(starts):
        starts = np.array([0.0])
    batch = []
    for st in starts:
        b = int(st * SR)
        seg = wav[b: b + n]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        batch.append(feat(torch.from_numpy(np.ascontiguousarray(seg))))
    x = torch.stack(batch).to(device)
    _, frame_logits = best_frames(x)
    fl = frame_logits.float().cpu().numpy()
    k = int(np.unravel_index(np.argmax(fl), fl.shape)[0])
    j = int(np.unravel_index(np.argmax(fl), fl.shape)[1])
    peak_time = float(starts[k] + (j + 0.5) / OUT_FPS)

    p = float(np.mean(probs))
    return {"probability": p,
            "spread": float(np.std(probs)),
            "member_probabilities": probs,
            "member_scores": raw,
            "peak_time_s": peak_time,
            "verdict": describe(p, float(np.std(probs)))}


def describe(p, spread):
    if spread > 0.25:
        tail = " but the models disagree with each other, so treat this as unresolved"
    else:
        tail = ""
    if p >= 0.80:
        return "likely present" + tail
    if p >= 0.60:
        return "leaning present" + tail
    if p >= 0.40:
        return "genuinely uncertain" + tail
    if p >= 0.20:
        return "leaning absent" + tail
    return "likely absent" + tail


# ---------------------------------------------------------------------------
# server
# ---------------------------------------------------------------------------

PAGE = """<!doctype html><meta charset="utf-8">
<title>help detector</title>
<style>
 body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      max-width:640px;margin:48px auto;padding:0 20px;color:#1a1a1a;line-height:1.6}
 h1{font-size:22px;font-weight:500;margin:0 0 4px}
 .sub{color:#666;font-size:14px;margin-bottom:28px}
 #drop{border:1.5px dashed #bbb;border-radius:12px;padding:44px 20px;text-align:center;
       color:#666;cursor:pointer;transition:.15s}
 #drop.over{border-color:#444;background:#fafafa;color:#222}
 #out{margin-top:28px;display:none}
 .big{font-size:46px;font-weight:500;letter-spacing:-1px}
 .verdict{font-size:17px;color:#333;margin-top:2px}
 .bar{height:8px;background:#eee;border-radius:4px;margin:18px 0 6px;overflow:hidden}
 .fill{height:100%;background:#444;width:0;transition:width .4s}
 .meta{font-size:13px;color:#777;margin-top:14px}
 .note{font-size:13px;color:#8a6d1f;background:#fdf6e3;border-radius:8px;
       padding:10px 12px;margin-top:14px}
 footer{margin-top:36px;font-size:12.5px;color:#888;border-top:1px solid #eee;padding-top:14px}
</style>
<h1>help detector</h1>
<div class="sub">optical receiver recordings, 16 kHz WAV</div>
<div id="drop">drop a .wav here, or click to choose</div>
<input id="file" type="file" accept=".wav,audio/wav" style="display:none">
<div id="out">
  <div class="big" id="pct">--</div>
  <div class="verdict" id="verdict"></div>
  <div class="bar"><div class="fill" id="fill"></div></div>
  <div class="meta" id="meta"></div>
  <div class="note" id="note" style="display:none"></div>
</div>
<footer>
Balanced accuracy across seven held-out speakers is about 78%, ranging from 93%
to 61% depending on the speaker. Negatives were synthesised by removing the
keyword from real recordings. Treat the number as a strong hint, not a verdict.
</footer>
<script>
const drop=document.getElementById('drop'), inp=document.getElementById('file');
drop.onclick=()=>inp.click();
inp.onchange=e=>{if(e.target.files[0])send(e.target.files[0])};
drop.ondragover=e=>{e.preventDefault();drop.classList.add('over')};
drop.ondragleave=()=>drop.classList.remove('over');
drop.ondrop=e=>{e.preventDefault();drop.classList.remove('over');
  if(e.dataTransfer.files[0])send(e.dataTransfer.files[0])};
async function send(f){
  drop.textContent='analysing '+f.name+' ...';
  try{
    const r=await fetch('/predict',{method:'POST',body:await f.arrayBuffer()});
    const d=await r.json();
    if(d.error){drop.textContent=d.error;return}
    const p=Math.round(d.probability*100);
    document.getElementById('pct').textContent=p+'%';
    document.getElementById('verdict').textContent=d.verdict;
    document.getElementById('fill').style.width=p+'%';
    document.getElementById('meta').textContent=
      'strongest moment at '+d.peak_time_s.toFixed(2)+' s   ·   '+
      d.member_probabilities.length+' models, spread '+d.spread.toFixed(2)+
      '   ·   '+d.duration_s.toFixed(1)+' s file';
    const n=document.getElementById('note');
    if(d.notes && d.notes.length){n.style.display='block';n.textContent=d.notes.join('; ')}
    else n.style.display='none';
    document.getElementById('out').style.display='block';
    drop.textContent='drop another .wav';
  }catch(err){drop.textContent='failed: '+err}
}
</script>"""


def serve(members, feat, device, port):
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            data = self.rfile.read(n)
            try:
                wav, dur, notes = load_audio(data)
                res = predict_one(wav, members, feat, device)
                res["duration_s"] = dur
                res["notes"] = notes
            except Exception as exc:
                res = {"error": f"could not read that file: {exc}"}
            body = json.dumps(res).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    print(f"\nopen http://localhost:{port}  (ctrl-c to stop)")
    HTTPServer(("127.0.0.1", port), H).serve_forever()


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", help="a .wav file or a folder of them")
    ap.add_argument("--models", default="outputs/detector")
    ap.add_argument("--csv", default=None, help="write results here")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if not args.input and not args.serve:
        ap.error("give a file or a folder, or use --serve")

    device = get_device(args.device)
    print(f"device: {device}\nloading detectors from {args.models}")
    members = load_ensemble(args.models, device)
    feat = Features()
    print(f"{len(members)} models in the ensemble")

    if args.serve:
        serve(members, feat, device, args.port)
        return

    paths = (sorted(glob(os.path.join(args.input, "*.wav")))
             if os.path.isdir(args.input) else [args.input])
    rows = []
    for p in paths:
        try:
            wav, dur, notes = load_audio(p)
            r = predict_one(wav, members, feat, device)
        except Exception as exc:
            print(f"{os.path.basename(p)}: could not read ({exc})")
            continue
        print(f"\n{os.path.basename(p)}   ({dur:.1f}s)")
        print(f"  probability that 'help' was spoken: {r['probability']:.1%}")
        print(f"  {r['verdict']}")
        print(f"  strongest moment at {r['peak_time_s']:.2f}s, "
              f"model spread {r['spread']:.2f}")
        for n in notes:
            print(f"  note: {n}")
        rows.append({"file": os.path.basename(p), "duration_s": round(dur, 2),
                     "probability": round(r["probability"], 4),
                     "spread": round(r["spread"], 4),
                     "peak_time_s": round(r["peak_time_s"], 2),
                     "verdict": r["verdict"]})

    if args.csv and rows:
        import pandas as pd
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
