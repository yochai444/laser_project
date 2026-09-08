"""
clean_metadata.py
-----------------
Normalizes Recordings_Metadata.json into a typed CSV that the rest of the
pipeline consumes.

Applies the following fixes (all decided from the label audit):

  * 6 recordings with StartTime = Duration = -1  -> has_keyword = False
  * 8 recordings with StartTime = Duration = 0   -> has_keyword = False
      (confirmed by the annotator: the word was not spoken in these)
  * recording 272: Duration 10.9 -> 0.9 (typo; 10.4 + 10.9 exceeds the
    20 s recording length)
  * free-text "Noise" field -> structured (noise_type, noise_level,
    has_background_speech)
  * recordings flagged as corrupt / accidental -> exclude = True

Usage:
    python clean_metadata.py --json Recordings_Metadata.json --out metadata_clean.csv
"""

import argparse
import json
import re

import pandas as pd

RECORDING_LENGTH_S = 20.0

# ---------------------------------------------------------------------------
# Explicit per-recording overrides. Anything not listed here is derived by rule.
# ---------------------------------------------------------------------------

# Duration typos: {recording_id: corrected_duration}
DURATION_FIXES = {
    "272": 0.9,  # was 10.9 -> would end at 21.3 s in a 20 s file
}

# Recordings to drop entirely from training and evaluation.
EXCLUDE = {
    "225": "annotator marked this an accidental recording",
    "184": "annotator marked the end of the file corrupt",
}

# ---------------------------------------------------------------------------
# Noise free-text -> structured fields
# ---------------------------------------------------------------------------
# has_background_speech = the reader was reading from the book during the take.
# "No Talking" means the keyword was spoken but there was no book reading, so
# those takes have a much cleaner keyword than the rest of the corpus.

NOISE_MAP = {
    "":                                  ("clean",       "none",   True),
    "none":                              ("clean",       "none",   True),
    "noise":                             ("unspecified", "medium", True),
    "white noise":                       ("white_noise", "medium", True),
    "white noise with medium level":     ("white_noise", "medium", True),
    "white noise with medium levelne":   ("white_noise", "medium", True),  # typo
    "white noise with high level":       ("white_noise", "high",   True),
    "white noise only":                  ("white_noise", "medium", False),
    "white noise no word":               ("white_noise", "medium", True),
    "white noise with word no talking":  ("white_noise", "medium", False),
    "music":                             ("music",       "medium", True),
    "music with high level":             ("music",       "high",   True),
    "no talking":                        ("clean",       "none",   False),
    "no talking corapt end":             ("clean",       "none",   False),  # typo
    "no word":                           ("clean",       "none",   True),
    "accedental record":                 ("unknown",     "none",   False),  # typo
}


def parse_noise(raw):
    key = re.sub(r"\s+", " ", (raw or "").strip().lower())
    if key not in NOISE_MAP:
        raise KeyError(
            f"unmapped Noise value {raw!r} -- add it to NOISE_MAP before continuing"
        )
    return NOISE_MAP[key]


def build(json_path):
    with open(json_path, encoding="utf-8-sig") as fh:
        raw = json.load(fh)

    rows = []
    for rec_id, meta in raw.items():
        start = float(meta["StartTime"])
        dur = float(DURATION_FIXES.get(rec_id, meta["Duration"]))

        # -1 sentinel and 0/0 both mean the keyword was never uttered.
        has_kw = not (start <= 0 or dur <= 0)

        noise_type, noise_level, has_bg_speech = parse_noise(meta.get("Noise"))

        end = start + dur if has_kw else float("nan")

        rows.append(
            {
                "recording_id": rec_id,
                "speaker_id": meta["RecorderID"],
                "has_keyword": has_kw,
                "start_s": start if has_kw else float("nan"),
                "end_s": end,
                "duration_s": dur if has_kw else float("nan"),
                "noise_type": noise_type,
                "noise_level": noise_level,
                "has_background_speech": has_bg_speech,
                "is_clean": noise_type == "clean",
                "exclude": rec_id in EXCLUDE,
                "exclude_reason": EXCLUDE.get(rec_id, ""),
                "raw_noise_text": meta.get("Noise", ""),
            }
        )

    df = pd.DataFrame(rows).sort_values("recording_id").reset_index(drop=True)

    # --- sanity checks -----------------------------------------------------
    kw = df[df.has_keyword]
    bad_end = kw[kw.end_s > RECORDING_LENGTH_S]
    if len(bad_end):
        raise ValueError(
            "keyword interval runs past the end of the recording for: "
            f"{bad_end.recording_id.tolist()}"
        )
    if kw.start_s.min() < 0:
        raise ValueError("negative start time survived cleaning")

    # Flag implausibly short annotations for a manual second look. 0.1 s is
    # very fast for a full /h-e-l-p/ and is more likely a sloppy annotation.
    df["short_label_warning"] = df.has_keyword & (df.duration_s <= 0.15)

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="Recordings_Metadata.json")
    ap.add_argument("--out", default="metadata_clean.csv")
    args = ap.parse_args()

    df = build(args.json)
    df.to_csv(args.out, index=False)

    n_kw = int(df.has_keyword.sum())
    print(f"wrote {args.out}")
    print(f"  recordings          : {len(df)}")
    print(f"  with keyword        : {n_kw}")
    print(f"  without keyword     : {len(df) - n_kw}")
    print(f"  excluded            : {int(df.exclude.sum())}")
    print(f"  short-label warnings: {int(df.short_label_warning.sum())}")
    print()
    print(df.groupby("speaker_id").agg(
        n=("recording_id", "size"),
        with_kw=("has_keyword", "sum"),
        clean=("is_clean", "sum"),
        mean_start=("start_s", "mean"),
        mean_dur=("duration_s", "mean"),
    ).round(2))


if __name__ == "__main__":
    main()
