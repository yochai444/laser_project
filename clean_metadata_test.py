#!/usr/bin/env python3
"""
clean_metadata.py
=================
Turns a Recordings_Metadata.json into the typed CSV the rest of the pipeline
reads. Works on any corpus, including held-out test sets recorded later.

WHAT IT DECIDES
    A recording has no keyword when StartTime or Duration is <= 0. Both the
    "-1" sentinel and the "0, 0" form appear in these files and both mean the
    same thing.

WHAT IT REFUSES TO GUESS
    An earlier version carried per-recording fixes keyed by id - a duration typo
    in 272, two recordings to exclude. Those are correct for one corpus and
    silently wrong for any other, because ids repeat across recording sessions.
    They are now command-line options that default to empty, and anything
    suspicious is FLAGGED rather than quietly repaired:

      - keyword intervals running past the end of the recording
      - implausibly short annotations
      - free-text noise descriptions the mapping has not seen

    Unknown noise text no longer aborts the run. It is reported, mapped to
    "unknown", and the recording is kept - a new corpus will always bring new
    wording, and losing the whole file over a spelling variant helps nobody.

USAGE
    python clean_metadata.py --json Recordings_Metadata.json --out metadata_clean.csv

    # corpus-specific corrections, when you have verified them by listening
    python clean_metadata.py --json test/Recordings_Metadata.json \
        --out test/labels.csv --fix-duration 272=0.9 --exclude 225,184
"""

import argparse
import json
import re
import sys

import numpy as np
import pandas as pd

# Free-text noise descriptions seen so far, with their spelling variants.
# has_background_speech records whether the reader was reading from the book
# during the take; "No Talking" means the keyword was spoken in isolation.
NOISE_MAP = {
    "":                                 ("clean",       "none",   True),
    "none":                             ("clean",       "none",   True),
    "nan":                              ("clean",       "none",   True),
    "noise":                            ("unspecified", "medium", True),
    "white noise":                      ("white_noise", "medium", True),
    "white noise with medium level":    ("white_noise", "medium", True),
    "white noise with medium levelne":  ("white_noise", "medium", True),
    "white noise with high level":      ("white_noise", "high",   True),
    "white noise with low level":       ("white_noise", "low",    True),
    "white noise only":                 ("white_noise", "medium", False),
    "white noise no word":              ("white_noise", "medium", True),
    "white noise with word no talking": ("white_noise", "medium", False),
    "music":                            ("music",       "medium", True),
    "music with high level":            ("music",       "high",   True),
    "music with medium level":          ("music",       "medium", True),
    "music with low level":             ("music",       "low",    True),
    "no talking":                       ("clean",       "none",   False),
    # New in the test set. "nothing" means nothing was said at all, so there is
    # no background reading either - the keyword-based fallback would have
    # guessed "unknown" and left has_background_speech True, which is wrong.
    "nothing":                          ("clean",       "none",   False),
    "no talking corapt end":            ("clean",       "none",   False),
    "no word":                          ("clean",       "none",   True),
    "accedental record":                ("unknown",     "none",   False),
    "accidental record":                ("unknown",     "none",   False),
}


def parse_noise(raw, unknown):
    key = re.sub(r"\s+", " ", str(raw or "").strip().lower())
    if key in NOISE_MAP:
        return NOISE_MAP[key]
    if key not in ("", "nan"):
        unknown.add(str(raw))
    # Guessing from keywords is better than dropping the recording, and the
    # unmapped text is reported so the map can be extended.
    if "music" in key:
        lvl = "high" if "high" in key else "low" if "low" in key else "medium"
        return ("music", lvl, "no talking" not in key)
    if "noise" in key:
        lvl = "high" if "high" in key else "low" if "low" in key else "medium"
        return ("white_noise", lvl, "no talking" not in key and "only" not in key)
    return ("unknown", "none", True)


def parse_pairs(spec):
    """'272=0.9,300=1.2' -> {'272': 0.9, '300': 1.2}"""
    out = {}
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            sys.exit(f"--fix-duration expects id=value, got {part!r}")
        k, v = part.split("=", 1)
        out[k.strip()] = float(v)
    return out


def build(json_path, rec_seconds, duration_fixes, exclude_ids, short_threshold):
    with open(json_path, encoding="utf-8-sig") as fh:
        raw = json.load(fh)

    unknown_noise = set()
    rows = []
    for rec_id, meta in raw.items():
        try:
            start = float(meta["StartTime"])
            dur = float(duration_fixes.get(rec_id, meta["Duration"]))
        except (KeyError, TypeError, ValueError):
            sys.exit(f"recording {rec_id}: StartTime/Duration missing or not a "
                     f"number -> {meta}")

        has_kw = start > 0 and dur > 0
        end = start + dur if has_kw else float("nan")
        noise_type, noise_level, has_bg = parse_noise(meta.get("Noise"), unknown_noise)

        rows.append({
            "recording_id": str(rec_id),
            "speaker_id": str(meta.get("RecorderID", "")),
            "has_keyword": has_kw,
            "start_s": start if has_kw else float("nan"),
            "end_s": end,
            "duration_s": dur if has_kw else float("nan"),
            "noise_type": noise_type,
            "noise_level": noise_level,
            "has_background_speech": has_bg,
            "is_clean": noise_type == "clean",
            "exclude": str(rec_id) in exclude_ids,
            "exclude_reason": "excluded on the command line"
                              if str(rec_id) in exclude_ids else "",
            "raw_noise_text": meta.get("Noise", ""),
        })

    df = pd.DataFrame(rows).sort_values("recording_id").reset_index(drop=True)
    df["past_end_warning"] = df.has_keyword & (df.end_s > rec_seconds)
    df["short_label_warning"] = df.has_keyword & (df.duration_s <= short_threshold)
    return df, unknown_noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="metadata_clean.csv")
    ap.add_argument("--rec-seconds", type=float, default=20.0,
                    help="nominal recording length, used only to flag intervals "
                         "that run past the end")
    ap.add_argument("--fix-duration", default="",
                    help="corrections you have verified, e.g. 272=0.9,300=1.2")
    ap.add_argument("--exclude", default="",
                    help="comma separated ids to drop, e.g. 225,184")
    ap.add_argument("--short-threshold", type=float, default=0.15,
                    help="flag annotations at or below this many seconds")
    args = ap.parse_args()

    exclude_ids = {s.strip() for s in args.exclude.split(",") if s.strip()}
    df, unknown = build(args.json, args.rec_seconds,
                        parse_pairs(args.fix_duration), exclude_ids,
                        args.short_threshold)
    df.to_csv(args.out, index=False)

    n_kw = int(df.has_keyword.sum())
    print(f"wrote {args.out}")
    print(f"  recordings      : {len(df)}")
    print(f"  with keyword    : {n_kw}")
    print(f"  without keyword : {len(df) - n_kw}")
    print(f"  excluded        : {int(df.exclude.sum())}")

    if unknown:
        print(f"\n  {len(unknown)} noise description(s) not in NOISE_MAP, mapped "
              f"by keyword instead:")
        for u in sorted(unknown):
            print(f"    {u!r}")
        print("  add them to NOISE_MAP if the guess is wrong.")

    past = df[df.past_end_warning]
    if len(past):
        print(f"\n  {len(past)} keyword interval(s) run past {args.rec_seconds:.0f}s. "
              f"NOT corrected - verify and pass --fix-duration:")
        for r in past.itertuples():
            print(f"    {r.recording_id}: start {r.start_s:.2f} + duration "
                  f"{r.duration_s:.2f} = {r.end_s:.2f}s")

    short = df[df.short_label_warning]
    if len(short):
        print(f"\n  {len(short)} annotation(s) at or below "
              f"{args.short_threshold}s - fast for a whole word, worth a listen:")
        print(f"    {', '.join(short.recording_id.tolist()[:20])}")

    if n_kw and (len(df) - n_kw) == 0:
        print("\n  No keyword-free recordings in this file. Specificity cannot "
              "be measured from it.")

    kw = df[df.has_keyword]
    if len(kw):
        print()
        print(kw.groupby("speaker_id").agg(
            n=("recording_id", "size"),
            mean_start=("start_s", "mean"),
            mean_dur=("duration_s", "mean"),
        ).round(2).to_string())
        print(f"\n  keyword position: {kw.start_s.min():.1f}-{kw.start_s.max():.1f}s, "
              f"std {kw.start_s.std():.2f}s")
        if kw.start_s.std() < 1.5:
            print("  The keyword falls in a narrow time window, so a constant-time")
            print("  guess will score well. Localization results from this set must")
            print("  be read against that baseline.")


if __name__ == "__main__":
    main()
