
#!/usr/bin/env python3
"""
concise_csv_extractor_v4.py

Robust extractor for PsychoPy wide-text spreadsheets.
Fixes the alias‑matching bug (previous version accidentally deleted all "s" characters).

Output: a 4-column CSV (Video_ID, Correct_Answer, Participant_Answer, Response_RT).

Usage:
    python concise_csv_extractor_v4.py wide.csv [slim.csv]
"""

import sys
import re
from pathlib import Path
import pandas as pd

CANON = ["Video_ID", "Correct_Answer", "Participant_Answer", "Response_RT"]

ALIASES = {
    "Video_ID": [
        "videoid", "video_id", "video", "videofile", "video_file",
        "movie", "stimulus"
    ],
    "Correct_Answer": [
        "correctans", "correct_answer", "corrans", "correct", "answer.correct",
        "correctlabel"
    ],
    "Participant_Answer": [
        "participant_answer", "participantans", "participant_ans",
        "response", "answer", "key_resp", "participantresponse", "resp"
    ],
    "Response_RT": [
        "response_rt", "responsert", "rt", "reactiontime", "reaction_time",
        "reaction", "key_rt", "responsetime"
    ]
}

def norm(col: str) -> str:
    "Lowercase & strip whitespace/underscores/dots."
    return re.sub(r"[\\s_.]", "", col.lower())

def find_match(df_cols, canon):
    norm_map = {norm(c): c for c in df_cols}
    for alias in ALIASES[canon]:
        alias_norm = norm(alias)
        if alias_norm in norm_map:
            return norm_map[alias_norm]
    return None

def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python concise_csv_extractor_v4.py wide.csv [slim.csv]")
    in_csv = Path(sys.argv[1])
    out_csv = Path(sys.argv[2]) if len(sys.argv) > 2 else in_csv.with_name(in_csv.stem + "_concise.csv")

    df = pd.read_csv(in_csv)
    mapping = {}
    for canon in CANON:
        colname = find_match(df.columns, canon)
        if colname is None:
            available = ", ".join(list(df.columns)[:25]) + (" ..." if len(df.columns) > 25 else "")
            raise SystemExit(f"[ERROR] Could not find a column matching '{canon}'.\\n"
                             f"Available columns: {available}")
        mapping[canon] = colname

    df_out = df[[mapping[c] for c in CANON]].copy()
    df_out.columns = CANON
    df_out.to_csv(out_csv, index=False)
    print(f"✓  Wrote {len(df_out)} rows → {out_csv}")
    for canon in CANON:
        print(f"   {canon:<18} ← {mapping[canon]}")

if __name__ == "__main__":
    main()
