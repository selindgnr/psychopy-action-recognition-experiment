#!/usr/bin/env python3
"""
Generate a balanced 75-item evaluation list in 5 blocks of 15 videos each (0–14, 15–29, ...).
Each block contains exactly 3 videos from each of 5 labels (5×3=15), sampled without replacement
and shuffled inside each block to avoid label clustering.

Input file:
  ./videos.xlsx
  Expected columns (current file):
    - class_label  (will be renamed to label)
    - video_path   (will be renamed to video_file)
    - video_id     (ignored)
    - any other columns (ignored)

Output file (ONLY these columns, in this order):
  ./sorted_test_list.csv
  Columns: video_file,label,correct_ans
"""

from pathlib import Path
import sys
import random
import os
import pandas as pd

# ========= INPUT / OUTPUT PATHS =========
# Portable path handling:
# - Optional CLI args: python script.py <input_excel> [output_csv]
# - Otherwise, try (in order): script folder ./videos.xlsx, current working directory ./videos.xlsx,
#   then a common Desktop fallback.
_input_arg = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else None
_output_arg = Path(sys.argv[2]).expanduser() if len(sys.argv) > 2 else None

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CANDIDATES = [
    _SCRIPT_DIR / "videos.xlsx",
    Path.cwd() / "videos.xlsx",
    Path.home() / "Desktop" / "videos.xlsx",
]

if _input_arg is not None:
    INPUT_PATH = _input_arg
else:
    for _cand in _DEFAULT_CANDIDATES:
        if _cand.exists():
            INPUT_PATH = _cand
            break
    else:
        # If nothing exists yet, default to script folder so a future videos.xlsx here will work
        INPUT_PATH = _DEFAULT_CANDIDATES[0]

OUTPUT_CSV = _output_arg if _output_arg is not None else (INPUT_PATH.parent / "sorted_test_list.csv")

# ========= COLUMN MAPPING / OUTPUT SCHEMA =========
SRC_LABEL_COL = "class_label"   # in your file
SRC_FILE_COL  = "video_path"    # in your file

DST_LABEL_COL = "label"         # in output
DST_FILE_COL  = "video_file"    # in output
DST_CORRECT   = "correct_ans"   # in output

OUTPUT_COLUMNS = [DST_FILE_COL, DST_LABEL_COL, DST_CORRECT]

# ========= ORDERING PARAMETERS =========
BLOCKS = 5
PER_LABEL_PER_BLOCK = 3
PER_LABEL = BLOCKS * PER_LABEL_PER_BLOCK  # total per label = 15
EXPECTED_NUM_LABELS = 5                   # warn if different

# Deterministic seeds (reproducible results)
SEED_SAMPLE = 123              # pick 15 per label (if more are available)
SEED_PER_LABEL_SHUFFLE = 999   # shuffle within each label pool
SEED_BLOCK_BASE = 1000         # shuffle inside each block with seed_block_base + block_idx

# ========= LABEL → CORRECT ANSWER MAPPING =========
LABEL_TO_ID = {
    "JumpingJack": 1,
    "Lunges": 2,
    "PullUps": 3,
    "PushUps": 4,
    "Swing": 5,
}


def load_table(path: Path) -> pd.DataFrame:
    """Load Excel or CSV/TSV file by extension."""
    ext = path.suffix.lower()
    try:
        if ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif ext == ".csv":
            return pd.read_csv(path)
        elif ext == ".tsv":
            return pd.read_csv(path, sep="\t")
        else:
            # Fallback: try Excel
            return pd.read_excel(path)
    except Exception as e:
        raise RuntimeError(f"Failed to read '{path}': {e}")


def prepare_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to the desired output schema and keep only necessary ones for processing."""
    missing = [c for c in (SRC_LABEL_COL, SRC_FILE_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required input column(s): {missing}. Found: {list(df.columns)}")

    # Rename to destination names
    df = df.rename(columns={
        SRC_LABEL_COL: DST_LABEL_COL,
        SRC_FILE_COL: DST_FILE_COL,
    }).copy()

    return df


def relativize_paths(df: pd.DataFrame, base_dir: Path) -> pd.DataFrame:
    """Convert absolute paths in video_file to paths relative to base_dir; leave relative paths unchanged."""
    def _to_rel(p):
        p_str = "" if pd.isna(p) else str(p)
        if not p_str:
            return p_str
        try:
            p_path = Path(p_str)
        except Exception:
            return p_str
        if p_path.is_absolute():
            try:
                return os.path.relpath(str(p_path), start=str(base_dir))
            except Exception:
                return p_str
        return p_str

    df[DST_FILE_COL] = df[DST_FILE_COL].map(_to_rel)
    return df


def make_balanced_order(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame reordered into 5 balanced blocks of 15 videos."""
    # Validate required columns now that they are renamed
    missing = [c for c in (DST_LABEL_COL, DST_FILE_COL) if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns after renaming: {missing}")

    labels = sorted(df[DST_LABEL_COL].dropna().unique().tolist())
    if len(labels) != EXPECTED_NUM_LABELS:
        print(f"[WARN] Detected {len(labels)} unique labels (expected {EXPECTED_NUM_LABELS}): {labels}", file=sys.stderr)

    # Ensure enough items per label
    counts = df[DST_LABEL_COL].value_counts()
    too_few = counts[counts < PER_LABEL]
    if not too_few.empty:
        raise ValueError(f"Some labels have fewer than {PER_LABEL} samples: {too_few.to_dict()}")

    # Downsample to exactly PER_LABEL per label (deterministic)
    sampled = (
        df.groupby(DST_LABEL_COL, group_keys=False)
          .apply(lambda g: g.sample(n=PER_LABEL, random_state=SEED_SAMPLE, replace=False))
          .reset_index(drop=True)
    )

    # Prepare per-label pools (deterministic shuffle)
    per_label_pool = {
        lbl: sampled[sampled[DST_LABEL_COL] == lbl]
             .sample(frac=1, random_state=SEED_PER_LABEL_SHUFFLE)
             .index.tolist()
        for lbl in labels
    }

    # Build 5 blocks: pick 3 from each label per block (without replacement)
    blocks = []
    for b in range(BLOCKS):
        block_idx = []
        for lbl in labels:
            chosen = [per_label_pool[lbl].pop() for _ in range(PER_LABEL_PER_BLOCK)]
            block_idx.extend(chosen)
        random.Random(SEED_BLOCK_BASE + b).shuffle(block_idx)
        blocks.append(block_idx)

    # Flatten blocks into final order
    ordered_indices = [i for block in blocks for i in block]
    out = sampled.loc[ordered_indices].reset_index(drop=True)
    return out


def attach_correct_answer(df: pd.DataFrame) -> pd.DataFrame:
    """Add the correct_ans column using LABEL_TO_ID mapping and validate labels."""
    # Map labels to numeric IDs
    df[DST_CORRECT] = df[DST_LABEL_COL].map(LABEL_TO_ID)

    # Ensure no unknown labels slipped in
    if df[DST_CORRECT].isna().any():
        unknown = df.loc[df[DST_CORRECT].isna(), DST_LABEL_COL].unique().tolist()
        raise ValueError(
            f"Found label(s) without a mapping for correct_ans: {unknown}. "
            f"Expected one of: {list(LABEL_TO_ID.keys())}"
        )
    df[DST_CORRECT] = df[DST_CORRECT].astype(int)
    return df


def print_validation(ordered: pd.DataFrame) -> None:
    """Print duplicate check and per-block label counts (informational)."""
    # Duplicate videos by video_file?
    dup_mask = ordered.duplicated(subset=[DST_FILE_COL], keep=False)
    if dup_mask.any():
        print("[WARN] Duplicate videos found:")
        print(ordered.loc[dup_mask, DST_FILE_COL].unique())
    else:
        print("[Check] No duplicate videos.")

    # Show per-block counts to verify 3 per label per block
    print("[Check] Per-sub-block label counts (expect 3 per label per block):")
    labels = sorted(ordered[DST_LABEL_COL].unique().tolist())
    block_size = PER_LABEL_PER_BLOCK * len(labels)  # should be 15
    ordered["_sub_block"] = ordered.index // block_size
    counts = (
        ordered.groupby(["_sub_block", DST_LABEL_COL])
               .size()
               .unstack(fill_value=0)
               .sort_index()
    )
    print(counts.to_string())
    ordered.drop(columns=["_sub_block"], inplace=True)


def main():
    # Load and prepare columns
    df = load_table(INPUT_PATH)
    df = prepare_columns(df)
    df = relativize_paths(df, INPUT_PATH.parent)  # <-- convert absolute paths to relative

    # Create balanced ordering
    ordered = make_balanced_order(df)

    # Add correct_ans from mapping
    ordered = attach_correct_answer(ordered)

    # Keep ONLY requested output columns in the exact order
    out_df = ordered[[DST_FILE_COL, DST_LABEL_COL, DST_CORRECT]].copy()

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)  # <-- auto-create output folder
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Wrote {len(out_df)} rows to: {OUTPUT_CSV}")

    # Optional: quick validation (prints only; does not change the saved file)
    print_validation(out_df.copy())


if __name__ == "__main__":
    main()
