#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import os
import sys

# Video extensions to include
VIDEO_EXTS = {
    ".mp4", ".mov", ".avi", ".mkv", ".wmv",
    ".m4v", ".webm", ".mpg", ".mpeg"
}

def _rel_for_output(file_path: Path, output_excel: Path) -> str:
    """Return file_path relative to the Excel file's folder; fall back to absolute on error."""
    base_dir = output_excel.parent
    try:
        return os.path.relpath(str(file_path.resolve()), start=str(base_dir))
    except Exception:
        return str(file_path.resolve())

def _classlike_count(candidate: Path) -> int:
    """
    Heuristic: how many immediate subfolders of 'candidate' contain at least one video
    (at any depth). Higher means this is likely the dataset root (class folders).
    """
    count = 0
    for sd in (d for d in candidate.iterdir() if d.is_dir()):
        try:
            if any(f.is_file() and f.suffix.lower() in VIDEO_EXTS for f in sd.rglob("*")):
                count += 1
        except PermissionError:
            pass
    return count

def _discover_root(base: Path) -> Path:
    """
    Choose the best root under 'base' that looks like the parent of class folders.
    Tries: base itself and each immediate subfolder; picks the one with the most
    subfolders containing videos. Falls back to 'base' if none found but videos exist.
    """
    candidates = [base] + [d for d in base.iterdir() if d.is_dir()]
    best = None
    best_count = -1
    for cand in candidates:
        try:
            cnt = _classlike_count(cand)
        except PermissionError:
            cnt = -1
        if cnt > best_count:
            best, best_count = cand, cnt

    # Fallback: if no classlike subfolders but there are videos somewhere under base, use base
    if best_count <= 0:
        if any(f.is_file() and f.suffix.lower() in VIDEO_EXTS for f in base.rglob("*")):
            return base
        raise SystemExit("No video class folders found under the script folder.")

    return best

def collect_videos(root: Path, output_excel: Path) -> pd.DataFrame:
    """
    Recursively collect videos under 'root'.
    Class label = first folder under 'root'. Files directly in 'root' are skipped.
    Paths are stored relative to the Excel file's folder for portability.
    """
    rows = []
    for f in root.rglob("*"):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            rel = f.relative_to(root)
            parts = rel.parts
            if len(parts) < 2:
                # file is directly under root, skip (no class folder)
                continue
            class_label = parts[0]
            video_id = f.name
            video_path = _rel_for_output(f, output_excel)
            rows.append(
                {"class_label": class_label, "video_id": video_id, "video_path": video_path}
            )

    df = pd.DataFrame(rows).sort_values(["class_label", "video_id"]).reset_index(drop=True)
    return df

def main():
    # Work entirely from the folder where this script lives
    script_dir = Path(__file__).resolve().parent

    # Auto-detect dataset root (folder whose subfolders look like class labels)
    root = _discover_root(script_dir)

    # Output Excel next to the script
    output_path = script_dir / "videos.xlsx"

    df = collect_videos(root, output_excel=output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_path, index=False)

    print(f"[OK] Detected dataset root: {root}")
    print(f"[OK] Wrote {len(df)} rows to {output_path}")

if __name__ == "__main__":
    main()
