"""Quickly inspect annotation file format and image folder."""

from pathlib import Path
from collections import Counter

for name in ["train_line_annotation.txt", "test_line_annotation.txt"]:
    p = Path("data/data_line") / name
    print("=" * 60)
    print(f"{name}   size = {p.stat().st_size} bytes")
    with open(p, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"  total lines: {len(lines)}")
    print("  --- first 3 lines (repr) ---")
    for i, line in enumerate(lines[:3]):
        print(f"  [{i}] {line.rstrip()!r}")
    print("  --- middle 3 lines (repr) ---")
    mid = len(lines) // 2
    for i, line in enumerate(lines[mid : mid + 3]):
        print(f"  [{mid + i}] {line.rstrip()!r}")
    print("  --- last 3 lines (repr) ---")
    for i, line in enumerate(lines[-3:]):
        print(f"  [{len(lines)-3+i}] {line.rstrip()!r}")

    # Count separator types per line
    sep_counts = Counter()
    for line in lines:
        line = line.rstrip()
        if "\t" in line:
            sep_counts["tab"] += 1
        elif "  " in line:  # 2+ spaces
            sep_counts["multi-space"] += 1
        elif " " in line:
            sep_counts["single-space"] += 1
        elif not line:
            sep_counts["blank"] += 1
        else:
            sep_counts["other"] += 1
    print("  separator counts:", dict(sep_counts))

    # Extension of referenced files
    ext_counts = Counter()
    for line in lines:
        line = line.rstrip()
        first_token = line.split(maxsplit=1)[0] if line else ""
        if first_token:
            ext_counts[Path(first_token).suffix.lower()] += 1
    print("  referenced extensions:", dict(ext_counts))
    print()

img_dir = Path("data/data_line/InkData_line_processed")
all_files = list(img_dir.iterdir())
print(f"Total files in {img_dir}: {len(all_files)}")
ext_counts = Counter(p.suffix.lower() for p in all_files)
print("Extensions:", dict(ext_counts))
