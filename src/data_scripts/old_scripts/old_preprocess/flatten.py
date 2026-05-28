from pathlib import Path
import shutil

BASE = Path("IPqM-Fall/raw")
OUT = Path("IPqM-Fall/data")
OUT.mkdir(parents=True, exist_ok=True)

for csv_path in BASE.rglob("*.csv"):
    # skip if already in the output folder
    if OUT in csv_path.parents:
        continue
    target = OUT / csv_path.name
    # if target exists, add a numeric suffix to avoid overwrite
    if target.exists():
        stem = target.stem
        suf = 1
        while True:
            candidate = OUT / f"{stem}_{suf}{target.suffix}"
            if not candidate.exists():
                target = candidate
                break
            suf += 1
    shutil.move(str(csv_path), str(target))
    print("Moved:", csv_path, "->", target)