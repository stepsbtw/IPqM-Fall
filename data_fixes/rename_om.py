from pathlib import Path

root = Path("IPqM-Fall")

for path in root.rglob("*OM*"):
    new_name = path.name.replace("OM", "MO")
    new_path = path.with_name(new_name)

    print(f"Renaming:\n  {path}\n  -> {new_path}")
    path.rename(new_path)