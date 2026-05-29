from pathlib import Path

root = Path("IPqM-Fall/raw")

# ---------- STEP 1: SAFE TEMP RENAMES ----------
temp_map = {
    "ADL11": "TEMP_ADL9",
    "ADL12": "TEMP_ADL10",
    "ADL13": "TEMP_ADL11",
    "ADL14": "TEMP_ADL12",
    "ADL15": "TEMP_ADL13",
    "FALL5": "TEMP_FALL4",
    "FALL6": "TEMP_FALL5",

    "ADL11R": "TEMP_ADL9R",
    "ADL12R": "TEMP_ADL10R",
    "ADL13R": "TEMP_ADL11R",
    "ADL14R": "TEMP_ADL12R",
    "ADL15R": "TEMP_ADL13R",
    "FALL5R": "TEMP_FALL4R",
    "FALL6R": "TEMP_FALL5R",
}

for file in root.rglob("*"):
    if file.is_file():

        new_name = file.name

        for old, temp in temp_map.items():
            if old in new_name:
                new_name = new_name.replace(old, temp)

        if new_name != file.name:
            new_path = file.with_name(new_name)

            print(f"TEMP: {file.name} -> {new_name}")
            file.rename(new_path)

# ---------- STEP 2: FINAL NAMES ----------
final_map = {
    "TEMP_ADL9": "ADL9",
    "TEMP_ADL10": "ADL10",
    "TEMP_ADL11": "ADL11",
    "TEMP_ADL12": "ADL12",
    "TEMP_ADL13": "ADL13",
    "TEMP_FALL4": "FALL4",
    "TEMP_FALL5": "FALL5",

    "TEMP_ADL9R": "ADL9R",
    "TEMP_ADL10R": "ADL10R",
    "TEMP_ADL11R": "ADL11R",
    "TEMP_ADL12R": "ADL12R",
    "TEMP_ADL13R": "ADL13R",
    "TEMP_FALL4R": "FALL4R",
    "TEMP_FALL5R": "FALL5R",
}

for file in root.rglob("*"):
    if file.is_file():

        new_name = file.name

        for temp, final in final_map.items():
            if temp in new_name:
                new_name = new_name.replace(temp, final)

        if new_name != file.name:
            new_path = file.with_name(new_name)

            print(f"FINAL: {file.name} -> {new_name}")
            file.rename(new_path)

print("Done.")