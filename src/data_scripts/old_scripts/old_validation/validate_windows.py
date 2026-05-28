import json
import numpy as np

WINDOW_SIZE = 180

with open("labelstudio_windows.json", "r") as f:
    tasks = json.load(f)

print("Total windows:", len(tasks))

bad = 0

for i, task in enumerate(tasks):

    values = task["data"]["values"]

    lengths = []

    for k, v in values.items():

        arr = np.asarray(v)

        lengths.append(len(arr))

        # ----------------------------------------
        # Check exact window size
        # ----------------------------------------

        if len(arr) != WINDOW_SIZE:

            print("\nBAD SIZE")
            print(i, k, len(arr))

            bad += 1

        # ----------------------------------------
        # Check NaNs / infs
        # ----------------------------------------

        if not np.isfinite(arr).all():

            print("\nNON-FINITE VALUES")
            print(i, k)

            bad += 1

    # --------------------------------------------
    # Check all channels same length
    # --------------------------------------------

    if len(set(lengths)) != 1:

        print("\nMISMATCHED CHANNEL LENGTHS")
        print(i, lengths)

        bad += 1

print("\nValidation done.")
print("Problems found:", bad)