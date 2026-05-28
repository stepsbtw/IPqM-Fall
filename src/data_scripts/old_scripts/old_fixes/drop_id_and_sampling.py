import pandas as pd
from pathlib import Path

base_path = 'IPqM-Fall/raw'

# Iterate through all ID folders
for id_folder in sorted(Path(base_path).glob('ID*')):
    for positioning in ['CHEST', 'LEFT', 'RIGHT']:
        # Drop 'id' column from sampling files
        sampling_file = id_folder / positioning / f"{id_folder.name}_{positioning}_sampling.csv"
        if sampling_file.exists():
            df = pd.read_csv(sampling_file)
            df = df.drop('id', axis=1)
            df.to_csv(sampling_file, index=False)
            print(f"Processed: {sampling_file}")
        
        # Drop 'sampling' column from acceleration files
        accel_file = id_folder / positioning / f"{id_folder.name}_{positioning}_acceleration.csv"
        if accel_file.exists():
            df = pd.read_csv(accel_file)
            df = df.drop('sampling', axis=1)
            df.to_csv(accel_file, index=False)
            print(f"Processed: {accel_file}")
        
        # Drop 'sampling' column from angular_speed files
        angular_file = id_folder / positioning / f"{id_folder.name}_{positioning}_angular_speed.csv"
        if angular_file.exists():
            df = pd.read_csv(angular_file)
            df = df.drop('sampling', axis=1)
            df.to_csv(angular_file, index=False)
            print(f"Processed: {angular_file}")