import pandas as pd
from pathlib import Path

base_path = 'IPqM-Fall/raw'

# Iterate through all ID folders
for id_folder in sorted(Path(base_path).glob('ID*')):
    for positioning in ['CHEST', 'LEFT', 'RIGHT']:
        sampling_file = id_folder / positioning / f"{id_folder.name}_{positioning}_sampling.csv"
        
        if sampling_file.exists():
            df = pd.read_csv(sampling_file)
            
            # Create trial column: number repetitions of same exercise/userId/positioning/withRifle
            df['trial'] = df.groupby(['exercise', 'userId', 'positioning', 'withRifle']).cumcount() + 1
            
            # Reorder columns to put trial after exercise
            cols = ['exercise', 'trial', 'userId', 'positioning', 'withRifle', 'beginning', 'ending']
            df = df[cols]
            
            df.to_csv(sampling_file, index=False)
            print(f"Processed: {sampling_file}")