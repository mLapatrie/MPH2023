import os
import pandas as pd
import numpy as np

for sub in range(1, 7):
    path = f"Dataset/sub-0{sub}/func/"
    runs = os.listdir(path)
    
    for run in runs:
        if "events.tsv" in run:
            run_num = run[-13:-11]
            
            stimuli_df = pd.read_csv(os.path.join(path, run))
            
            sound_indices = []
            
            for index, stimulus in stimuli_df.iterrows():
                sound_name = str(stimulus).split(" ")[4].split("\\")[2].split('\n')[0][1:]
                sound_indices.append(sound_name)
            
            np.save(f"parsed_output_data/sub0{sub}_run{run_num}_stims.npy", sound_indices)