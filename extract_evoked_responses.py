import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, datasets, plotting
from nilearn.input_data import NiftiLabelsMasker
from scipy.ndimage import label
import os


threshold = 0.5
common_labels = [1, 5, 6, 12, 13, 14, 15, 16, 17, 18, 21, 23, 24, 31, 32, 33, 34, 35, 37, 38, 39, 40, 48, 49, 50, 51, 53, 54, 62, 63, 64, 65, 67, 71, 73, 74, 76, 80, 81, 87, 88, 89, 90, 91, 92, 93, 96, 98, 99, 106, 107, 109, 110, 112, 113, 114, 115, 123, 124, 125, 126, 128, 129, 137, 138, 139, 140, 142, 146, 148, 149]

def compute_fmri_data(fmri_filename):
    # Load fMRI data
    #fmri_filename = "Dataset/sub-02/func/sub-02_task-auditory_run-01_bold.nii.gz"
    fmri_img = nib.load(fmri_filename)

    # Load Destrieux atlas
    atlas_destrieux = datasets.fetch_atlas_destrieux_2009(lateralized=True)
    destrieux_atlas_img = atlas_destrieux.maps
    destrieux_labels = atlas_destrieux.labels

    # Compute mean EPI
    mean_fmri_img = image.mean_img(fmri_filename)

    # Resample atlas to match the fMRI data
    resampled_atlas_img = image.resample_to_img(destrieux_atlas_img, mean_fmri_img, interpolation="nearest")

    # Parcellate the fMRI time series
    masker = NiftiLabelsMasker(labels_img=resampled_atlas_img, standardize=True, memory="nilearn_cache", 
                            verbose=5, resampling_target="data", interpolation="nearest")
    time_series = masker.fit_transform(fmri_img)
    filtered_time_series = filter_time_series(time_series, masker.labels_)
    
    # Load stimuli .tsv file
    stimuli_df = pd.read_csv(fmri_filename[:-11] + "events.tsv")
    
    # Iterate over stimui and extract the time series
    hemodynamic_delay = 0.5 # Assuming a delay for the hemodynamic response
    repetition_time = 2.8
    sampling_rate = 1 / repetition_time
    window = 10 # The number of timepoints for the average
    
    data_object = []
    
    sounds_base = os.listdir('Dataset/stimuli')
    for index, stimulus in stimuli_df.iterrows():
        onset_time = float(str(stimulus).split(" ")[4].split("\\")[0])
        onset_volume = int((onset_time + hemodynamic_delay) / repetition_time)
        end_volume = onset_volume + window

        sound_name = str(stimulus).split(" ")[4].split("\\")[2].split('\n')[0][1:]
        
        sound_index = sounds_base.index(sound_name)
        
        if end_volume > time_series.shape[0]:
            end_volume = time_series.shape[0]
            
        evoked_response = filtered_time_series[onset_volume:end_volume].mean(axis=0)
        
        print(index, sound_index)
        
        data_object.append(evoked_response)
        
    return data_object
    

def filter_time_series(time_series, labels):
    index_mask = np.array([i in common_labels for i in labels])
    
    filtered_time_series = time_series[:, index_mask]
    
    return filtered_time_series


def find_common_labels(list_of_label_arrays):
    
    common_labels = set(list_of_label_arrays[0])
    
    for labels in list_of_label_arrays[1:]:
        common_labels = common_labels.intersection(labels)
        
    return common_labels


big_data = []
for sub in range(1, 2):
    path = f"Dataset/sub-0{sub}/func/"
    runs = os.listdir(path)
    
    do = True
    for run in runs:
        if "run-01_bold.nii.gz" in run or "run-02_bold.nii.gz" in run:
            evoked_responses = compute_fmri_data(os.path.join(path, run))
            for i in evoked_responses:
                big_data.append(i)
    
    
df = pd.DataFrame(big_data)
df.to_csv('big_data.csv', index=False)