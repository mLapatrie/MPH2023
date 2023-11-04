import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, datasets
from nilearn.input_data import NiftiLabelsMasker
from scipy.ndimage import label

# Load fMRI data
fmri_filename = "Dataset/sub-02/func/sub-02_task-auditory_run-01_bold.nii.gz"
fmri_img = nib.load(fmri_filename)

# Load Destrieux atlas
atlas_destrieux = datasets.fetch_atlas_destrieux_2009(lateralized=True)
destrieux_atlas_img = atlas_destrieux.maps
destrieux_labels = atlas_destrieux.labels

# Parcellate the fMRI time series
masker = NiftiLabelsMasker(labels_img=destrieux_atlas_img, standardize=True, memory="nilearn_cache", 
                           verbose=5, resampling_target="data", interpolation="nearest")
time_series = masker.fit_transform(fmri_img)

print(time_series.shape)