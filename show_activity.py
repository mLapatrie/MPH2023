import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image, datasets, plotting
import os


def show_data(activation_values):
    masked_data = np.zeros(destrieux_atlas_img.shape)

    common_labels = [1, 5, 6, 12, 13, 14, 15, 16, 17, 18, 21, 23, 24, 31, 32, 33, 34, 35, 37, 38, 39, 40, 48, 49, 50, 51, 53, 54, 62, 63, 64, 65, 67, 71, 73, 74, 76, 80, 81, 87, 88, 89, 90, 91, 92, 93, 96, 98, 99, 106, 107, 109, 110, 112, 113, 114, 115, 123, 124, 125, 126, 128, 129, 137, 138, 139, 140, 142, 146, 148, 149]

    for i, index in enumerate(common_labels):
        masked_data[destrieux_atlas_img.get_fdata() == index] = activation_values[i]
        
    masked_img = image.new_img_like(destrieux_atlas_img, masked_data)

    display = plotting.plot_stat_map(masked_img, display_mode='ortho', cut_coords=[0, -70, 0], title='Destrieux Atlas Activation')
    return display
    
# Load Destrieux atlas
atlas_destrieux = datasets.fetch_atlas_destrieux_2009(lateralized=True)
destrieux_atlas_img = image.load_img(atlas_destrieux.maps)
destrieux_labels = atlas_destrieux.labels
def generate_frames(big_data):
    

    frame_dir = 'frames'
    os.makedirs(frame_dir, exist_ok=True)

    for i, activation_values in enumerate(big_data):
        frame_path = os.path.join(frame_dir, f'frame_{i:04d}.png')
        
        display = show_data(activation_values)
        display.savefig(frame_path)
        display.close()
        
        print("DONE", i)


#big_data = np.array(pd.read_csv('big_data.csv'))
big_data = np.load('big_data.npy')
generate_frames(big_data)