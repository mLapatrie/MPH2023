
import librosa
import torch
import torch.nn as nn
import random
from nilearn import datasets, image
import numpy as np
import pandas as pd
import os
#import pygame
#from pygame.locals import *

from sound_to_frequency import samples_to_bins
from show_activity import show_data
#from cochlea_visual import renderCochlea
from nn import Net


# Take an audio input e.g (10 s)
# Break it down into 1 s segments 0.1 s apart.
# An audio input of 10s would have 90 windows of 1 s

common_labels = [1, 5, 6, 12, 13, 14, 15, 16, 17, 18, 21, 23, 24, 31, 32, 33, 34, 35, 37, 38, 39, 40, 48, 49, 50, 51, 53, 54, 62, 63, 64, 65, 67, 71, 73, 74, 76, 80, 81, 87, 88, 89, 90, 91, 92, 93, 96, 98, 99, 106, 107, 109, 110, 112, 113, 114, 115, 123, 124, 125, 126, 128, 129, 137, 138, 139, 140, 142, 146, 148, 149]
rois_labels = [17, 18, 33, 34, 49, 87, 92, 107, 109, 110, 124]

filename = "intro_silence.wav"#"McGill University - Biochemistry 2.wav"#"silent_1-second.wav"#"Dataset/stimuli/sub01_Animal1_Loc1_ramp10.wav"

# parse data
input_array = np.array(pd.read_csv("big_sound.csv"), dtype=np.float32)[:-1]
print(input_array.shape)
output_array = np.array(pd.read_csv("big_data.csv"), dtype=np.float32)[:-1]
print(input_array.shape)

# create movie from final images

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(input_array.shape[1], 50), # First fully connected layer
            #nn.ReLU(),
            nn.Linear(50, 100),
            nn.ReLU(),
            nn.Linear(100, 75),
            #nn.ReLU(),
            nn.Linear(75, 75),
            nn.Softmax(),
            nn.Linear(75, output_array.shape[1])  # Second fully connected layer
        )

    def forward(self, input):
        return self.main(input)


net = Net()
net.load_state_dict(torch.load("checkpoints_2/netG_epoch_100.pth"))
net.eval()
print("Model loaded")

# Samples is an array of sample_rate * length length
samples, sample_rate = librosa.load(filename, sr=None)

rms_energy = librosa.feature.rms(y=samples)

# The rms_energy contains a frame-wise RMS energy measurement.
# You can take the mean of these frames to get a single value of RMS energy.
average_rms_energy = rms_energy.mean()

print("Average RMS Energy: ", average_rms_energy)

max_amp = samples.max()
min_amp = samples.min()

# Load Destrieux atlas
atlas_destrieux = datasets.fetch_atlas_destrieux_2009(lateralized=True)
destrieux_atlas_img = image.load_img(atlas_destrieux.maps)

# Initialize constants
length = len(samples) / sample_rate * 1000
step = 100 # Go forward x ms before updating the display
nn_window = 1000 # Length of time to input into the neural network
cochlea_window = 100 # Length of time to input into the cochlea algorithm

index = 0
a = 0
while index <= length - nn_window:
    nn_window_samples = samples[index:index+nn_window]
    cochlea_window_samples = samples[index:index+cochlea_window]
    
    nn_amplitudes = np.array(samples_to_bins(nn_window_samples, sample_rate, 127)[1]).astype(np.float32)
    cochlea_amplitudes = samples_to_bins(cochlea_window_samples, sample_rate, 127)[1]
    print(type(nn_amplitudes))
    # compute cochlea image
    
    # predict brain activity
    predictions = np.zeros((nn_amplitudes.shape[0], 71))

    with torch.no_grad():
        for i, sample in enumerate([nn_amplitudes]):
            newTensor = torch.from_numpy(sample)
            predictions[i] = net(newTensor)
            
    predictions[0] = 2 * ( (predictions[0] - predictions[0].min()) / (predictions[0].max() - predictions[0].min()) ) - 1
    rms_energy = librosa.feature.rms(y=nn_window_samples)
    average_rms_energy = rms_energy.mean()
    for i in range(len(predictions[0])):
        if common_labels[i] in rois_labels:
            predictions[0][i] -= 1.5*(1 - random.random()*(average_rms_energy*10))
        else:
            predictions[0][i] += random.random()*0.5 - 0.25
        
    predictions[0] = 2 * ( (predictions[0] - predictions[0].min()) / (predictions[0].max() - predictions[0].min()) ) - 1
    
    print("done predicting")
    
    # create brain image
    
    frame_path = os.path.join("frames/", f'frame_{a:04d}.png')
    brain_display = show_data(predictions[0], destrieux_atlas_img)
    brain_display.savefig(frame_path)
    
    # create a mix of the 2 images
    
    
    # save final image
    
    
    brain_display.close()
    
    a += 1
    index += step