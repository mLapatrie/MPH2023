
import librosa
import torch
from nilearn import datasets, image

from sound_to_frequency import samples_to_bins
from show_activity import show_data
from nn import Net

# Take an audio input e.g (10 s)
# Break it down into 1 s segments 0.1 s apart.
# An audio input of 10s would have 100 windows of 1 s

filename = "Dataset/stimuli/sub01_Animal1_Loc1_ramp10.wav"

net = Net()
net.load_state_dict(torch.load('need a path'))

# Samples is an array of sample_rate * length length
samples, sample_rate = librosa.load(filename, sr=None)

# Load Destrieux atlas
atlas_destrieux = datasets.fetch_atlas_destrieux_2009(lateralized=True)
destrieux_atlas_img = image.load_img(atlas_destrieux.maps)

# Initialize constants
length = len(samples) * sample_rate
step = 100 # Go forward x ms before updating the display
nn_window = 1000 # Length of time to input into the neural network
cochlea_window = 100 # Length of time to input into the cochlea algorithm

index = 0
while index <= length - nn_window:
    nn_window_samples = samples[index:index+nn_window]
    cochlea_window_samples = samples[index:index+cochlea_window]
    
    nn_amplitudes = samples_to_bins(nn_window_samples, sample_rate, 16)[1]
    cochlea_amplitudes = samples_to_bins(cochlea_window_samples, sample_rate, 16)[1]
    
    # compute cochlea image
    
    
    # predict brain activity
    outputs = net(nn_amplitudes)
    
    # create brain image
    brain_display = show_data(outputs, destrieux_atlas_img)
    
    # create a mix of the 2 images
    
    
    # save final image
    
    
    brain_display.close()
    
    index += step
    
# create movie from final images