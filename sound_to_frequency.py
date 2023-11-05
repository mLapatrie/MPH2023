import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa
import os
import pandas as pd


def stimuli_to_bins(num_bins):
    stimuli = os.listdir("Dataset/stimuli")
    left_stimuli = []
    for stim in stimuli:
        if "Loc1" in stim or "Loc2" in stim:
            left_stimuli.append(stim)
    
    bin_stim_edges = []
    for sub in range(1, 7):
        sub_stim_amplitudes = []
        sub_stim_names = []
        for stim in left_stimuli:
            if f"sub0{sub}" in stim:
                sub_stim_names.append(stim[:-4])
                bin_edges, total_amplitudes = wav_to_bins(f"Dataset/stimuli/stim", num_bins)
                if len(bin_stim_edges) == 0:
                    bin_stim_edges = bin_edges
                sub_stim_amplitudes.append(total_amplitudes)
           
        sub_stims = np.array(sub_stim_names)
            
        np.save(f"parsed_input_data/sub0{sub}_stims.npy", sub_stims)
        np.save(f"parsed_input_data/sub0{sub}_stim_amplitudes.npy", sub_stim_amplitudes)
        print(f"DONE Subject 0{sub}")
    
    print("Saving bin edges")
    np.save(f"parsed_input_data/bin_edges.npy", bin_stim_edges)


def wav_to_bins(filename, num_bins):
    # Load the audio file with librosa
    samples, sample_rate = librosa.load(filename, sr=None)

    # For CWT
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(samples, scales, 'morl', 1.0 / sample_rate)

    # Define logarithmically spaced frequency bins
    bin_edges = np.logspace(np.log10(frequencies[1]), np.log10(frequencies[-1]), num_bins + 1)

    # Initialize an array to hold the total amplitude for each bin
    total_amplitudes = np.zeros(num_bins)

    # Sum the amplitudes within each bin
    for i in range(num_bins):
        x = zip(coefficients, frequencies)
        for pair in x:
            if bin_edges[i+1] <= pair[1] < bin_edges[i]:
                total_amplitudes[i] += np.sum(abs(pair[0]))
                
                
    return bin_edges, total_amplitudes


def csv_big_sound():
    big_sound = []
    sounds = os.listdir('Dataset/stimuli/')
    for s in sounds:
        print(f'C:/Repos/Simulated-Auditory-Evoked-Hemodynamics/Dataset/stimuli/{s}')
        big_sound.append(wav_to_bins(f'C:/Repos/Simulated-Auditory-Evoked-Hemodynamics/Dataset/stimuli/{s}', 16)[1])

    df = pd.DataFrame(big_sound)
    df.to_csv('big_sound.csv', index=False)
    
    
big_sound = []
for sub in range(1, 2):
    path = f"Dataset/sub-0{sub}/func/"
    runs = os.listdir(path)
    
    for run in runs:
        if "events.tsv" in run:
            # Load stimuli .tsv file
            stimuli_df = pd.read_csv(os.path.join(path, run))
            
            for index, stimulus in stimuli_df.iterrows():
                sound_name = str(stimulus).split(" ")[4].split("\\")[2].split('\n')[0][1:]
                print(sound_name)
                bin_edges, amplitudes = wav_to_bins(os.path.join("Dataset/stimuli/", sound_name), 16)
                normed_amplitudes = (amplitudes - amplitudes.min())/(amplitudes.max() - amplitudes.min())
                big_sound.append(normed_amplitudes)
            
df = pd.DataFrame(big_sound)
df.to_csv('big_sound.csv', index=False)