import numpy as np
import matplotlib.pyplot as plt
import pywt
import librosa


def wav_to_bins(filename, num_bins):
    # Replace 'your_audio_file.wav' with the path to your audio file
    filename = 'sounds\sub01_Tool9_Loc2_ramp10.wav'

    # Load the audio file with librosa
    samples, sample_rate = librosa.load(filename, sr=None)

    # For CWT
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(samples, scales, 'morl', 1.0 / sample_rate)

    # Define logarithmically spaced frequency bins
    num_bins = 16  # You can choose the number of bins based on your requirements
    bin_edges = np.logspace(np.log10(frequencies[1]), np.log10(frequencies[-1]), num_bins + 1)

    # Initialize an array to hold the total amplitude for each bin
    total_amplitudes = np.zeros(num_bins)

    # Sum the amplitudes within each bin
    for i in range(num_bins):
        x = zip(coefficients, frequencies)
        for pair in x:
            print(bin_edges[i+1], pair[1], bin_edges[i])
            if bin_edges[i+1] <= pair[1] < bin_edges[i]:
                total_amplitudes[i] += np.sum(abs(pair[0]))
                
                
    return bin_edges, total_amplitudes
            