import numpy as np
data = np.load("parsed_input_data/sub01_stim_amplitudes.npy")
labels = np.load("parsed_input_data/sub01_stims.npy")
#edges = np.load("parsed_input_data/bin_edges.npy")
print(labels)
print(data)