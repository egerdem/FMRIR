


import torch
from torchvision import transforms
from torchvision.utils import make_grid
import os
import json
import time
import wandb
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('Qt5Agg', force=True)   # or 'TkAgg'
import matplotlib.pyplot as plt

data_path = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"
# processed_file = os.path.join(data_path, f'processed_train.pt')
# data = torch.load(processed_file)

src_splits = {
    'test': [10, 11]
}
source_indices = range(*src_splits["test"])

fs = 2000  # 8000#16000  # Sampling frequency

irlen_algn = 136  # 128#512 # IR length for time-alignment
fftlen_algn = 128  # 512 # FFT length for time-alignment
t_algn = np.arange(0, irlen_algn) / fs  # Time for time-alignment
freq_algn = np.arange(1, fftlen_algn // 2 + 1) / fftlen_algn * fs  # Frequency for time-alignment

for src_id in tqdm(source_indices, desc=f"Loading test NPZ files"):
    npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")

    with np.load(npz_file) as data:
        specs = data['spec']  # Shape: (1331, 16, 16)
        source_pos = data['posSrc']  # Shape: (3,)
        mic_pos = data['posMic']  # Shape: (1331, 3)
        atf_mag_algn = data['atf_mag_algn'] #atf is 1331x64
        f_spec_algn = data['f_spec_algn']
        t_spec_algn = data['t_spec_algn']

f_test_ind = 10
atf_mags_algn_for_test_freq = atf_mag_algn[:, f_test_ind] #1331

# Extract coordinates and values
x_coords = mic_pos[:, 0]
y_coords = mic_pos[:, 1]
values = atf_mags_algn_for_test_freq

# Determine grid dimensions from unique coordinates
unique_x = np.unique(x_coords)
unique_y = np.unique(y_coords)
nx = len(unique_x)
ny = len(unique_y)

# Create a 2D grid to store the magnitude values, initializing with NaNs
grid_values = np.full((ny, nx), np.nan)

# Create a mapping from coordinate values to grid indices
x_map = {val: i for i, val in enumerate(unique_x)}
y_map = {val: i for i, val in enumerate(unique_y)}

# Populate the grid with magnitude values
for i in range(len(mic_pos)):
    ix = x_map[x_coords[i]]
    iy = y_map[y_coords[i]]
    grid_values[iy, ix] = values[i]

# Plotting the 2D image
plt.figure(figsize=(10, 8))
im = plt.imshow(grid_values, origin='lower', cmap='viridis',
           extent=[unique_x.min(), unique_x.max(), unique_y.min(), unique_y.max()],
           aspect='auto')
plt.colorbar(im, label="Magnitude")
plt.xlabel("X-coordinate")
plt.ylabel("Y-coordinate")
plt.title("ATF Magnitudes at Microphone Positions")
plt.show()






