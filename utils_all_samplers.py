





class OldSampleable(ABC):
    """
    Distribution which can be sampled from
    """

    @abstractmethod
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Args:
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, ...)
        """
        pass

class MNISTSampler(nn.Module, Sampleable):
    """
    Sampleable wrapper for the MNIST dataset
    """

    def __init__(self):
        super().__init__()
        # Try to handle SSL certificate issues
        import ssl
        import urllib.request

        # Create unverified SSL context as a workaround
        ssl._create_default_https_context = ssl._create_unverified_context

        try:
            self.dataset = datasets.MNIST(
                root='./data',
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((32, 32)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5,), (0.5,)),
                ])
            )
        except Exception as e:
            print(f"Error downloading MNIST: {e}")
            print("Please download MNIST manually or check your SSL certificates")
            raise e

        self.dummy = nn.Buffer(torch.zeros(1))  # Will automatically be moved when self.to(...) is called...

    def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:save_path
            - num_samples: the desired number of samples
        Returns:
            - samples: shape (batch_size, c, h, w)
            - labels: shape (batch_size, label_dim)
        """
        if num_samples > len(self.dataset):
            raise ValueError(f"num_samples exceeds dataset size: {len(self.dataset)}")

        indices = torch.randperm(len(self.dataset))[:num_samples]
        samples, labels = zip(*[self.dataset[i] for i in indices])
        samples = torch.stack(samples).to(self.dummy)
        labels = torch.tensor(labels, dtype=torch.int64).to(self.dummy.device)
        return samples, labels


# class SpectrogramSampler(nn.Module, Sampleable):
#     """
#     Sampleable wrapper for the RIR Spectrogram dataset.
#     Splits data into train/valid/test based on hardcoded source indices.
#     """
#
#     def __init__(self, data_path: str, mode: str, src_splits: Dict, transform: Optional[callable] = None):
#         super().__init__()
#         import os
#
#         self.transform = transform
#         self.mode = mode
#         self.src_splits = src_splits
#
#         # Check if a pre-processed file exists
#         # processed_file = os.path.join(os.path.dirname(data_path), 'processed_spectrograms.pt')
#         processed_file = os.path.join(data_path, f'processed_{self.mode}.pt')
#
#         re_process = True
#         if os.path.exists(processed_file):
#             print(f"Loading pre-processed {self.mode} data from {processed_file}")
#             data = torch.load(processed_file)
#             if 'sample_info' in data:
#                 self.spectrograms = data['spectrograms']
#                 self.coords = data['coords']
#                 self.sample_info = data['sample_info']
#                 re_process = False
#             else:
#                 print(f"Cached file {processed_file} is outdated. Re-processing.")
#
#         if re_process:
#             print(f"Processing {self.mode} data from .npz files...")
#
#             all_spectrograms = []
#             all_coords = []
#             all_sample_info = []
#
#             source_indices = parse_source_indices(self.src_splits, self.mode)
#
#             for src_id in tqdm(source_indices, desc=f"Loading {self.mode} NPZ files"):
#                 # Construct file path based on source index
#                 npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")
#
#                 if not os.path.exists(npz_file):
#                     print(f"Warning: File not found {npz_file}, skipping.")
#                     continue
#
#                 with np.load(npz_file) as data:
#                     specs = data['spec']  # Shape: (1331, 16, 16)
#                     source_pos = data['posSrc']  # Shape: (3,)
#                     mic_pos = data['posMic']  # Shape: (1331, 3)
#
#                     # Log-magnitude conversion
#                     log_specs = 10 * np.log10(specs + 1e-8)
#
#                     for i in range(log_specs.shape[0]):
#                         all_spectrograms.append(torch.tensor(log_specs[i], dtype=torch.float32))
#                         # Create the 6D coordinate vector [xs, ys, zs, xm, ym, zm]
#                         coord_vec = np.concatenate([source_pos, mic_pos[i]])
#                         all_coords.append(torch.tensor(coord_vec, dtype=torch.float32))
#                         all_sample_info.append(torch.tensor([src_id, i], dtype=torch.int32))
#
#             if not all_spectrograms:
#                 raise ValueError(f"No data loaded for mode '{self.mode}'. Check file paths and splits.")
#
#             # Stack all spectrograms and coordinates into tensors
#             self.spectrograms = torch.stack(all_spectrograms)
#             self.coords = torch.stack(all_coords)
#             self.sample_info = torch.stack(all_sample_info)
#
#             # Save the processed tensors for faster loading next time
#             torch.save({'spectrograms': self.spectrograms, 'coords': self.coords, 'sample_info': self.sample_info},
#                        processed_file)
#             print(f"Saved processed {self.mode} data to {processed_file}")
#
#         self.dummy = nn.Buffer(torch.zeros(1))
#         print(
#             f"Loaded {len(self.spectrograms) / 1331} * {1331} = {len(self.spectrograms)} spectrograms for {self.mode} set.")
#         print(f"Spectrogram tensor shape: {self.spectrograms.shape}")
#         print(f"Coordinate tensor shape: {self.coords.shape}")
#
#     def __len__(self):
#         return len(self.spectrograms)
#
#     def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """
#         Args:
#             - num_samples: the desired number of samples
#         Returns:
#             - samples: shape (batch_size, 1, H, W)
#             - labels: shape (batch_size, 6) for coordinates
#         """
#         if num_samples > len(self.spectrograms):
#             # Sample with replacement if requesting more than available
#             indices = torch.randint(0, len(self.spectrograms), (num_samples,))
#         else:
#             indices = torch.randperm(len(self.spectrograms))[:num_samples]
#
#         samples = self.spectrograms[indices]
#         labels = self.coords[indices]
#
#         # Apply transformations if they exist
#         if self.transform:
#             samples = self.transform(samples)
#
#         # Add channel dimension and move to device
#         return samples.unsqueeze(1).to(self.dummy.device), labels.to(self.dummy.device)
#
#     def get_item_by_idx(self, item_idx: int):
#         """Gets a single item (spectrogram, coords, info) by its flat index."""
#         sample = self.spectrograms[item_idx]
#         label = self.coords[item_idx]
#         info = self.sample_info[item_idx] if self.sample_info is not None else None
#
#         if self.transform:
#             # Apply transform to a single sample. We need to add a batch dim and remove it.
#             sample = self.transform(sample.unsqueeze(0)).squeeze(0)
#
#         return sample.unsqueeze(0).to(self.dummy.device), label.unsqueeze(0).to(self.dummy.device), info.unsqueeze(
#             0).to(self.dummy.device) if info is not None else None
#
#     def find_sample_index(self, src_id: int, mic_id: int):
#         """Finds the flat index for a given source and mic ID."""
#         if self.sample_info is None:
#             return None
#         # self.sample_info is a tensor of shape [N, 2] where each row is (src_id, mic_id)
#         results = (self.sample_info[:, 0] == src_id) & (self.sample_info[:, 1] == mic_id)
#         indices = torch.where(results)[0]
#         return indices[0].item() if len(indices) > 0 else None

# class ATFSliceSampler(torch.nn.Module, Sampleable):
#     """
#     Loads and serves 2D spatial slices of ATF magnitudes.
#
#     Each sample is a tensor of shape (64, 11, 11), representing the
#     64 frequency bins for an 11x11 grid of microphones at a single height.
#     """
#     def __init__(self, data_path: str, mode: str, src_splits: dict, transform: Optional[callable] = None,
#                  freq_up_to: Optional[int] = None):
#         super().__init__()
#         self.transform = transform
#         self.mode = mode
#         self.src_splits = src_splits
#         self.freq_up_to = freq_up_to
#
#         processed_file = os.path.join(data_path, f'processed_atf_{self.mode}.pt')
#
#         if os.path.exists(processed_file):
#             print(f"Loading pre-processed ATF {self.mode} data from {processed_file}")
#             data = torch.load(processed_file)
#             self.slices = data['slices']
#             self.coords = data['coords']
#             self.sample_info = data.get('sample_info')
#         else:
#             print(f"Processing ATF {self.mode} data from .npz files...")
#             source_indices = parse_source_indices(src_splits, self.mode)
#             all_slices = []
#             all_coords = []
#             # **NEW: Create a list to store metadata**
#             all_sample_info = []
#
#             for src_id in tqdm(source_indices, desc=f"Loading {self.mode} NPZ files"):
#                 npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")
#                 with np.load(npz_file) as data:
#                     atf_mags = data['atf_mag_algn']   # Shape: (1331, 64)
#                     mic_pos = data['posMic']          # Shape: (1331, 3)
#                     source_pos = data['posSrc']       # Shape: (3,)
#
#                     unique_z = np.unique(mic_pos[:, 2])
#
#                     for z_val in unique_z:
#                         slice_indices = np.where(mic_pos[:, 2] == z_val)[0]
#                         mic_pos_slice = mic_pos[slice_indices]
#                         atf_mags_slice = atf_mags[slice_indices]
#
#                         unique_x = sorted(np.unique(mic_pos_slice[:, 0]))
#                         unique_y = sorted(np.unique(mic_pos_slice[:, 1]))
#                         nx, ny = len(unique_x), len(unique_y)
#
#                         if nx * ny != len(mic_pos_slice):
#                             print(f"Warning: Skipping slice for src_id {src_id} at z={z_val} due to irregular grid.")
#                             continue
#
#                         x_map = {val: i for i, val in enumerate(unique_x)}
#                         y_map = {val: i for i, val in enumerate(unique_y)}
#
#                         # Pre-allocate for full frequency dimension; we'll crop later if requested
#                         grid_slice = torch.zeros((64, ny, nx), dtype=torch.float32)
#                         for i in range(len(mic_pos_slice)):
#                             ix, iy = x_map[mic_pos_slice[i, 0]], y_map[mic_pos_slice[i, 1]]
#                             grid_slice[:, iy, ix] = torch.tensor(atf_mags_slice[i])
#
#                         all_slices.append(grid_slice)
#                         coord_vec = np.concatenate([source_pos, [z_val]])
#                         all_coords.append(torch.tensor(coord_vec, dtype=torch.float32))
#                         all_sample_info.append(torch.tensor([src_id, z_val], dtype=torch.float32))
#
#             self.slices = torch.stack(all_slices)
#             self.coords = torch.stack(all_coords)
#             self.sample_info = torch.stack(all_sample_info)
#             torch.save({'slices': self.slices,
#                         'coords': self.coords,
#                        'sample_info': self.sample_info
#                         }, processed_file)
#             print(f"Saved processed ATF {self.mode} data to {processed_file}")
#
#         self.dummy = torch.nn.Buffer(torch.zeros(1))
#         # Optionally crop frequency channels after loading/processing
#         if self.freq_up_to is not None:
#             if self.freq_up_to < self.slices.shape[1]:
#                 self.slices = self.slices[:, :self.freq_up_to, :, :]
#
#         print(f"Loaded {len(self.slices)} ATF slices for {self.mode} set.")
#         print(f"Slice tensor shape: {self.slices.shape}")
#         print(f"Coordinate tensor shape: {self.coords.shape}")
#
#
#     def __len__(self):
#         return len(self.slices)
#
#
#     def plot(self, ind: int = 5, sample_idx: int = None):
#         """
#         Plots a 2D spatial slice of ATF magnitudes for a given sample and frequency.
#         'ind' corresponds to the frequency index.
#         """
#         if sample_idx is None:
#             sample_idx = random.randint(0, len(self) - 1)
#
#         # The user used 'ind', which we interpret as frequency index
#         freq_idx = ind
#
#         slice_to_plot = self.slices[sample_idx, freq_idx].cpu().numpy()
#
#         # plt.figure(figsize=(8, 6))
#         # im = plt.imshow(slice_to_plot, origin='lower', cmap='viridis', aspect='auto')
#         # plt.colorbar(im, label="Magnitude")
#         # plt.xlabel("X-index")
#         # plt.ylabel("Y-index")
#         # plt.title(f"ATF Slice - Sample {sample_idx}, Freq Index {freq_idx}")
#         # plt.show()
#
#
#     def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """
#         Args:
#             - num_samples: The desired number of samples.
#         Returns:
#             - samples: Tensor of shape (batch_size, 64, H, W).
#             - labels: Tensor of shape (batch_size, 4) for coordinates.
#         """
#         if num_samples > len(self.slices):
#             # Sample with replacement if requesting more samples than available
#             indices = torch.randint(0, len(self.slices), (num_samples,))
#         else:
#             # Sample without replacement for a random, unique batch
#             indices = torch.randperm(len(self.slices))[:num_samples]
#
#         samples = self.slices[indices]
#         labels = self.coords[indices]
#
#         if self.transform:
#             samples = self.transform(samples)
#
#         # The data is already (C, H, W), so we just move it to the correct device
#         return samples.to(self.dummy.device), labels.to(self.dummy.device)
#
#     def get_slice_by_id(self, src_id: int, z_height: float):
#         """Finds and returns a specific slice by source ID and z-height."""
#         if self.sample_info is None:
#             raise RuntimeError("Sampler was not initialized with sample_info. Please re-process the data.")
#
#         # Find all entries matching the source ID
#         src_matches = self.sample_info[:, 0] == src_id
#         # Find all entries matching the z-height (with a small tolerance for float comparison)
#         z_matches = torch.isclose(self.sample_info[:, 1], torch.tensor(z_height))
#
#         # Find the index where both conditions are true
#         combined_matches = src_matches & z_matches
#         indices = torch.where(combined_matches)[0]
#
#         if len(indices) == 0:
#             print(f"Warning: No slice found for Source ID {src_id} and Z-Height {z_height}.")
#             return None, None
#
#         # Get the first matching index
#         item_idx = indices[0].item()
#
#         # Retrieve the data
#         sample = self.slices[item_idx]
#         label = self.coords[item_idx]
#
#         if self.transform:
#             sample = self.transform(sample.unsqueeze(0)).squeeze(0)
#
#         # Return with a batch dimension of 1
#         return sample.unsqueeze(0).to(self.dummy.device), label.unsqueeze(0).to(self.dummy.device)
#
# class FreqConditionalATFSampler(torch.nn.Module, Sampleable):
#     """
#     Serves 2D spatial slices of ATF magnitudes, treating each frequency bin
#     as a separate sample and adding the frequency index to the conditioning vector.
#     """
#     def __init__(self, data_path: str, mode: str, src_splits: dict, freq_up_to: int,
#                  transform: Optional[callable] = None, ):
#         super().__init__()
#
#         # **NEW: Store the actual frequency values (in Hz)**
#         # We assume a fixed fftlen_algn of 128 and fs of 2000 from your generation script
#         fftlen_algn = 128
#         fs = 2000
#         # This creates the same frequency table you have in your screenshot
#         self.freq_algn = np.arange(1, fftlen_algn // 2 + 1) / fftlen_algn * fs
#         self.nyquist_freq = fs / 2  # The maximum possible frequency
#
#         self.transform = transform
#         self.mode = mode
#         self.src_splits = src_splits
#         self.num_freqs = freq_up_to
#
#         processed_file = os.path.join(data_path, f'processed_atf_{self.mode}.pt')
#
#         if os.path.exists(processed_file):
#             print(f"Loading pre-processed ATF {self.mode} data from {processed_file}")
#             data = torch.load(processed_file)
#             self.slices = data['slices']
#             self.coords = data['coords']
#             self.sample_info = data.get('sample_info')
#             # self.freq_algn = data['freq_algn']
#             # self.nyquist_freq = self.freq_algn[-1]
#
#         else:
#             print(f"Processing ATF {self.mode} data from .npz files...")
#             source_indices = parse_source_indices(src_splits, self.mode)
#             all_slices = []
#             all_coords = []
#             all_sample_info = []
#
#             for src_id in tqdm(source_indices, desc=f"Loading {self.mode} NPZ files"):
#                 npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")
#                 with np.load(npz_file) as data:
#                     atf_mags = data['atf_mag_algn']   # Shape: (1331, 64)
#                     mic_pos = data['posMic']          # Shape: (1331, 3)
#                     source_pos = data['posSrc']       # Shape: (3,)
#                     # self.freq_algn = data['freq_algn']
#                     # self.nyquist_freq = self.freq_algn[-1]
#
#                     unique_z = np.unique(mic_pos[:, 2])
#
#                     for z_val in unique_z:
#                         slice_indices = np.where(mic_pos[:, 2] == z_val)[0]
#                         mic_pos_slice = mic_pos[slice_indices]
#                         atf_mags_slice = atf_mags[slice_indices]
#
#                         unique_x = sorted(np.unique(mic_pos_slice[:, 0]))
#                         unique_y = sorted(np.unique(mic_pos_slice[:, 1]))
#                         nx, ny = len(unique_x), len(unique_y)
#
#                         if nx * ny != len(mic_pos_slice):
#                             print(f"Warning: Skipping slice for src_id {src_id} at z={z_val} due to irregular grid.")
#                             continue
#
#                         x_map = {val: i for i, val in enumerate(unique_x)}
#                         y_map = {val: i for i, val in enumerate(unique_y)}
#
#                         # Pre-allocate for full frequency dimension; we'll crop later if requested
#                         grid_slice = torch.zeros((64, ny, nx), dtype=torch.float32)
#                         for i in range(len(mic_pos_slice)):
#                             ix, iy = x_map[mic_pos_slice[i, 0]], y_map[mic_pos_slice[i, 1]]
#                             grid_slice[:, iy, ix] = torch.tensor(atf_mags_slice[i])
#
#                         all_slices.append(grid_slice)
#                         coord_vec = np.concatenate([source_pos, [z_val]])
#                         all_coords.append(torch.tensor(coord_vec, dtype=torch.float32))
#                         all_sample_info.append(torch.tensor([src_id, z_val], dtype=torch.float32))
#
#             self.slices = torch.stack(all_slices)
#             self.coords = torch.stack(all_coords)
#             self.sample_info = torch.stack(all_sample_info)
#             torch.save({'slices': self.slices,
#                         'coords': self.coords,
#                        'sample_info': self.sample_info
#                         }, processed_file)
#             print(f"Saved processed ATF {self.mode} data to {processed_file}")
#
#             # --- New Logic for Frequency-Conditional Sampling ---
#             # 1. Crop to the desired number of frequencies immediately after loading.
#
#         if freq_up_to > self.slices.shape[1]:
#             raise ValueError(
#                 f"freq_up_to ({freq_up_to}) cannot be larger than the number of available frequency bins ({self.slices.shape[1]}).")
#
#         self.slices = self.slices[:, :freq_up_to, :, :]
#
#
#         # --- Final Setup ---
#         self.dummy = torch.nn.Buffer(torch.zeros(1))
#
#         print(f"\n--- FreqConditionalATFSampler Initialized ({self.mode} mode) ---")
#         print(f"  Using {self.num_freqs} frequency bins per slice.")
#         print(f"  Number of original spatial slices: {len(self.slices)}")
#         print(f"  Total number of samples (slices * freqs): {len(self)}")
#         print(f"  Sample shape (before transform): (1, {self.slices.shape[2]}, {self.slices.shape[3]})")
#         print(f"  Label shape: ({self.coords.shape[1] + 1},)")
#         print("--------------------------------------------------")
#
#
#     def __len__(self):
#         # The total number of samples is num_slices * num_frequencies
#         return len(self.slices) * self.num_freqs
#
#
#     def plot(self, ind: int = 5, sample_idx: int = None):
#         """
#         Plots a 2D spatial slice of ATF magnitudes for a given sample and frequency.
#         'ind' corresponds to the frequency index.
#         """
#         if sample_idx is None:
#             sample_idx = random.randint(0, len(self) - 1)
#
#         # The user used 'ind', which we interpret as frequency index
#         freq_idx = ind
#
#         slice_to_plot = self.slices[sample_idx, freq_idx].cpu().numpy()
#
#         # plt.figure(figsize=(8, 6))
#         # im = plt.imshow(slice_to_plot, origin='lower', cmap='viridis', aspect='auto')
#         # plt.colorbar(im, label="Magnitude")
#         # plt.xlabel("X-index")
#         # plt.ylabel("Y-index")
#         # plt.title(f"ATF Slice - Sample {sample_idx}, Freq Index {freq_idx}")
#         # plt.show()
#
#     def sample(self, num_samples: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         # Generate random indices for the flattened dataset
#         indices = torch.randint(0, len(self), (num_samples,))
#
#         # Convert flat indices back to slice and frequency indices
#         slice_indices = indices // self.num_freqs
#         freq_indices = indices % self.num_freqs
#
#         # Get the single-frequency spatial slices
#         # Note: self.slices is (N, C, H, W). We need to gather along N and C.
#         samples = self.slices[slice_indices, freq_indices, :, :]
#
#         # Get the corresponding 4D coordinate labels
#         coord_labels = self.coords[slice_indices]
#         freq_hz_vals = torch.tensor(self.freq_algn[freq_indices.cpu()], dtype=torch.float32)
#
#         # 2. Normalize the Hz values to the range [0, 1]
#         normalized_freqs = freq_hz_vals / self.nyquist_freq
#
#         # Create the new 5D conditioning vector: [coords, freq_idx]
#         # freq_labels = freq_indices.float().unsqueeze(1).to(coord_labels.device) # old
#         # freq_labels = normalized_freqs.unsqueeze(1).to(coord_labels.device)
#         freq_labels = normalized_freqs.view(-1, 1).to(coord_labels.device)
#         labels = torch.cat([coord_labels, freq_labels], dim=1)
#
#         if self.transform:
#             samples = self.transform(samples)
#
#         # Return a single channel (the magnitude) and the new 5D label
#         # The channel dim is added here to make it (batch, 1, H, W)
#         return samples.unsqueeze(1).to(self.dummy.device), labels.to(self.dummy.device)
#
#     def get_slice_by_id(self, src_id: int, z_height: float, freq_idx: int):
#         """Finds and returns a specific slice by source ID and z-height."""
#         if self.sample_info is None:
#             raise RuntimeError("Sampler was not initialized with sample_info. Please re-process the data.")
#         if not (0 <= freq_idx < self.num_freqs):
#             raise IndexError(f"freq_idx {freq_idx} is out of bounds for the number of frequencies ({self.num_freqs}).")
#
#         # Find all entries matching the source ID
#         src_matches = self.sample_info[:, 0] == src_id
#         # Find all entries matching the z-height (with a small tolerance for float comparison)
#         z_matches = torch.isclose(self.sample_info[:, 1], torch.tensor(z_height))
#
#         # Find the index where both conditions are true
#         combined_matches = src_matches & z_matches
#         indices = torch.where(combined_matches)[0]
#
#         if len(indices) == 0:
#             print(f"Warning: No slice found for Source ID {src_id} and Z-Height {z_height}.")
#             return None, None
#
#         # Get the first matching index
#         item_idx = indices[0].item()
#
#         # 2. Retrieve the multi-channel slice and its 4D coordinate
#         full_slice = self.slices[item_idx]  # Shape: (num_freqs, 11, 11)
#         base_coord = self.coords[item_idx]  # Shape: (4,)
#
#         # 3. Select the specific frequency plane
#         sample = full_slice[freq_idx]  # Shape: (11, 11)
#
#         # 2. Look up the actual Hz value and normalize it
#         freq_hz_val = self.freq_algn[freq_idx]
#         normalized_freq = freq_hz_val / self.nyquist_freq
#
#         # 4. Construct the final 5D conditioning vector
#         freq_label = torch.tensor([normalized_freq], dtype=torch.float32, device=base_coord.device)
#         label = torch.cat([base_coord, freq_label])  # Shape: (5,)
#
#         # 5. Apply transform and return with a batch dimension of 1
#         if self.transform:
#             sample = self.transform(sample.unsqueeze(0)).squeeze(0)
#
#         # Return with a batch dimension of 1
#         return sample.unsqueeze(0).to(self.dummy.device), label.unsqueeze(0).to(self.dummy.device)

class SetEncoder(nn.Module):
    """
    Encodes a sparse set of observations into a sequence of tokens and a pooled context vector,
    using a dedicated positional encoding for the coordinates.
    """

    def __init__(self, num_freqs, d_model, nhead, num_layers):
        super().__init__()
        self.d_model = d_model

        # ### <<< CHANGE 1: Create two separate MLPs

        # An MLP for the "what": the ATF values
        self.value_tokenizer = nn.Sequential(
            nn.Linear(num_freqs, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # A separate MLP for the "where": the relative coordinates. This is our positional encoding.
        self.positional_encoder = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )

        # Transformer encoder remains the same
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.y_null_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, obs_coords_rel, obs_values, obs_mask):
        """
        Args:
            obs_coords_rel (Tensor): Relative mic coordinates [B, M_max, 3]
            obs_values (Tensor): ATF magnitudes at those mics [B, M_max, 20]
            obs_mask (Tensor): Boolean mask indicating valid observations [B, M_max]
        """
        # ### <<< CHANGE 2: Process values and positions separately

        # 1. Create value embeddings
        value_tokens = self.value_tokenizer(obs_values)

        # 2. Create positional embeddings
        positional_tokens = self.positional_encoder(obs_coords_rel)

        # 3. Add them together to get the final input tokens for the transformer
        tokens = value_tokens + positional_tokens

        # 4. Use transformer to let observations communicate with each other
        padding_mask = ~obs_mask
        encoded_tokens = self.transformer_encoder(tokens, src_key_padding_mask=padding_mask)

        # 5. Create the pooled context vector (this logic remains the same)
        masked_tokens = encoded_tokens.masked_fill(~obs_mask.unsqueeze(-1), 0.0)
        num_valid_tokens = obs_mask.sum(dim=1, keepdim=True)
        pooled_context = masked_tokens.sum(dim=1) / (num_valid_tokens + 1e-8)

        return encoded_tokens, pooled_context

# class CFGVectorFieldODE(ODE):
#     # V0: Original version without the
#     # Used in 2d UNET ATFSliceGenerator, and original MNIST
#     def __init__(self, net: ConditionalVectorField, guidance_scale: float = 1.0, y_dim: int = 6, y_embed_dim: int = 40):
#         self.net = net
#         self.guidance_scale = guidance_scale
#         # A learned embedding for the unconditional (null) case
#         self.y_null = nn.Parameter(torch.randn(y_embed_dim))
#
#     def drift_coefficient(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#         - x: (bs, c, h, w)
#         - t: (bs, 1, 1, 1)
#         - y: (bs, y_dim)
#         """
#         # For CFG, we need both the conditional and unconditional outputs
#         guided_vector_field = self.net(x, t, y)
#
#         # Create a batch of null embeddings for the unguided field
#         bs = x.shape[0]
#         unguided_y = self.y_null.repeat(bs, 1)  # was: unguided_y = torch.ones_like(y) * 10
#         unguided_vector_field = self.net(x, t, unguided_y)
#
#         combined_field = (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field
#
#         # --- ADD THIS CHECK TO HANDLE OLD MODELS ---
#         # The data part of the input state `x` has x.shape[1] - 1 channels.
#         # If the model's output has more channels than that, it's an old model.
#         num_data_channels = x.shape[1] - 1
#         if combined_field.shape[1] > num_data_channels:
#             # Slice off the extra, meaningless channel(s) to match the data.
#             return combined_field[:, :num_data_channels]
#
#         return combined_field

# class GenerativeSDE(SDE):
#     """
#     Implements the generative SDE for a DDPM trained with a Gaussian probability path.
#     This can be used for both stochastic sampling (SDE) and deterministic sampling (ODE).
#     """
#
#     def __init__(self, noise_predictor_network, set_encoder, config, guidance_scale=1.0):
#         super().__init__()
#         self.epsilon_theta = noise_predictor_network
#         self.set_encoder = set_encoder
#         self.guidance_scale = guidance_scale
#
#         # This is the sigma for the SDE solver, NOT the path sigma
#         self.sigma_sde = config['training'].get('sigma', 0.1)
#         self.architecture = config['model'].get('architecture_version', 'v1_legacy')
#
#     def _get_alpha_beta_derivatives(self, t):
#         """ For path xt = alpha_t * z + beta_t * epsilon, with alpha_t = t, beta_t = 1-t """
#         alpha_t = t
#         beta_t = 1 - t
#         alpha_dot_t = torch.ones_like(t)
#         beta_dot_t = -torch.ones_like(t)
#         return alpha_t, beta_t, alpha_dot_t, beta_dot_t
#
#     def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
#         # 1. Get conditioning from the SetEncoder
#         obs_coords_rel = kwargs['obs_coords_rel']
#         obs_values = kwargs['obs_values']
#         obs_mask = kwargs['obs_mask']
#         guided_y_tokens, guided_pooled_context = self.set_encoder(obs_coords_rel, obs_values, obs_mask)
#
#         # 2. Prepare unguided (null) conditioning
#         unguided_y_tokens = self.set_encoder.y_null_token.expand(xt.shape[0], guided_y_tokens.shape[1], -1)
#         unguided_pooled_context = self.set_encoder.y_null_token.squeeze(1).expand(xt.shape[0], -1)
#
#         # 3. Get guided and unguided noise predictions from the network (epsilon_theta)
#         model_kwargs_guided = {'context': guided_y_tokens, 'context_mask': obs_mask}
#         model_kwargs_unguided = {'context': unguided_y_tokens, 'context_mask': obs_mask}
#
#         if self.architecture == "v2_residual_context":
#             model_kwargs_guided['pooled_context'] = guided_pooled_context
#             model_kwargs_unguided['pooled_context'] = unguided_pooled_context
#
#         epsilon_theta_guided = self.epsilon_theta(xt, t.squeeze(), **model_kwargs_guided)
#         epsilon_theta_unguided = self.epsilon_theta(xt, t.squeeze(), **model_kwargs_unguided)
#
#         # 4. Combine using Classifier-Free Guidance (CFG)
#         epsilon_theta_final = (1 - self.guidance_scale) * epsilon_theta_unguided + self.guidance_scale * epsilon_theta_guided
#
#         # 5. Convert noise prediction to score prediction (s_theta)
#         # From notes, s_t = -epsilon_t / beta_t
#         alpha_t, beta_t, alpha_dot_t, beta_dot_t = self._get_alpha_beta_derivatives(t)
#         s_theta = -epsilon_theta_final / (beta_t + 1e-8)  # Add epsilon for stability as t->1
#
#         # 6. Calculate the flow field (u_theta) from the score (s_theta)
#         # From notes (Proposition 1, eq. 54), with our alpha_t, beta_t
#         # u_t = (beta_t^2 * alpha_dot_t / alpha_t - beta_dot_t * beta_t) * s_t + (alpha_dot_t / alpha_t) * x
#         # u_t = ((1-t)^2 * 1/t - (-1)*(1-t)) * s_t + (1/t) * xt
#         u_theta = ((beta_t.pow(2) * alpha_dot_t / (alpha_t + 1e-8) - beta_dot_t * beta_t)) * s_theta + (
#                     alpha_dot_t / (alpha_t + 1e-8)) * xt
#
#         # 7. Calculate the final drift for the generative SDE
#         # From notes (Summary 23, eq. 62): drift = u_t + (sigma_t^2 / 2) * s_t
#         drift = u_theta + (self.sigma_sde ** 2 / 2) * s_theta
#
#         return drift
#
#     def diffusion_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
#         return torch.tensor(self.sigma_sde, device=xt.device)

class ATFInpaintingTrainer(Trainer):
    def __init__(self, path: GaussianConditionalProbabilityPath, model: ConditionalVectorField,
                 eta: float, M: int, y_dim: int, sigma: float, flag_gaussian_mask: bool, model_mode: str,
                 **kwargs):
        super().__init__(models={'unet': model}, **kwargs)
        self.path = path
        self.eta = eta
        self.y_null = torch.nn.Parameter(torch.randn(1, y_dim))
        self.m = M
        self.sigma = sigma
        self.FLAG_GAUSSIAN_MASK = flag_gaussian_mask
        self.model_mode = model_mode

        # Flag to print shapes only on the first run
        self.shapes_printed = False

    def _create_sparse_mask(self, z: torch.Tensor) -> torch.Tensor:
        """Helper function to create a sparse mask for a batch of slices."""
        batch_size, _, H, W = z.shape
        mask = torch.zeros(batch_size, 1, H, W, device=z.device)

        # --- EFFICIENT BATCHED MASKING ---
        # Generate random indices for each sample in the batch without a loop
        num_pixels = (H - 1) * (W - 1)
        # We use multinomial to sample M indices for each of the batch_size samples
        indices = torch.multinomial(torch.ones(batch_size, num_pixels), self.m, replacement=False).to(z.device)

        rows = indices // (W - 1)
        cols = indices % (W - 1)

        # Use advanced indexing to set the mask values for the entire batch at once
        batch_indices = torch.arange(batch_size, device=z.device).view(-1, 1)
        mask[batch_indices, 0, rows, cols] = 1

        return mask

    def get_train_loss(self, **kwargs) -> torch.Tensor:

        # 1. Sample a batch of COMPLETE, clean ATF slices 'z'
        batch_size = kwargs.get('batch_size')
        z, y = self.path.p_data.sample(batch_size)

        # 2. Create the sparse mask efficiently
        mask = self._create_sparse_mask(z)
        z_masked = z * mask

        # 3. --- CORRECT INPAINTING PATH ---
        # The path is a straight line from the masked image to the full image, with optional noise.
        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        noise = torch.randn_like(z) * self.sigma

        if self.FLAG_GAUSSIAN_MASK:
            z0 = z_masked + (1 - mask) * noise
        # Create the noisy sample on the path between masked and full data
            x_t = (1 - t) * z0 + t * z
            ut_ref = z - z0
        else:
            # x_t = (1 - t) * z_masked + t * z + noise #
            x_t = (1 - (1-self.sigma)*t) * z_masked + t * z #
            # The target vector field is the difference vector
            ut_ref = z - z_masked  # The target velocity is the difference vector

        # x_t = self.path.sample_conditional_path(z_masked, t) # original version


        # --- Concatenate mask as 65th channel for the MODEL INPUT ---
        model_input = torch.cat([x_t, mask], dim=1)  # Shape becomes (bs, 65, 12, 12)

        # --- LABEL MASKING for CFG ---
        is_conditional_mask = (torch.rand(y.shape[0], device=y.device) > self.eta).view(-1, 1)

        y_null_on_device = self.y_null.to(y.device)
        y_cond = torch.where(is_conditional_mask, y, y_null_on_device)

        # --- Loss Calculation ---
        ut_theta = self.model(model_input, t, y_cond)

        # if self.model_mode == 'spatial':
            # Crop output and reference to 11x11 before comparing
            # ut_theta_crop = ut_theta[:, :-1, :-1, :-1]

        # elif self.model_mode == 'freq_cond':
        ut_theta_crop = ut_theta[:, :, :-1, :-1]

        ut_ref_crop = ut_ref[:, :, :-1, :-1]

        region_crop = (1.0 - mask)[:, :, :-1, :-1]
        squared_err = torch.square(ut_theta_crop - ut_ref_crop)*region_crop
        error = squared_err.sum() / region_crop.sum()
        # error = torch.mean()

        if not self.shapes_printed:
            print("\\n--- Tensor Shapes (First Training Step) ---")
            print(f"  Input Slice (z):          {z.shape}")
            print(f" Model Input (x_t + mask): {model_input.shape}\\n")
            print(f"  Masked Slice (z_masked):    {z_masked.shape}")
            print(f"  Noisy Sample (x_t):         {x_t.shape}")
            print(f"  Ground Truth Coords (y):    {y.shape}")
            print(f"  Null Embedding (y_null):    {self.y_null.shape}")
            print(f"  Final Condition (y_cond):   {y_cond.shape}")
            print(f"  Model Output (ut_theta):    {ut_theta.shape}")
            print(f"  No. of observations (M): {self.m}")
            print(" cropped loss' shape: ut_theta[:, :, :-1, :-1] ", ut_theta_crop.shape)
            print("------------------------------------------\\n")
            self.shapes_printed = True

        return error

    @torch.no_grad()
    def get_valid_loss(self, valid_sampler: Sampleable, **kwargs) -> torch.Tensor:
        # Validation loss should also simulate the inpainting task
        batch_size = kwargs.get('batch_size')
        z, y = valid_sampler.sample(batch_size)

        # Use the same efficient masking and path logic for validation
        mask = self._create_sparse_mask(z)
        # print(f"Validation mask shape: {mask.shape}, LHS: {mask.sum(dim=(-3, -2, -1)).unique().item()}, "
        #       f"RHS: {z.shape[-2] * z.shape[-1] - 1 - self.m}, z.shape: {z.shape}")
        # assert (1. - mask).sum(dim=(-3, -2, -1)).unique().item() == z.shape[-2] * z.shape[-1] - 1 - self.m

        z_masked = z * mask

        t = torch.rand(batch_size, 1, 1, 1, device=z.device)
        noise = torch.randn_like(z) * self.sigma

        if self.FLAG_GAUSSIAN_MASK:
            z0 = z_masked + (1 - mask) * noise
            x_t = (1 - t) * z0 + t * z
            ut_ref = z - z0
        else:
            # x_t = (1 - t) * z_masked + t * z + noise  #
            x_t = (1 - (1-self.sigma)*t) * z_masked + t * z + noise  #
            # The target vector field is the difference vector
            ut_ref = z - z_masked  # The target velocity is the difference vector

        model_input = torch.cat([x_t, mask], dim=1)

        ut_theta = self.model(model_input, t, y)  # Use the true label for validation
        # error = torch.mean(torch.square(ut_theta[:, :-1, :-1, :-1] - ut_ref[:, :, :-1, :-1]))

        # if self.model_mode == 'spatial':
            # Crop output and reference to 11x11 before comparing
            # ut_theta_crop = ut_theta[:, :-1, :-1, :-1]

        # elif self.model_mode == 'freq_cond':
        ut_theta_crop = ut_theta[:, :, :-1, :-1]

        # Crop output and reference to 11x11 before comparing
        ut_ref_crop = ut_ref[:, :, :-1, :-1]

        region_crop = (1.0 - mask)[:, :, :-1, :-1]
        squared_err = torch.square(ut_theta_crop - ut_ref_crop) * region_crop
        error = squared_err.sum() / region_crop.sum()

        return error

    # def visualize_masking(self, crop, sample_idx: int = 0, freq_idx: int = 5):
    #     """
    #     Samples one slice, applies the inpainting mask, and plots the original
    #     and masked versions side-by-side for a specific frequency index.
    #     """
    #     # 1. Sample a single complete, clean ATF slice 'z' and its condition 'y'
    #     z, y = self.path.p_data.sample(sample_idx)
    #
    #     # Get the padded height and width
    #     _, _, H, W = z.shape
    #
    #     # 2. --- DATA MASKING (Inpainting) ---
    #     # Create a mask that is the same size as the padded 12x12 image
    #     mask = torch.zeros(sample_idx, 1, H, W, device=z.device)
    #
    #     # Get M random pixel locations to keep
    #     indices = torch.randperm((H-1) * (W-1))[:self.m]
    #     # IM changing this to 11x11 since masking the last row and column is not meaningful as we'll discard
    #     rows = indices // (W - 1)
    #     cols = indices % (W - 1)
    #     mask[0, 0, rows, cols] = 1
    #
    #     z_masked = z * mask
    #
    #     # 3. --- Plotting ---
    #     # Detach tensors and move to CPU for numpy/matplotlib
    #     original_slice = z[0, freq_idx].cpu().numpy()
    #     masked_slice = z_masked[0, freq_idx].cpu().numpy()
    #
    #     if crop:
    #         # Crop to the region of interest if needed
    #         original_slice = original_slice[:-1, :-1]
    #         masked_slice = masked_slice[:-1, :-1]
    #
    #     fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    #     fig.suptitle(f"Masking Visualization (Frequency Index: {freq_idx})")
    #
    #     # Plot Original
    #     im1 = axes[0].imshow(original_slice, origin='upper', cmap='viridis')
    #     axes[0].set_title(f'Original Slice')
    #     axes[0].set_xlabel("X-index")
    #     axes[0].set_ylabel("Y-index")
    #     fig.colorbar(im1, ax=axes[0], label="Magnitude")
    #
    #     # Plot Masked
    #     im2 = axes[1].imshow(masked_slice, origin='upper', cmap='viridis')
    #     axes[1].set_title(f'Masked Slice ({self.m} points visible)')
    #     axes[1].set_xlabel("X-index")
    #     axes[1].set_ylabel("Y-index")
    #     fig.colorbar(im2, ax=axes[1], label="Magnitude")
    #
    #     # Plot Mask
    #     mask_slice = mask[0, 0].cpu().numpy()
    #     im3 = axes[2].imshow(mask_slice, origin='upper', cmap='gray', vmin=0, vmax=1)
    #     axes[2].set_title(f'Mask (1 = Visible, 0 = Hidden)')
    #     axes[2].set_xlabel("X-index")
    #     axes[2].set_ylabel("Y-index")
    #     fig.colorbar(im3, ax=axes[2], label="Mask binary value")
    #
    #     plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle
    #     plt.show()

# class ResidualLayer3D(nn.Module):
#     def __init__(self, channels: int, time_embed_dim: int, context_embed_dim: int):
#         super().__init__()
#
#         self.block1 = nn.Sequential(
#             nn.SiLU(),
#             nn.BatchNorm3d(channels),
#             nn.Conv3d(channels, channels, kernel_size=3, padding=1)
#         )
#         self.block2 = nn.Sequential(
#             nn.SiLU(),
#             nn.BatchNorm3d(channels),
#             nn.Conv3d(channels, channels, kernel_size=3, padding=1)
#         )
#         # Converts (bs, time_embed_dim) -> (bs, channels)
#         self.time_adapter = nn.Sequential(
#             nn.Linear(time_embed_dim, time_embed_dim),
#             nn.SiLU(),
#             nn.Linear(time_embed_dim, channels)
#         )
#
#         # Adapter for pooled context embedding
#         self.context_adapter = nn.Sequential(
#             nn.Linear(context_embed_dim, channels),
#             nn.SiLU(),
#             nn.Linear(channels, channels)
#         )
#
#     def forward(self, x: torch.Tensor, t_embed: torch.Tensor, context_embed) -> torch.Tensor:
#         """
#         Args:
#         - x: (bs, c, h, w)
#         - t_embed: (bs, t_embed_dim)
#         - y_embed: (bs, y_embed_dim)
#         """
#         res = x.clone()  # (bs, c, h, w)
#
#         # Initial conv block
#         x = self.block1(x)  # (bs, c, h, w)
#
#         # Add time embedding
#         t_add = self.time_adapter(t_embed).view(x.shape[0], -1, 1, 1, 1)
#         x = x + t_add
#
#         # Add y embedding (conditional embedding)
#         c_add = self.context_adapter(context_embed).view(x.shape[0], -1, 1, 1, 1)
#         x = x + c_add
#
#         # Second conv block
#         x = self.block2(x)  # (bs, c, h, w)
#
#         # Add back residual
#         x = x + res  # (bs, c, h, w)
#
#         return x

#NEW: DiT

def modulate(x, shift, scale):
    """ Helper function for adaLN """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DiTBlock(nn.Module):
    """
    A block of a Diffusion Transformer, with adaptive layer norm (adaLN) for conditioning.
    """

    def __init__(self, d_model, nhead, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.SiLU(),
            nn.Linear(mlp_hidden_dim, d_model),
        )
        # The adaLN MLP that generates scale and shift parameters
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_model, 6 * d_model, bias=True)
        )

    def forward(self, x, c):
        # c is the conditioning vector (from time + pooled_context)
        # It's used to generate scale/shift for norm1, norm2, and a final output gate
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)

        # Self-Attention block
        x_msa = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(x_msa, x_msa, x_msa)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # MLP block
        x_mlp = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_mlp)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


# class DiffusionTransformer3D(nn.Module):
#     """
#     A Diffusion Transformer for 3D volumetric data.
#     """
#
#     def __init__(self, input_size=11, patch_size=4, in_channels=20, out_channels=20,
#                  d_model=512, depth=12, nhead=8):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.patch_size = patch_size
#         self.d_model = d_model
#
#         # --- Padding Calculation ---
#         # Pad input to be divisible by the patch size
#         target_size = math.ceil(input_size / patch_size) * patch_size
#         total_pad = target_size - input_size
#         pad_front = total_pad // 2
#         pad_back = total_pad - pad_front
#         self.padding_tuple = (pad_front, pad_back, pad_front, pad_back, pad_front, pad_back)
#         self.crop_start = pad_front
#         self.crop_end = pad_front + input_size
#
#         # 1. Patching and Linear Embedding (done in one step with a Conv3D)
#         self.patch_embed = nn.Conv3d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
#
#         # Calculate number of patches
#         num_patches = (target_size // patch_size) ** 3
#
#         # 2. Positional Embedding
#         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, d_model))
#
#         # 3. Time Embedding and Conditioning MLP
#         self.time_embedder = FourierEncoder(d_model)
#         # This MLP will process time + pooled_context to generate the adaLN parameters
#         self.conditioning_mlp = nn.Sequential(
#             nn.Linear(d_model * 2, d_model),
#             nn.SiLU(),
#             nn.Linear(d_model, d_model)
#         )
#
#         # 4. Transformer Blocks
#         self.blocks = nn.ModuleList([
#             DiTBlock(d_model, nhead) for _ in range(depth)
#         ])
#
#         # 5. Final Layer and Un-patching
#         self.final_norm = nn.LayerNorm(d_model, elementwise_affine=False, eps=1e-6)
#         self.final_adaLN_modulation = nn.Sequential(
#             nn.SiLU(),
#             nn.Linear(d_model, 2 * d_model, bias=True)
#         )
#         self.unpatch_proj = nn.Linear(d_model, patch_size * patch_size * patch_size * out_channels)
#         self.unpatch_pd, self.unpatch_ph, self.unpatch_pw = target_size // patch_size, target_size // patch_size, target_size // patch_size
#
#     def forward(self, x, t, pooled_context):
#         # Note: This forward signature is different from the U-Net.
#         # It takes 'pooled_context' directly, not the token sequence.
#
#         x = F.pad(x, self.padding_tuple, mode='reflect')
#         B = x.shape[0]
#
#         # Patching and embedding
#         x = self.patch_embed(x)  # (B, d_model, D/p, H/p, W/p)
#         x = x.flatten(2).transpose(1, 2)  # (B, num_patches, d_model)
#
#         # Add positional embedding
#         x = x + self.pos_embed
#
#         # Prepare conditioning vector
#         t_emb = self.time_embedder(t.squeeze())
#         c = self.conditioning_mlp(torch.cat([t_emb, pooled_context], dim=1))
#
#         # Apply Transformer blocks
#         for block in self.blocks:
#             x = block(x, c)
#
#         # Final modulation and projection
#         shift, scale = self.final_adaLN_modulation(c).chunk(2, dim=1)
#         x = modulate(self.final_norm(x), shift, scale)
#         x = self.unpatch_proj(x)
#
#         # Un-patchify
#         x = x.view(B, self.unpatch_pd, self.unpatch_ph, self.unpatch_pw, self.patch_size, self.patch_size,
#                    self.patch_size, self.out_channels)
#         x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous().view(B, self.out_channels, self.unpatch_pd * self.patch_size,
#                                                                 self.unpatch_ph * self.patch_size,
#                                                                 self.unpatch_pw * self.patch_size)
#
#         # Crop back to original size
#         s, e = self.crop_start, self.crop_end
#         return x[..., s:e, s:e, s:e]


# class CFGVectorFieldODE_DiT_3D(ODE):
#     """ ODE wrapper for the 3D Diffusion Transformer. """
#
#     def __init__(self, unet, set_encoder, guidance_scale=1):
#         self.unet = unet  # Here 'unet' is actually our DiT model
#         self.set_encoder = set_encoder
#         self.guidance_scale = guidance_scale
#
#     def drift_coefficient(self, xt: torch.Tensor, t: torch.Tensor, **kwargs) -> torch.Tensor:
#         # Note: We only need obs_coords_rel and obs_values to get the pooled_context
#         # The DiT itself does not use the token sequence y_tokens.
#         obs_coords_rel = kwargs['obs_coords_rel']
#         obs_values = kwargs['obs_values']
#         obs_mask = kwargs['obs_mask']
#
#         # Get the pooled context for the guided prediction
#         _, guided_pooled_context = self.set_encoder(obs_coords_rel, obs_values, obs_mask)
#         guided_vector_field = self.unet(xt, t.squeeze(), pooled_context=guided_pooled_context)
#
#         # Get the pooled context for the unguided prediction (the null token)
#         unguided_pooled_context = self.set_encoder.y_null_token.squeeze(1).expand(xt.shape[0], -1)
#         unguided_vector_field = self.unet(xt, t.squeeze(), pooled_context=unguided_pooled_context)
#
#         # Combine using the CFG formula
#         return (1 - self.guidance_scale) * unguided_vector_field + self.guidance_scale * guided_vector_field


# class DiTTrainer3D(Trainer):
#     def __init__(self, path, model, set_encoder, eta, M_range, sigma, grid_xyz, loss_type: str,
#                  coord_mean: torch.Tensor, coord_std: torch.Tensor, **kwargs):
#         super().__init__(models={'dit': model, 'set_encoder': set_encoder})
#         self.path = path
#         self.set_encoder = set_encoder
#         self.eta = eta
#         self.M_range = (int(M_range[0]), int(M_range[1]))
#         self.sigma = sigma
#         self.grid_xyz = grid_xyz.to(next(model.parameters()).device)
#         self.loss_type = loss_type
#         self.coord_mean = coord_mean.to(next(model.parameters()).device)
#         self.coord_std = coord_std.to(next(model.parameters()).device)
#
#         if self.loss_type == 'weighted':
#             print("--- DiT Trainer: Using PERCEPTUALLY WEIGHTED training loss. ---")
#         else:
#             print("--- DiT Trainer: Using STANDARD training loss. ---")
#
#     def make_observation_set(self, z_full, src_xyz):
#         # This method can be copied directly from ATF3DTrainer
#         # Or you can move it to a shared parent class to avoid code duplication.
#         B, C, D, H, W = z_full.shape
#         dev = z_full.device
#         grid_xyz = self.grid_xyz.to(dev)
#         src_xyz = src_xyz.to(dev)
#         N = self.grid_xyz.shape[0]
#         M_max = self.M_range[1]
#         obs_coords_rel_list, obs_values_list, obs_mask_list = [], [], []
#
#         for i in range(B):
#             M = torch.randint(self.M_range[0], self.M_range[1] + 1, (1,)).item()
#             obs_indices = torch.randperm(N, device=dev)[:M]
#             obs_xyz = self.grid_xyz[obs_indices]
#             obs_coords_rel = obs_xyz - src_xyz[i].unsqueeze(0)
#             obs_coords_rel = (obs_coords_rel - self.coord_mean) / (self.coord_std + 1e-8)
#             z_flat = z_full[i].view(C, -1)
#             obs_values = z_flat[:, obs_indices].transpose(0, 1)
#             pad_len = M_max - M
#             obs_coords_rel_padded = nn.functional.pad(obs_coords_rel, (0, 0, 0, pad_len))
#             obs_values_padded = nn.functional.pad(obs_values, (0, 0, 0, pad_len))
#             mask = torch.zeros(M_max, dtype=torch.bool, device=dev)
#             mask[:M] = True
#             obs_coords_rel_list.append(obs_coords_rel_padded)
#             obs_values_list.append(obs_values_padded)
#             obs_mask_list.append(mask)
#
#         return torch.stack(obs_coords_rel_list), torch.stack(obs_values_list), torch.stack(obs_mask_list)
#
#     def get_train_loss(self, **kwargs) -> torch.Tensor:
#         batch_size = kwargs.get('batch_size')
#         z_full, src_xyz, _ = self.path.p_data.sample(batch_size)
#         dev = next(self.model.parameters()).device
#         z_full, src_xyz = z_full.to(dev), src_xyz.to(dev)
#         x1 = z_full
#
#         obs_coords_rel, obs_values, obs_mask = self.make_observation_set(z_full, src_xyz)
#
#         # The DiT only needs the pooled_context for conditioning.
#         _, pooled_context = self.set_encoder(obs_coords_rel, obs_values, obs_mask)
#
#         t = torch.rand(batch_size, device=x1.device).view(-1, 1, 1, 1, 1)
#         x0 = torch.randn_like(x1)
#         xt = (1 - (1 - self.sigma) * t) * x0 + t * x1
#         ut_ref = x1 - (1 - self.sigma) * x0
#
#         # Apply CFG dropout to the pooled_context
#         is_conditional_mask = (torch.rand(batch_size, device=x1.device) > self.eta)
#         null_context = self.set_encoder.y_null_token.squeeze(1).expand(batch_size, -1)
#         final_pooled_context = torch.where(is_conditional_mask.view(-1, 1), pooled_context, null_context)
#
#         # Get the model's prediction
#         ut_theta = self.model(xt, t, pooled_context=final_pooled_context)
#
#         # Compute loss (standard or weighted)
#         if self.loss_type == 'weighted':
#             with torch.no_grad():
#                 xt_denorm = xt * self.path.p_data.std + self.path.p_data.mean
#                 xt_linear = 10 ** (xt_denorm / 20.0)
#                 weights = 1.0 / (xt_linear + 1e-6)
#                 weights = torch.clamp(weights, max=10.0)
#             loss = torch.mean(torch.square(weights * (ut_theta - ut_ref)))
#         else:
#             loss = torch.mean(torch.square(ut_theta - ut_ref))
#
#         return loss
#
#     @torch.no_grad()
#     def get_valid_loss(self, valid_sampler: Sampleable, **kwargs) -> torch.Tensor:
#         batch_size = kwargs.get('batch_size')
#         z_full, src_xyz, _ = valid_sampler.sample(batch_size)
#         dev = next(self.model.parameters()).device
#         z_full, src_xyz = z_full.to(dev), src_xyz.to(dev)
#         x1 = z_full
#
#         obs_coords_rel, obs_values, obs_mask = self.make_observation_set(z_full, src_xyz)
#         _, pooled_context = self.set_encoder(obs_coords_rel, obs_values, obs_mask)
#
#         t = torch.rand(batch_size, device=x1.device).view(-1, 1, 1, 1, 1)
#         x0 = torch.randn_like(x1)
#         xt = (1 - (1 - self.sigma) * t) * x0 + t * x1
#         ut_ref = x1 - (1 - self.sigma) * x0
#
#         # For validation, we are always conditional
#         ut_theta = self.model(xt, t, pooled_context=pooled_context)
#
#         # Validation loss is always standard MSE
#         loss = torch.mean(torch.square(ut_theta - ut_ref))
#
#         return loss
