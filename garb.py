import torch
import sys
# sys.path.append('AUTOENCODER/src')
# from configs import config_FSMPAE_10026
from model_paths import MODEL_LOAD_PATH
from fm_utils import ATF3DSampler
# Load the reference model output
# data_path = 'AUTOENCODER/outputs/out_20250323_FSMPAE_10026/atf_mag/atf_mag_test_ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200.pt'
# atf_mag_est = torch.load(data_path, weights_only=False)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
# checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
# config = checkpoint.get('config', {})
# data_dir = "ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200/"  # Override with local
# src_split = config['data']['src_splits']
# src_split = {"train": [[0, 820], [1324, 8192]], "valid": [[820, 922], [1024, 1324]],
#                                 "test": [922, 1024]}
# src_split = {"train": [0, 820], "valid": [820, 922], "test": [922, 1024]}
# freq_up_to = config['model'].get('freq_up_to')


# Load data
# train_sampler = ATF3DSampler(
#     data_path=data_dir, mode='train', src_splits=src_split,
#     normalize=True, freq_up_to=20
# )

import os
import numpy as np
# for src_id in tqdm(source_indices, desc=f"Loading {self.mode} NPZ files"):
# for src_id in range(6889, 6890):
#     data_path = "/Users/ege/Projects/FMRIR/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200_freq20"
#     npz_file = os.path.join(data_path, f"data_s{src_id + 1:04d}.npz")
#
#     with np.load(npz_file) as data_single:
#         atf_mag_algn = data_single['atf_mag_algn']  # (1331, 64)
#         np_of_mics, np_of_freqs = atf_mag_algn.shape
#         source_pos = data_single['posSrc']  # (3,)

idx_mes_pos_mat = np.load("idx_mes_pos_s1024_m1331.npy")

z = idx_mes_pos_mat[:5, 0]

import torch
data = torch.load('RESULTS/out_20250324_KRR_10004/atf_mag/atf_mag_test_ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200.pt',weights_only=False)
print('KRR Shape:', data.shape)
print('Type:', type(data))