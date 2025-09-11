import torch
import sys
sys.path.append('AUTOENCODER/src')
from configs import config_FSMPAE_10026

# Load the reference model output
data_path = 'AUTOENCODER/outputs/out_20250323_FSMPAE_10026/atf_mag/atf_mag_test_ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200.pt'
atf_mag_est = torch.load(data_path, weights_only=False)