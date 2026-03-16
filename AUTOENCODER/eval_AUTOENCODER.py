import numpy as np
import matplotlib
matplotlib.use('Qt5Agg', force=True)  # or 'TkAgg'
from matplotlib import pyplot as plt
import torch as th

import src.dataset as dataset
from src.configs import config_FSMPAE_10026
import src.utils as utils

model = 'FSMPAE'#'NF'#'FSMPAE'#'PINN'#'FSMPAEPI'#
pt_dir = 'outputs/out_20250323_FSMPAE_10026'#'outputs/out_20250325_NF_10003'#'outputs/out_20250324_KRR_10004'#'outputs/out_20241129_PINN_10005'#'outputs/out_20250320_FSMPAEPI_10007'#
config = config_FSMPAE_10026.copy()#config_NF_10003.copy()#config_KRR_10004.copy()#config_PINN_10005.copy()#config_FSMPAEPI_10007.copy()#

idataset = dataset.ATFdataset(config=config)
data = idataset.Data

dataset_name = config['dataset'][0]
freq = np.arange(1,config["num_freq"]+1)/config["num_freq"]*config['fs']/2

src_index = config['src_index'][dataset_name]['test']
src_position = data['test']['src_position'][dataset_name]
mic_position = data['test']['mic_position'][dataset_name]
atf_mag_gt = data['test']['atf_mag'][dataset_name]

pt_path = f'{pt_dir}/atf_mag/atf_mag_test_{dataset_name}.pt'
atf_mag_est = th.load(pt_path)

lsd = utils.LSD()

lsd_val = lsd(atf_mag_gt, atf_mag_est, dim=1, mean=False)
lsd_mean = lsd_val.mean()
lsd_std = lsd_val.std()
print(f'LSD: {lsd_mean:.4f} +- {lsd_std:.4f} dB')

src_id = 0
freq_id = 15#31#
print("Frequency: ", freq[freq_id])

idx_dist = np.where(np.abs(mic_position[:,2,src_id])<1e-6)
dist_gt = atf_mag_gt[idx_dist, freq_id, src_id].reshape(11, 11)
dist_est = atf_mag_est[idx_dist, freq_id, src_id].reshape(11, 11)

plt.rcParams["font.size"] = 18

xx = mic_position[idx_dist,0,0].reshape(11, 11)
yy = mic_position[idx_dist,1,0].reshape(11, 11)

fig, ax = plt.subplots(figsize=(6,5))
color = plt.pcolormesh(xx, yy, dist_gt, cmap='pink', shading='auto')
ax.set_aspect('equal')
color.set_clim(-20,5)
cbar = plt.colorbar(color)
cbar.set_label('Magnitude (dB)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
# plt.savefig("dist_gt.pdf")
plt.show()


fig, ax = plt.subplots(figsize=(6,5))
color = plt.pcolormesh(xx, yy, dist_est, cmap='pink', shading='auto')
ax.set_aspect('equal')
color.set_clim(-20,5)
cbar = plt.colorbar(color)
cbar.set_label('Magnitude (dB)')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
# plt.savefig("dist_est.pdf")
plt.show()

src_id = 931
src_id = src_id - 922
mic_id = 0#4630#
print(f"Src: {src_position[mic_id,:,src_id]}, Mic: {mic_position[mic_id,:,src_id]}")

fig, ax = plt.subplots(figsize=(14,4))
plt.plot(freq, atf_mag_gt[mic_id,:,src_id], 'k--')
plt.plot(freq, atf_mag_est[mic_id,:,src_id], 'r-')
ax.set_xscale('log')
plt.axis('tight')
plt.grid()
#plt.title(f"Src: ({src_position[mic_id,0,src_id]: .1f}, {src_position[mic_id,1,src_id]: .1f}, {src_position[mic_id,2,src_id]: .1f}), Mic: ({mic_position[mic_id,0,src_id]: .1f}, {mic_position[mic_id,1,src_id]: .1f}, {mic_position[mic_id,2,src_id]: .1f})")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.legend(["True", "Estimated"])
plt.tight_layout()
# plt.savefig("atf_mag.pdf")
plt.show()