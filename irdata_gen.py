import numpy as np
import json
import argparse
import os
import pyroomacoustics as pra
from scipy.fftpack import fft
from scipy.signal import windows
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# import matplotlib
# matplotlib.use('TKAgg')

from utils import dim2cuboid

if __name__ == "__main__":
    # Set random seed
    seed = 0
    np.random.seed(seed)

    parser = argparse.ArgumentParser(description='Generate impulse responses by image source method')
    parser.add_argument('--num_src', '-s', type=int, default=1, help='Number of sources')
    parser.add_argument('--rt60', '-r', type=int, default=200, help='Target RT60 (ms)')
    parser.add_argument('--int_mic', '-i', type=float, default=0.1, help='Microphone interval (m)')

    args = parser.parse_args()

    # Parameters
    fs = 8000#16000  # Sampling frequency
    c = 343  # Speed of sound
    rt60_tgt = args.rt60/1000  # Target RT60 (s) (default=200)
    num_src = args.num_src # Number of sources (default=1)

    irlen = 1024#2048#4096  # IR length
    fftlen = 1024#2048#4096  # FFT length
    t = np.arange(0, irlen)/fs # Time
    freq = np.arange(1,fftlen//2+1)/fftlen*fs # Frequency
    window = windows.hamming(irlen) # Window function
    
    irlen_algn = 512 # IR length for time-alignment
    fftlen_algn = 512 # FFT length for time-alignment
    t_algn = np.arange(0, irlen_algn)/fs # Time for time-alignment
    freq_algn = np.arange(1,fftlen_algn//2+1)/fftlen_algn*fs # Frequency for time-alignment
    window_algn = windows.hamming(irlen_algn) # Window function for time-alignment

    room_dim = [4, 6, 3]  # Room size (m)
    mic_dim = [1, 1, 1] # Size of measurement region (m)
    mic_region = [[1.5, 2.5], [2.5, 3.5], [1, 2]]  # Range of measruement region (m)
    src_region = [[0, 4], [0, 6], [0, 3]]  # Range of source region (m)
    center = [mic_region[0][0]+mic_dim[0]/2, mic_region[1][0]+mic_dim[1]/2, mic_region[2][0]+mic_dim[2]/2] # Coordinate origin (center of measurement region)
    
    # Source positions randomly generated from the region outside the measurement region
    for i in range(num_src):
        while True:
            pos = np.array([np.random.uniform(low=src_region[0][0], high=src_region[0][1]),
                            np.random.uniform(low=src_region[1][0], high=src_region[1][1]),
                            np.random.uniform(low=src_region[2][0], high=src_region[2][1])])
            #if not((pos[0]>=mic_region[0][0] and pos[0]<=mic_region[0][1]) and (pos[1]>=mic_region[1][0] and pos[1]<=mic_region[1][1]) and (pos[2]>=mic_region[2][0] and pos[2]<=mic_region[2][1])):
            if (pos[0]<mic_region[0][0] or pos[0]>mic_region[0][1]) or (pos[1]<mic_region[1][0] or pos[1]>mic_region[1][1]) or (pos[2]<mic_region[2][0] or pos[2]>mic_region[2][1]):
                print(f"Source {i+1} position: {pos}")
                if i == 0:
                    src_position = pos[None,:]
                else:
                    src_position = np.vstack((src_position, pos))
                break

    # Microphone grid
    mic_interval = args.int_mic #0.2 #0.1 #0.125 #
    numGridAxis = int(mic_dim[0]/mic_interval)+1
    x_vals = np.linspace(mic_region[0][0], mic_region[0][1], numGridAxis)
    y_vals = np.linspace(mic_region[1][0], mic_region[1][1], numGridAxis)
    z_vals = np.linspace(mic_region[2][0], mic_region[2][1], numGridAxis)
    grid_x, grid_y, grid_z = np.meshgrid(x_vals, y_vals, z_vals)
    mic_position = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)
    num_mic = mic_position.shape[0]
    print("Number of microphones: ", num_mic)

    # Room simulation
    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    room = pra.ShoeBox(room_dim, fs=fs, materials=pra.Material(e_absorption), max_order=max_order)
    room.set_sound_speed(c)
    for i in range(num_src):
        room.add_source(src_position[i,:])
    room.add_microphone_array(mic_position.T)
    room.compute_rir()
    rir = np.zeros((num_src, num_mic, irlen))
    atf = np.zeros((num_src, num_mic, fftlen//2), dtype=complex)
    atf_mag = np.zeros((num_src, num_mic, fftlen//2))
    rir_algn = np.zeros((num_src, num_mic, irlen_algn))
    atf_algn = np.zeros((num_src, num_mic, fftlen_algn//2), dtype=complex)
    atf_mag_algn = np.zeros((num_src, num_mic, fftlen_algn//2))
    for i in range(num_src):
        for j in range(num_mic):
            rir[i,j,:] = room.rir[j][i][:irlen]
            atf[i,j,:] = fft(rir[i,j,:] * window, fftlen)[1:fftlen//2+1]
            atf_mag[i,j,:] = 20*np.log10(np.abs(atf[i,j,:]))

            d_s2m = np.linalg.norm(mic_position[j,:]-src_position[i,:])
            delay_algn = int(np.round(d_s2m/c * fs - 1e-3))
            rir_algn[i,j,:] = room.rir[j][i][delay_algn:(delay_algn+irlen_algn)]
            atf_algn[i,j,:] = fft(rir_algn[i,j,:] * window_algn, fftlen_algn)[1:fftlen_algn//2+1]
            atf_mag_algn[i,j,:] = 20*np.log10(np.abs(atf_algn[i,j,:]))

    # Save data
    dir = f"ir_fs{fs}_s{num_src}_m{num_mic}_room{room_dim[0]:.1f}x{room_dim[1]:.1f}x{room_dim[2]:.1f}_rt{int(rt60_tgt*1000)}/"
    print("Save directoy: ", dir)
    
    # Create directory if it doesn't exist
    os.makedirs(dir, exist_ok=True)

    config = {"fs": fs, "c": c, "rt60_tgt": rt60_tgt, "num_src": num_src, "num_mic": num_mic, "irlen": irlen, "fftlen": fftlen, "irlen_algn": irlen_algn, "fftlen_algn": fftlen_algn, "room_dim": room_dim, "mic_dim": mic_dim, "mic_region": mic_region, "mic_interval": mic_interval, "src_region": src_region, "center": center}
    json.dump(config, open(dir+"config.json", "w"), indent=4)

    posSrc = src_position - np.tile(center, (num_src, 1))
    posMic = mic_position - np.tile(center, (num_mic, 1))
    np.savez(dir+"data.npz", rir, atf, atf_mag, rir_algn, atf_algn, atf_mag_algn, posSrc, posMic)

    # Plot impulse response of the first source and microphone
    fig, ax = plt.subplots()
    plt.plot(t, rir[0,0])
    plt.title("Impulse Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig(dir+"ir.pdf")

    fig, ax = plt.subplots()
    plt.plot(t_algn, rir_algn[0,0])
    plt.title("Impulse Response")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.savefig(dir+"ir_algn.pdf")

    fig, ax = plt.subplots()
    plt.plot(freq, atf_mag[0,0])
    ax.set_xscale('log')
    plt.title("Acoustic Transfer Function")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.savefig(dir+"atf.pdf")

    fig, ax = plt.subplots()
    plt.plot(freq_algn, atf_mag_algn[0,0])
    ax.set_xscale('log')
    plt.title("Acoustic Transfer Function")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.savefig(dir+"atf_algn.pdf")

    room_corners, room_verts = dim2cuboid(room_dim)
    target_corners, target_verts = dim2cuboid(mic_dim, np.array([mic_region[0][0],mic_region[1][0],mic_region[2][0]]))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #ax.scatter(mic_position[:, 0], mic_position[:, 1], mic_position[:, 2], s=.1, c='b', label='Evaluation')
    ax.scatter(src_position[:,0], src_position[:,1], src_position[:,2], s=10., c='r', label='Source')
    ax.add_collection3d(Poly3DCollection(room_verts, linewidths=1, edgecolors='k', alpha=0.))
    ax.add_collection3d(Poly3DCollection(target_verts, linewidths=.5, facecolors='c', edgecolors='c', alpha=.1))
    ax.legend()
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    ax.set_aspect('equal')
    # plt.savefig(dir+"geometry.pdf")
    # plt.show()