import matplotlib

matplotlib.use('Qt5Agg', force=True)  # or 'TkAgg'
from matplotlib import pyplot as plt
import torch
import os
import numpy as np
import random
from fm_utils import (ATF3DSampler, CFGVectorFieldODE_3D, EulerSimulator,
                      SetEncoder, CrossAttentionUNet3D)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 42  # You can use any integer you like
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)  # for GPU
np.random.seed(SEED)
random.seed(SEED)

# def main():
# --- Universal Setup ---
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq64_M5to50_20250825-184335_iter200000/model.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_20250825-201433_iter200000/model.pt"
MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_20250826-183304_iter200000/model_CONVoldcheckpoint.pt"
# MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_20250825-201433_iter200000/modelCONVoldcheckpoint.pt"
MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_UNET256_20250826-192413_iter200000/model.pt"

MODEL_NAME = MODEL_LOAD_PATH.split("artifacts/")[1].split("/")[0]

print(f"Model artifact: {MODEL_NAME}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)

config = checkpoint.get('config', {})  # Use .get for safety
training_params = config.get('training', {})
sigma_train = training_params.get('sigma')

print("\n--- Automatically Configured from Loaded Model ---")
print(f"  Training Sigma: {sigma_train:.4f}")
print("--------------------------------------------------\n")

data_dir = config['data']['data_dir']
# override data_dir with local
data_dir = "ir_fs2000_s1024_m1331_room4.0x6.0x3.0_rt200/"

src_split = config['data']['src_splits']

model_mode = config["training"].get('model_mode', "spatial")
freq_up_to = config['model'].get('freq_up_to')

model_cfg = config.get('model', {})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
config = checkpoint.get('config', {})
model_states_cfg = checkpoint['model_states']

# --- KEY CHANGE: Detect Model Type ---
# Check if the checkpoint has keys associated with the 3D model
is_3d_model = 'set_encoder' in model_states_cfg

if is_3d_model:
    print("--- Detected 3D Conditional Generation Model ---")

    # --- CONFIGURATION FOR VISUALIZATION ---
    PLOT_3D = True  # Set to True to generate a separate 3D scatter plot
    data_config_path = os.path.join(data_dir, "config.json")
    with open(data_config_path, 'r') as f:
        data_config = json.load(f)
    room_dim = data_config.get('room_dim')
    center = data_config.get('center')


    # Helper function to plot a 3D box
    def plot_room_box(ax, dimensions):
        w, d, h = dimensions  # width, depth, height
        # Define the 8 corners of the box
        corners = [
            [0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0],
            [0, 0, h], [w, 0, h], [w, d, h], [0, d, h]
        ]
        corners = np.array(corners)
        # Define the 6 faces of the box
        faces = [
            [corners[0], corners[1], corners[5], corners[4]],  # Front
            [corners[2], corners[3], corners[7], corners[6]],  # Back
            [corners[0], corners[3], corners[7], corners[4]],  # Left
            [corners[1], corners[2], corners[6], corners[5]],  # Right
            [corners[0], corners[1], corners[2], corners[3]],  # Bottom
            [corners[4], corners[5], corners[6], corners[7]]  # Top
        ]
        # Create and add the 3D polygon collection
        ax.add_collection3d(Poly3DCollection(
            faces, facecolors='cyan', linewidths=1, edgecolors='darkblue', alpha=0.05
        ))
        # Set axis limits
        ax.set_xlim(0, w);
        ax.set_ylim(0, d);
        ax.set_zlim(0, h)


    # --- 1. Data Loading ---
    # Create train sampler to get normalization stats and grid coordinates
    train_sampler = ATF3DSampler(
        data_path=data_dir, mode='train', src_splits=src_split, normalize=True, freq_up_to=freq_up_to
    )
    # Create test sampler with raw data
    test_sampler = ATF3DSampler(
        data_path=data_dir, mode='test', src_splits=src_split, normalize=False, freq_up_to=freq_up_to
    )
    # Normalize the test data using the stats from the training set
    test_sampler.cubes = (test_sampler.cubes - train_sampler.mean) / (train_sampler.std + 1e-8)

    grid_xyz = train_sampler.grid_xyz.to(device)
    spec_mean = train_sampler.mean.item()
    spec_std = train_sampler.std.item()

    print(f"Loaded Stats from 3D Training Set: Mean={spec_mean:.4f}, Std={spec_std:.4f}")

    # --- 2. Re-create Models ---
    model_cfg = config['model']
    set_encoder = SetEncoder(
        num_freqs=train_sampler.cubes.shape[1],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_layers=model_cfg['num_encoder_layers']
    ).to(device)

    unet_3d = CrossAttentionUNet3D(
        in_channels=train_sampler.cubes.shape[1],
        out_channels=train_sampler.cubes.shape[1],
        channels=model_cfg['channels'],
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead']
    ).to(device)

    # --- 3. Load Weights ---
    set_encoder.load_state_dict(model_states_cfg['set_encoder'])
    unet_3d.load_state_dict(model_states_cfg['unet'])
    set_encoder.eval()
    unet_3d.eval()
    print("--- Loaded 3D U-Net and SetEncoder models for inference ---")

    # --- 4. Inference & Visualization ---
    M_range = config['training'].get('M_range')
    M_range = [5, 15]
    num_examples = 5
    num_timesteps = 10
    guidance_scales = [1.0, 2.0, 3.0, 5]
    freq_idx_to_plot = 10  # Pick a frequency channel to visualize
    z_slice_idx_to_plot = 5

    # instance of your the ODE wrapper and the simulator
    ode_3d = CFGVectorFieldODE_3D(unet=unet_3d, set_encoder=set_encoder)
    simulator = EulerSimulator(ode=ode_3d)
    # --- SETUP THE PLOT GRID ---
    # Add an extra column at the far left for a 3D snapshot view
    num_cols = 3 + len(guidance_scales)
    fig, axes = plt.subplots(num_examples, num_cols, figsize=(4.5 * num_cols, 4 * num_examples), squeeze=False)
    fig.suptitle(
        f"3D Conditional Generation (Freq Idx={freq_idx_to_plot}, Z-Slice={z_slice_idx_to_plot}) | {MODEL_NAME}",
        fontsize=16)

    center_np = np.array(center)

    for row in range(num_examples):
        # Get a random ground truth sample
        z_true, src_xyz = test_sampler.sample(1)
        z_true, src_xyz = z_true.to(device), src_xyz.to(device)

        # --- Create a sparse observation set on the fly ---
        M = torch.randint(M_range[0], M_range[1] + 1, (1,)).item()
        obs_indices = torch.randperm(grid_xyz.shape[0])[:M]
        obs_xyz_abs = grid_xyz[obs_indices]
        obs_coords_rel = obs_xyz_abs - src_xyz

        z_flat = z_true.view(z_true.shape[1], -1)
        obs_values = z_flat[:, obs_indices].transpose(0, 1)

        # Batchify for the set encoder
        obs_coords_rel = obs_coords_rel.unsqueeze(0)
        obs_values = obs_values.unsqueeze(0)
        obs_mask = torch.ones(1, M, dtype=torch.bool, device=device)

        # --- Plot Ground Truth and Sparse Input ---
        z_true_denorm = (z_true * spec_std + spec_mean)
        gt_cube_to_plot = z_true_denorm[0, freq_idx_to_plot].cpu().numpy()  # This is the (11, 11, 11) cube

        gt_slice = gt_cube_to_plot[z_slice_idx_to_plot, :, :]  # Select the specific slice

        axes[row, 1].imshow(gt_slice, origin='lower', cmap='viridis', vmin=gt_slice.min(), vmax=gt_slice.max())
        axes[row, 1].set_title("True (Z-projection)" if row == 0 else "")
        axes[row, 1].axis('off')

        # --- NEW: Plot Sparse Input (2D Scatter Projection) ---
        ax_scatter = axes[row, 2]
        obs_xyz_plot = obs_xyz_abs.cpu().numpy()
        # Plot X vs Y, and use Z for the color
        sc = ax_scatter.scatter(obs_xyz_plot[:, 0], obs_xyz_plot[:, 1], c=obs_xyz_plot[:, 2], cmap='coolwarm', s=20,
                                vmin=-0.5, vmax=0.5)
        ax_scatter.set_title(f"Input Mics" if row == 0 else "")
        ax_scatter.set_aspect('equal', adjustable='box')
        ax_scatter.set_xlim(-0.6, 0.6);
        ax_scatter.set_ylim(-0.6, 0.6)  # Example limits
        ax_scatter.set_xticks([]);
        ax_scatter.set_yticks([])

        cbar_z = fig.colorbar(sc, ax=ax_scatter, fraction=0.046, pad=0.04)
        cbar_z.set_label('Z-height (m)', size=8)
        cbar_z.ax.tick_params(labelsize=7)

        # Show per-row microphone count BETWEEN GT and scatter columns
        pos_gt = axes[row, 1].get_position(fig)
        pos_sc = axes[row, 2].get_position(fig)
        x_mid_M = (pos_gt.x1 + pos_sc.x0) * 0.5
        y_mid_row = (pos_gt.y0 + pos_gt.y1) * 0.5
        fig.text(x_mid_M - 0.03, y_mid_row, f"M={M}", ha='center', va='center', fontsize=9)

        # Annotate source coordinates between 3D and GT columns (no matching needed)
        # src_xyz_global = (src_xyz.cpu().numpy() + center_np)[0]
        # src_label = f"Src: ({src_xyz_global[0]:.2f}, {src_xyz_global[1]:.2f}, {src_xyz_global[2]:.2f})"
        # pos_3d = axes[row, 0].get_position(fig)
        # x_mid_src = (pos_3d.x1 + pos_gt.x0) * 0.5
        # fig.text(x_mid_src-0.060, y_mid_row, src_label, ha='center', va='center', fontsize=8)

        gs = axes[row, 0].get_gridspec()
        axes[row, 0].remove()
        ax3d_inline = fig.add_subplot(gs[row, 0], projection='3d')

        # Room box
        plot_room_box(ax3d_inline, room_dim)

        # Global positions for plotting
        obs_xyz_global = obs_xyz_abs.cpu().numpy() + center_np
        src_xyz_global = src_xyz.cpu().numpy() + center_np

        # Scatter microphones and source
        ax3d_inline.scatter(
            obs_xyz_global[:, 0], obs_xyz_global[:, 1], obs_xyz_global[:, 2],
            s=20, c='b'
        )
        ax3d_inline.scatter(
            src_xyz_global[0, 0], src_xyz_global[0, 1], src_xyz_global[0, 2],
            s=60, c='r', marker='*'
        )

        # Labels and limits
        ax3d_inline.set_xlabel('X (m)');
        ax3d_inline.set_ylabel('Y (m)');
        ax3d_inline.set_zlabel('Z (m)')
        ax3d_inline.set_title('Room (3D)' if row == 0 else '')

        # --- Generate for each guidance scale ---

        for g_idx, w in enumerate(guidance_scales):

            # Start from pure noise
            x0 = torch.randn_like(z_true)
            xt = x0.clone()  # The simulation starts from x0
            # Get conditioning tokens
            y_tokens, _ = set_encoder(obs_coords_rel, obs_values, obs_mask)

            ts = torch.linspace(0, 1, num_timesteps + 1, device=device)
            ts = ts.view(1, -1, 1, 1, 1, 1).expand(xt.shape[0], -1, -1, -1, -1, -1)

            # Set the guidance scale on the ODE object
            simulator.ode.guidance_scale = w

            # Simulation loop
            x1_recon = simulator.simulate(xt,
                                          ts,
                                          x0=x0,
                                          z_true=z_true,
                                          y_tokens=y_tokens,
                                          obs_mask=obs_mask,
                                          paste_observations=False,
                                          obs_indices=obs_indices
                                          )

            # De-normalize and plot
            x1_recon_denorm = (x1_recon * spec_std + spec_mean)
            recon_cube_to_plot = x1_recon_denorm[0, freq_idx_to_plot].detach().cpu().numpy()
            mse = torch.mean((x1_recon_denorm - z_true_denorm) ** 2).item()
            print(f"MSE: {mse:.4f}")

            col_idx = g_idx + 3

            recon_slice = recon_cube_to_plot[z_slice_idx_to_plot, :, :]  # Select the same slice
            im = axes[row, col_idx].imshow(recon_slice, origin='lower', cmap='viridis', vmin=gt_slice.min(),
                                           vmax=gt_slice.max())

            axes[row, col_idx].set_title(f"w={w}" if row == 0 else "")
            axes[row, col_idx].axis('off')

        # Shared colorbar for GT and generated columns (exclude scatter input)
        ax_list = [axes[row, 1]] + [axes[row, i + 3] for i in range(len(guidance_scales))]
        mappable = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=gt_slice.min(), vmax=gt_slice.max()), cmap='viridis'
        )
        cbar_mag = fig.colorbar(mappable, ax=ax_list, fraction=0.046, pad=0.04)
        cbar_mag.set_label('Magnitude (dB)', size=8)
        cbar_mag.ax.tick_params(labelsize=7)

    # plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=0.5, w_pad=1.5)
    plt.show()

    model_dir = os.path.dirname(MODEL_LOAD_PATH)
    outfile_name = f"{model_mode}_finf{freq_idx_to_plot}_z{z_slice_idx_to_plot}_{M_range[0]}to{M_range[1]}.png"

    if os.path.exists(os.path.join("artifacts", outfile_name)):
        # rand = np.random.randint(1000)
        rand = 5
        outfile_name = f"{model_mode}_finf{freq_idx_to_plot}_z{z_slice_idx_to_plot}_{M_range[0]}to{M_range[1]}_{rand}.png"

    save_path = os.path.join(model_dir, outfile_name)
    print(f"Saving figure to: {save_path}")
    fig.savefig(save_path, dpi=200, bbox_inches='tight')

# if __name__ == '__main__':
#     main()
