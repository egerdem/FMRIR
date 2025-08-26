import torch
from collections import OrderedDict

OLD_CKPT_PATH = '/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_20250826-183304_iter200000/checkpoints/ckpt_200000.pt'
NEW_CKPT_PATH = '/Users/ege/Projects/FMRIR/artifacts/ATF3D-CrossAttn-v1-freq20_M5to50_sigmaE3_20250826-183304_iter200000/checkpoints/ckpt_200000_CONV.pt'

print(f"Loading old checkpoint from: {OLD_CKPT_PATH}")
checkpoint = torch.load(OLD_CKPT_PATH, map_location='cpu')

# Assuming the UNet state is nested as we designed
old_unet_state = checkpoint['model_states']['unet']
new_unet_state = OrderedDict()

# --- Manually map the old layer names to the new dynamic names ---
key_map = {
    # Encoder mapping (this was correct)
    'down1': 'encoders.0',
    'attn1': 'encoder_attns.0',
    'down2': 'encoders.1',
    'attn2': 'encoder_attns.1',
    # Add more if your static model was deeper (e.g., 'down3': 'encoders.2')

    # --- CORRECTED DECODER MAPPING ---
    'up1_trans': 'decoders.0.up_conv',  # up1 is the first decoder module (index 0)
    'up1_conv':  'decoders.0.conv',
    'up2_trans': 'decoders.1.up_conv',  # up2 is the second decoder module (index 1)
    'up2_conv':  'decoders.1.conv',
    # Add more if your static model was deeper (e.g., 'up3_trans': 'decoders.2.up_conv')
}

# --- Translation Logic (No changes needed here) ---
translated_keys = set()

# First, copy over any weights that do not need renaming (e.g., init_conv, bottleneck)
for key, value in old_unet_state.items():
    if not any(prefix in key for prefix in key_map.keys()):
        new_unet_state[key] = value
        translated_keys.add(key)

# Second, translate the mapped keys
for old_prefix, new_prefix in key_map.items():
    for key, value in old_unet_state.items():
        if key.startswith(old_prefix):
            new_key = key.replace(old_prefix, new_prefix, 1)
            new_unet_state[new_key] = value
            translated_keys.add(key)

# --- Finalization ---
# Update the checkpoint's state dict with the new, correctly named keys
checkpoint['model_states']['unet'] = new_unet_state

# Verify all old keys were translated
if len(translated_keys) != len(old_unet_state):
    print("WARNING: Some keys in the old state dict were not translated!")
    untranslated = set(old_unet_state.keys()) - translated_keys
    print(f"Untranslated keys: {untranslated}")

torch.save(checkpoint, NEW_CKPT_PATH)
print(f"\nSuccessfully saved converted checkpoint to: {NEW_CKPT_PATH}")