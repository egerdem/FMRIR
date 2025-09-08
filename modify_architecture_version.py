#!/usr/bin/env python3
"""
Script to modify the architecture_version in both model.pt and config.json
from "v1_legacy" to "v2_residual_context"
"""

import torch
import json
import os
from pathlib import Path

# Model path for architecture_version modification
MODEL_LOAD_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet4_layer6_20250906-191258_iter300000/model.pt"
CONFIG_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma0_lrWARM5k_e4_toe5_unet4_layer6_20250906-191258_iter300000/config.json"

# Model path for setencoder_version modification
SETENCODER_MODEL_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma1e3_lrWARM5k_e4_toe5_unet4_setv3_20250908-151143_iter300000/model.pt"
SETENCODER_CONFIG_PATH = "/Users/ege/Projects/FMRIR/artifacts/M5to50_freq20_layer3_d512_head8_sigma1e3_lrWARM5k_e4_toe5_unet4_setv3_20250908-151143_iter300000/config.json"

def load_model_and_config(model_path, device):
    """Load model checkpoint and extract configuration"""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get('config', {})
    model_states_cfg = checkpoint['model_states']
    return checkpoint, config, model_states_cfg

def modify_architecture_version():
    """Modify architecture_version in both model.pt and config.json"""
    device = torch.device('cpu')  # Use CPU for loading
    
    print("Loading model checkpoint...")
    # Load the model checkpoint
    checkpoint, config, model_states_cfg = load_model_and_config(MODEL_LOAD_PATH, device)
    
    # Get model config from checkpoint
    model_cfg = config.get('model', {})
    
    print(f"Current architecture_version: {model_cfg.get('architecture_version', 'Not found')}")
    
    # Modify the architecture_version
    model_cfg['architecture_version'] = "v2_residual_context"
    
    print(f"Updated architecture_version to: {model_cfg['architecture_version']}")
    
    # Update the checkpoint with modified config
    checkpoint['config']['model'] = model_cfg
    
    # Save the modified model checkpoint
    print("Saving modified model checkpoint...")
    torch.save(checkpoint, MODEL_LOAD_PATH)
    print(f"Model saved to: {MODEL_LOAD_PATH}")
    
    # Now modify the config.json file
    print("Loading config.json...")
    with open(CONFIG_PATH, 'r') as f:
        json_config = json.load(f)
    
    print(f"Current config.json architecture_version: {json_config['model'].get('architecture_version', 'Not found')}")
    
    # Modify the architecture_version in config.json
    json_config['model']['architecture_version'] = "v2_residual_context"
    
    print(f"Updated config.json architecture_version to: {json_config['model']['architecture_version']}")
    
    # Save the modified config.json
    print("Saving modified config.json...")
    with open(CONFIG_PATH, 'w') as f:
        json.dump(json_config, f, indent=4)
    
    print(f"Config.json saved to: {CONFIG_PATH}")
    print("\nModification completed successfully!")

def add_setencoder_version():
    """Add setencoder_version = "v3" to both model.pt and config.json"""
    device = torch.device('cpu')  # Use CPU for loading
    
    print("Loading model checkpoint...")
    # Load the model checkpoint
    checkpoint, config, model_states_cfg = load_model_and_config(SETENCODER_MODEL_PATH, device)
    
    # Get model config from checkpoint
    model_cfg = config.get('model', {})
    
    print(f"Current setencoder_version: {model_cfg.get('setencoder_version', 'Not found')}")
    
    # Add the setencoder_version
    model_cfg['setencoder_version'] = "v3"
    
    print(f"Added setencoder_version: {model_cfg['setencoder_version']}")
    
    # Update the checkpoint with modified config
    checkpoint['config']['model'] = model_cfg
    
    # Save the modified model checkpoint
    print("Saving modified model checkpoint...")
    torch.save(checkpoint, SETENCODER_MODEL_PATH)
    print(f"Model saved to: {SETENCODER_MODEL_PATH}")
    
    # Now modify the config.json file
    print("Loading config.json...")
    with open(SETENCODER_CONFIG_PATH, 'r') as f:
        json_config = json.load(f)
    
    print(f"Current config.json setencoder_version: {json_config['model'].get('setencoder_version', 'Not found')}")
    
    # Add the setencoder_version to config.json
    json_config['model']['setencoder_version'] = "v3"
    
    print(f"Added config.json setencoder_version: {json_config['model']['setencoder_version']}")
    
    # Save the modified config.json
    print("Saving modified config.json...")
    with open(SETENCODER_CONFIG_PATH, 'w') as f:
        json.dump(json_config, f, indent=4)
    
    print(f"Config.json saved to: {SETENCODER_CONFIG_PATH}")
    print("\nSetencoder version addition completed successfully!")

if __name__ == "__main__":
    print("Choose an operation:")
    print("1. Modify architecture_version to v2_residual_context")
    print("2. Add setencoder_version = v3")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Check if files exist for architecture_version modification
        if not os.path.exists(MODEL_LOAD_PATH):
            print(f"Error: Model file not found at {MODEL_LOAD_PATH}")
            exit(1)
        
        if not os.path.exists(CONFIG_PATH):
            print(f"Error: Config file not found at {CONFIG_PATH}")
            exit(1)
        
        # Run the architecture_version modification
        modify_architecture_version()
        
    elif choice == "2":
        # Check if files exist for setencoder_version addition
        if not os.path.exists(SETENCODER_MODEL_PATH):
            print(f"Error: Model file not found at {SETENCODER_MODEL_PATH}")
            exit(1)
        
        if not os.path.exists(SETENCODER_CONFIG_PATH):
            print(f"Error: Config file not found at {SETENCODER_CONFIG_PATH}")
            exit(1)
        
        # Run the setencoder_version addition
        add_setencoder_version()
        
    else:
        print("Invalid choice. Please run the script again and choose 1 or 2.")
        exit(1)
