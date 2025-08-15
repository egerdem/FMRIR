import os
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

results_dir = "/Users/ege/Projects/FMRIR/artifacts"

# Navigate in the directories under results_dir and load all file names starting with "model", don't traverse the nested folders, only do first level
def find_model_files(results_dir):
    """Find all model files starting with 'model' in first-level subdirectories"""
    model_files = []
    
    # Get all subdirectories in results_dir
    for item in os.listdir(results_dir):
        item_path = os.path.join(results_dir, item)
        
        # Check if it's a directory
        if os.path.isdir(item_path):
            # Look for files starting with "model" in this directory
            for file in os.listdir(item_path):
                if file.startswith("model") and file.endswith(".pt"):
                    full_path = os.path.join(item_path, file)
                    model_files.append(full_path)
    
    return model_files

# Get all model files
available_models = find_model_files(results_dir)

print("Found model files:")

all = []

available_models = ["/Users/ege/Projects/FMRIR/experiments/None_20250815-140723_iter4/checkpoints/model_4.pt"]
for i, model_path in enumerate(available_models):

    # You can choose which model to load by index
    # For now, let's use the first one as an example
    MODEL_LOAD_PATH = available_models[i]  # Change index to select different model

    # Load the checkpoint
    try:
        checkpoint = torch.load(MODEL_LOAD_PATH, map_location=device)
        all.append(checkpoint)
        print(f"\n{i}: model: {MODEL_LOAD_PATH[35:]}")
        # print(f"Checkpoint keys: {list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'Not a dictionary'}")

        try:
            it = checkpoint["config"]["training"].get("num_iterations")
            print(f"Total iter: {it}")
            print(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A')} at iteration {checkpoint.get('best_iteration', 'N/A')}")
        except (KeyError, AttributeError) as e:
            print(f"Could not read training info: {e}")
            
    except Exception as e:
        print(f"Error loading model {MODEL_LOAD_PATH}: {e}")
        continue

