import os
import numpy as np
from tqdm import tqdm

def process_npz_files():
    # Source and destination directories
    source_dir = "/home/eerdem/DATA/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200"
    dest_dir = "/home/eerdem/DATA/ir_fs2000_s8192_m1331_room4.0x6.0x3.0_rt200_freq20"
    
    # Create destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)
    
    # Process all 1024 sources (src_id from 0 to 1023, but files are numbered 1-1024)
    for src_id in tqdm(range(8192), desc="Processing NPZ files"):
        # Source file path (files are numbered from 1 to 1024)
        source_file = os.path.join(source_dir, f"data_s{src_id + 1:04d}.npz")
        dest_file = os.path.join(dest_dir, f"data_s{src_id + 1:04d}.npz")
        
        # Check if source file exists
        if not os.path.exists(source_file):
            print(f"Warning: Source file {source_file} not found, skipping...")
            continue
        
        try:
            # Load the NPZ file
            with np.load(source_file) as data:
                # Create a dictionary to hold all the data
                output_data = {}
                
                # Copy all arrays from the original file
                for key in data.files:
                    if key == 'atf_mag_algn':
                        # Slice atf_mag_algn to keep only first 20 frequency components
                        original_data = data[key]
                        sliced_data = original_data[:, :20]
                        output_data[key] = sliced_data
                        print(f"File {src_id + 1:04d}: Sliced atf_mag_algn from {original_data.shape} to {sliced_data.shape}")
                    else:
                        # Copy other data as-is
                        output_data[key] = data[key]
            
            # Save the processed data to the new file
            np.savez(dest_file, **output_data)
            
        except Exception as e:
            print(f"Error processing file {source_file}: {str(e)}")
            continue
    
    print(f"\nProcessing complete! Files saved to: {dest_dir}")

if __name__ == "__main__":
    process_npz_files()
