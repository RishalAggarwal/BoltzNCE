import os
import numpy as np

def collate_npy_files(input_dir, output_path):
    combined_dict = {}
    total_files = 0

    for filename in os.listdir(input_dir):
        if filename.endswith(".npy"):
            filepath = os.path.join(input_dir, filename)
            try:
                data = np.load(filepath, allow_pickle=True).item()
                if isinstance(data, dict):
                    total_files += 1
                    for key, value in data.items():
                        if key in combined_dict:
                            combined_dict[key] = np.concatenate((combined_dict[key], value), axis=0)
                        else:
                            combined_dict[key] = value
                else:
                    print(f"Skipped {filename}: not a dict.")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

  
    np.save(output_path, combined_dict)
    print(f"\n✅ Combined data saved to: {output_path}")
    print(f"🔍 Total files processed: {total_files}")
    print(f"📦 Peptides found: {list(combined_dict.keys())}")

    # Print a couple of quick checks
    for i, (key, value) in enumerate(combined_dict.items()):
        print(f"\n🔬 Peptide '{key}': shape {value.shape}, dtype {value.dtype}")
        print(f"First 1-2 entries:\n{value[:2]}")
        if i >= 1:
            break  # show only first 2 peptides



if __name__ == "__main__":
    input_directory = "../generated/correctedbias"  # change this if your directory is different
    output_file = "../data/2AA-1-large/all_gen_correctedbias.npy"
    collate_npy_files(input_directory, output_file)
