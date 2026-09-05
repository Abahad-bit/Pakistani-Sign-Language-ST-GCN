import os

dataset_path = "dataset/PakistanSignLanguageDataset"
total_files = 0

for root, dirs, files in os.walk(dataset_path):
    # Only count the .npy files
    total_files += len([f for f in files if f.endswith('.npy')])

print(f"Total .npy samples found: {total_files}")