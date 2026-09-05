import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset

class PSLDataset(Dataset):
    def __init__(self, root, max_frames=16, limit=5000): # Defaulted to 16 frames for speed
        self.samples = []
        self.labels = []
        self.label_map = {}
        self.max_frames = max_frames

        self._load(root)
        
        # Shuffle everything so the slice is representative of all classes
        combined = list(zip(self.samples, self.labels))
        random.shuffle(combined)
        
        # Limit dataset size
        combined = combined[:limit]
        if combined:
            self.samples, self.labels = zip(*combined)
            self.samples = list(self.samples)
            self.labels = list(self.labels)
        else:
            self.samples, self.labels = [], []

    def _load(self, root):
        # Removed print statements for speed
        for dirpath, _, filenames in os.walk(root):
            npy_files = [f for f in filenames if f.endswith(".npy")]
            if not npy_files:
                continue

            # beard/1/file.npy -> label is 'beard'
            label = os.path.basename(os.path.dirname(dirpath))

            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)

            for f in npy_files:
                path = os.path.join(dirpath, f)
                self.samples.append(path)
                self.labels.append(self.label_map[label])

    def _read_npy(self, path):
        try:
            data = np.load(path)
            # Standard MediaPipe hand landmarks (21 points * 3 coords = 63, or 2 hands = 126)
            if data.shape[0] != 126:
                return np.zeros((2, self.max_frames, 42))
            
            data = data.reshape(42, 3)
            data = data[:, :2]  # Keep X, Y only
            
            # 1. SPATIAL CENTERING (Wrist at 0,0)
            # Subtract wrist coordinate from all points
            data = data - data[0]
            
            # 2. SHAPE TRANSFORMATION
            # (42, 2) -> (2, 42)
            data = np.transpose(data, (1, 0)) 
            # (2, 42) -> (2, 1, 42)
            data = np.expand_dims(data, axis=1) 
            
            # 3. TEMPORAL EXPANSION
            # Repeat the pose to match the time dimension expected by STGCN
            return np.repeat(data, self.max_frames, axis=1) 
            
        except Exception:
            return np.zeros((2, self.max_frames, 42))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self._read_npy(self.samples[idx])
        
        # 4. SAMPLE-LEVEL NORMALIZATION (Significant speedup)
        # Normalizing here prevents the training loop from waiting on CPU math
        mean = x.mean()
        std = x.std() + 1e-6
        x = (x - mean) / std
        
        y = self.labels[idx]
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)