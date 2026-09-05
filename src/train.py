import torch
from torch.utils.data import DataLoader, random_split
from src.data_loader import PSLDataset
from src.model import STGCN
import numpy as np
import json
import os

def train(data_dir):
    # 1. Load Data
    # Using max_frames=16 and limit=5000 for a good balance of speed and accuracy
    dataset = PSLDataset(data_dir, max_frames=16, limit=None) 

# In the DataLoader

    # Save label map so you can use it for inference later
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(dataset.label_map, f, ensure_ascii=False)

    # Train/Val Split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    # Increased batch size (32) helps stability and speed
    # num_workers=0 is safer for Windows to avoid 'hanged' processes
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    # 2. Setup Model & Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = STGCN(num_classes=len(dataset.label_map)).to(device)

    # 3. Balanced Loss Function
    class_counts = np.bincount(dataset.labels)
    weights = 1. / (class_counts + 1e-6)
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    
    # Optimizer & Learning Rate Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

    best_acc = 0
    print(f"Dataset size: {len(dataset)} | Classes: {len(dataset.label_map)}")
    print("Starting training...")

    for epoch in range(50): # Increased epochs as scheduler will handle slowdowns
        # ===== TRAIN =====
        model.train()
        total_loss = 0
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            # NOTE: Normalization is now handled inside PSLDataset for speed!

            optimizer.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = torch.argmax(out, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        avg_train_loss = total_loss / len(train_loader)
        train_acc = 100 * correct / total

        # ===== VALIDATION =====
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                
                out = model(x)
                pred = torch.argmax(out, dim=1)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)

        val_acc = 100 * val_correct / val_total
        
        # 4. Step the Scheduler
        scheduler.step(val_acc)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:03d} | Loss: {avg_train_loss:.4f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | LR: {current_lr:.6f}")

        # 5. Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "stgcn_best_model.pth")
            print(f"--> New Best Model Saved (Acc: {best_acc:.2f}%)")

    print(f"\nTraining Complete. Best Validation Accuracy: {best_acc:.2f}%")
