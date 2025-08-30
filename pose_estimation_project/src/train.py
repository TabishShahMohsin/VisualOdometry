import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import PoseNet
from dataset import PoseDataset
from utils import total_supervised_loss
import os
from tqdm import tqdm
import numpy as np

# ==============================================================================
#                             CONFIGURATION
# ==============================================================================

BACKBONE = 'mobilenet_v2'
LEARNING_RATE = 1e-4
BATCH_SIZE = 16 # Reduced for lower memory usage
NUM_EPOCHS = 10

DATASET_DIR = "dataset"
IMAGE_DIR = os.path.join(DATASET_DIR, "images")
CSV_FILE = os.path.join(DATASET_DIR, "labels.csv")

MODEL_SAVE_DIR = "models"
MODEL_SAVE_NAME = f"posenet_{BACKBONE}_best.pth"

# ==============================================================================
#                               MAIN SCRIPT
# ==============================================================================

def main():
    # --- Setup ---
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    model_save_path = os.path.join(MODEL_SAVE_DIR, MODEL_SAVE_NAME)

    # --- Data Loading ---
    print("Loading dataset...")
    full_dataset = PoseDataset(csv_file=CSV_FILE, image_dir=IMAGE_DIR, use_sobel=True)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0) # num_workers=0 for macOS
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0) # num_workers=0 for macOS
    
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    # --- Model, Optimizer, and Loss ---
    print(f"Initializing model with backbone: {BACKBONE}...")
    model = PoseNet(backbone=BACKBONE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = total_supervised_loss

    # --- Training Loop ---
    best_val_loss = float('inf')
    print("\nStarting training...")

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Train]")
        
        for batch in train_loop:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            pred_periods, pred_offsets = model(images)
            loss = criterion(pred_periods, pred_offsets, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS} [Val  ]")
        
        with torch.no_grad():
            for batch in val_loop:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                
                pred_periods, pred_offsets = model(images)
                loss = criterion(pred_periods, pred_offsets, labels)
                val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch Summary: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"   -> New best model saved to {model_save_path} (Val Loss: {best_val_loss:.6f})")

    print("\nTraining finished!")
    print(f"Best model saved at {model_save_path} with validation loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    main()