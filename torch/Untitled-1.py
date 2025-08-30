# %%
import torch # This is the main library for pytorch
import torch.nn as nn # This  would have the functions realted to the neural network
import torch.optim as optim # This would have the optimizing functions
from torch.utils.data import Dataset, DataLoader # This is for the first step to train a model, to load the dataset

# Specific things related to a vision based model
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
import sys
import torchvision
print('System Version: ', sys.version)
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)
print("Numpy Version: ", np.__version__)
print("Pandas Version: ", pd.__version__)

# %%
class PlayingCardDataset(Dataset): # This class inherits from the torch.utils.data
    def __init__(self, data_dir, transform=None): # Would tell what to do when created
        self.data = ImageFolder(data_dir, transform=transform) # This would make the data so that the sub folder would be the class name, and would create the labels for us
    
    def __len__(self): # The data loader would need to know the number of entries in our datasets
        return len(self.data)

    def __getitem__(self, idx): # This would indicate how is an entry of the dataset accessed by the loader.
        return self.data[idx] # This would return the image and the class

    # This enables us to find the class names easily
    @property
    def classes(self):
        return self.data.classes



# %%
dataset = PlayingCardDataset(
    data_dir="dataset/train"
)

# %%
len(dataset)

# %%
dataset[5]

# %%
image, label = dataset[6000]
image

# %%
datadir = "dataset/train"
target_to_class = {v:k for k, v in ImageFolder(datadir).class_to_idx.items()}
print(target_to_class)

# %%
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
]) 

data_dir = "dataset/train"

dataset = PlayingCardDataset(data_dir, transform)

# %%
image, label = dataset[100]
image.shape

# %%
# Can loop over a dataset like:
for image, label in dataset:
    print(image.shape) # Here is 3 is for the RGB channels, and 128 is because of use resizing it to 128 * 128

# %%
# However this loop would take things one at a time, to create a prallelized data loader we would:
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# It parallelizes to batch size of 32 with suffling, however shuffling isn't required for train and valdation datasets

# %%
for images, labels in dataloader:
    print("Image Shape:", images.shape) # The size becomes [32, 3, 128, 128] from [3, 128, 128] so that we would use 32 pics for processing parallely
    print("Label Shape:", labels.shape) # This also becomes 32x
    break

# %%
class SimpleCardClassifier(nn.Module):
    def __init__(self, num_classes=53):
        super(SimpleCardClassifier, self).__init__()
        
        # Load the pretrained model
        self.base_model = timm.create_model("efficientnet_b0", pretrained=True)
        
        # Get the number of input features for the classifier
        enet_out_size = self.base_model.get_classifier().in_features # More robust way
        
        # Replace the classifier with a new one for our task
        # timm has a reset_classifier method, or you can do it manually
        self.base_model.classifier = nn.Linear(enet_out_size, num_classes)

    def forward(self, x):
        # The forward pass is now simpler, just call the whole model
        return self.base_model(x)

# %%
model = SimpleCardClassifier(num_classes=53).to(device="mps")

# %%
print(str(model)[:500])

# %%
model.to(device="cpu")
images.to(device="cpu")
model(images) # The for loop had set one batch into images at the end of the loop

# %%
model(images).shape

# %%
# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %%
criterion((model(images)), labels)

# %%
# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # 👇 CRITICAL: Add this normalization step
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dataset folders
train_folder = 'dataset/train'
valid_folder = 'dataset/valid'
# test_folder = 'dataset/test'  # Uncomment when you need it

# Load datasets
train_dataset = PlayingCardDataset(train_folder, transform=transform)
valid_dataset = PlayingCardDataset(valid_folder, transform=transform)
# test_dataset = PlayingCardDataset(test_folder, transform=transform)  # Uncomment when needed

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)  # Uncomment when needed

# %%
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

from tqdm.notebook import tqdm

# %%
# Number of epochs
num_epoch = 5
train_losses, valid_losses = list(), list()

# Initialize model
model = SimpleCardClassifier(num_classes=53)
model.to(device)
model = model.to("mps", dtype=torch.float32)

# Training loop with tqdm progress bars
for epoch in range(num_epoch):
    # Training phase
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epoch} [Training]", leave=False)
    for images, labels in train_loader_tqdm:
        images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        
        optimizer.zero_grad()  # ✅ Fixed: Added parentheses
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        train_loader_tqdm.set_postfix({"Batch Loss": loss.item()})
    
    train_loss = running_loss / len(train_loader.dataset)
    train_losses.append(train_loss)

    # Validation phase
    model.eval()
    running_loss = 0.0
    valid_loader_tqdm = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epoch} [Validation]", leave=False)
    with torch.no_grad():
        for images, labels in valid_loader_tqdm:
            images, labels = images.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            valid_loader_tqdm.set_postfix({"Batch Loss": loss.item()})
    
    val_loss = running_loss / len(valid_loader.dataset)
    valid_losses.append(val_loss)

    # Epoch summary
    print(f"Epoch {epoch + 1}/{num_epoch} - Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# %%



