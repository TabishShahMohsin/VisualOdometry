import torch
from torch.utils.data import Dataset
import pandas as pd
import cv2
import os
from torchvision import transforms
import numpy as np

class PeriodDataset(Dataset):
    """
    Custom PyTorch Dataset for loading synthetic pose estimation data.
    This version is designed to predict the tile periods (t_x, t_y) in pixels.
    """
    def __init__(self, csv_file, image_dir, transform=None, use_sobel=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied.
            use_sobel (bool): Whether to apply a Sobel filter as a high-pass filter.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.use_sobel = use_sobel
        
        if transform is None:
            # Default transformations for grayscale images
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]) 
            ])
        else:
            self.transform = transform

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.labels_df)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        Returns a dictionary containing the image and the ground truth periods.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        row = self.labels_df.iloc[idx]
        img_name = os.path.join(self.image_dir, row['image_name'])
        image = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise FileNotFoundError(f"Image not found at {img_name}")

        if self.use_sobel:
            sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            image = np.hypot(sobelx, sobely)
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if self.transform:
            image = self.transform(image)

        # Extract parameters to calculate ground truth periods
        tile_w = row['tile_w']
        tile_h = row['tile_h']
        fx = row['fx']
        fy = row['fy']
        tz = row['tz']

        # Calculate period in pixels.
        t_x_px = (tile_w / tz) * fx
        t_y_px = (tile_h / tz) * fy
        
        label = torch.tensor([t_x_px, t_y_px], dtype=torch.float32)

        sample = {'image': image, 'label': label}
        return sample