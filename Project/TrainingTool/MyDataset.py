import os
from PIL import Image
from torch.utils.data import Dataset

import pandas as pd
import torch

class CarlaData(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)

        self.img_dir = img_dir

        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        target = str(self.img_labels.iloc[idx, 0])
        padding = 8 - len(target)
        add_zero = lambda str1: "0"*padding + str1 if padding > 0 else str1
        
        img_path = os.path.join(self.img_dir, add_zero(target) + ".png")
        
        try:
            img = Image.open(img_path)
        except:
            print(img_path + "no found.")
            return None
        
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        # 目前不考慮煞車與油門
        label = torch.tensor([self.img_labels.iloc[idx, 2]]).float()

        return img, label
    
