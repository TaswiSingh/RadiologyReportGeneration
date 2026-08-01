import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ChestXrayDataset(Dataset):

    def __init__(self, csv_path, img_dir, transform=None):

        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        img_path = os.path.join(
            self.img_dir,
            row["filename"]
        )

        image = Image.open(img_path).convert("RGB")

        image = self.transform(image)

        report = row["report"]

        return {
            "image": image,
            "report": report,
            "filename": row["filename"]
        }