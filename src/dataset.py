import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from tokenizer import ReportTokenizer


class ChestXrayDataset(Dataset):

    def __init__(
        self,
        csv_path,
        img_dir,
        vocab_path,
        transform=None,
        max_length=128
    ):

        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.tokenizer = ReportTokenizer(vocab_path)
        self.max_length = max_length

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

        if not os.path.exists(img_path):
            raise FileNotFoundError(
                f"Image not found: {img_path}"
            )

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        report = str(row["report"])

        token_ids = self.tokenizer.encode(report)
        if len(token_ids) < 2:
            raise ValueError(
               f"Invalid token sequence at index {idx}: {report}"
            )

        eos_id = self.tokenizer.vocab.word2idx["<eos>"]

        if len(token_ids) > self.max_length:
            token_ids = (
                token_ids[:self.max_length - 1]
                + [eos_id]
            )

        input_ids = torch.tensor(
            token_ids,
            dtype=torch.long
        )

        return {
            "image": image,
            "input_ids": input_ids,
            "report": report,
            "filename": row["filename"]
        }


def collate_fn(batch):

    images = torch.stack([
        item["image"]
        for item in batch
    ])

    sequences = [
        item["input_ids"]
        for item in batch
    ]

    pad_id = 0

    max_length = max(
        len(seq)
        for seq in sequences
    )

    input_ids = torch.full(
        (len(sequences), max_length),
        pad_id,
        dtype=torch.long
    )

    attention_mask = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.bool
    )

    for i, seq in enumerate(sequences):

        length = len(seq)

        input_ids[i, :length] = seq
        attention_mask[i, :length] = True

    return {
        "images": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "filenames": [
            item["filename"]
            for item in batch
        ],
        "reports": [
            item["report"]
            for item in batch
        ]
    }