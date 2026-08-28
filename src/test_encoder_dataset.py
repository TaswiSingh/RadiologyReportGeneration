import torch
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset, collate_fn
from models.visual_encoder import VisualEncoder


def main():

    print("Loading chest X-ray dataset...")

    dataset = ChestXrayDataset(
        csv_path="data/train.csv",
        img_dir="images/images_normalized",
        vocab_path="data/vocab.pkl",
        max_length=128
    )

    print("Dataset size:", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    batch = next(iter(loader))

    images = batch["images"]

    print("\nReal X-ray batch shape:")
    print(images.shape)

    print("\nLoading Visual Encoder...")

    encoder = VisualEncoder(
        d_model=512,
        pretrained=True,
        freeze_backbone=True
    )

    encoder.eval()

    print("Encoder loaded.")

    with torch.no_grad():
        visual_features = encoder(images)

    print("\nVisual feature shape:")
    print(visual_features.shape)

    print("\nExpected:")
    print("torch.Size([2, 49, 512])")

    assert visual_features.shape == (2, 49, 512)

    print("\nReal X-ray Encoder Test PASSED!")


if __name__ == "__main__":
    main()