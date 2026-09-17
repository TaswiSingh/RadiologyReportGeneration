import torch

from models.visual_encoder import VisualEncoder


def main():

    print("Creating Visual Encoder...")

    encoder = VisualEncoder(
        d_model=512,
        pretrained=True,
        freeze_backbone=True
    )

    encoder.eval()

    print("Encoder created successfully.")

    # Create a fake batch of 2 X-rays
    images = torch.randn(
        2,
        3,
        224,
        224
    )

    print("\nInput shape:")
    print(images.shape)

    with torch.no_grad():
        features = encoder(images)

    print("\nOutput shape:")
    print(features.shape)

    print("\nExpected:")
    print("torch.Size([2, 49, 512])")

    assert features.shape == (2, 49, 512)

    print("\nVisual Encoder test PASSED!")


if __name__ == "__main__":
    main()