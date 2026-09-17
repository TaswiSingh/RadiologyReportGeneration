import pickle
import torch
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset, collate_fn
from models.visual_encoder import VisualEncoder
from models.text_decoder import TextDecoder


def main():

    print("Loading vocabulary...")

    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab.word2idx)

    print("Vocabulary size:", vocab_size)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

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
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    print("\nImages:")
    print(images.shape)

    print("\nInput IDs:")
    print(input_ids.shape)

    print("\nAttention mask:")
    print(attention_mask.shape)

    # --------------------------------------------------
    # Create visual encoder
    # --------------------------------------------------

    print("\nCreating Visual Encoder...")

    encoder = VisualEncoder(
        d_model=512,
        pretrained=True,
        freeze_backbone=True
    )

    encoder.eval()

    # --------------------------------------------------
    # Create text decoder
    # --------------------------------------------------

    print("Creating Text Decoder...")

    decoder = TextDecoder(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_length=128
    )

    decoder.eval()

    # --------------------------------------------------
    # Image → Visual features
    # --------------------------------------------------

    print("\nRunning Visual Encoder...")

    with torch.no_grad():

        visual_features = encoder(images)

    print("Visual features:")
    print(visual_features.shape)

    # --------------------------------------------------
    # Visual features + text → decoder
    # --------------------------------------------------

    print("\nRunning Text Decoder...")

    with torch.no_grad():

        logits = decoder(
            input_ids=input_ids,
            memory=visual_features,
            attention_mask=attention_mask
        )

    print("Decoder output:")
    print(logits.shape)

    # --------------------------------------------------
    # Verify shapes
    # --------------------------------------------------

    batch_size = images.size(0)
    sequence_length = input_ids.size(1)

    assert visual_features.shape == (
        batch_size,
        49,
        512
    )

    assert logits.shape == (
        batch_size,
        sequence_length,
        vocab_size
    )

    print("\n================================")
    print("FULL MODEL TEST PASSED!")
    print("================================")


if __name__ == "__main__":
    main()