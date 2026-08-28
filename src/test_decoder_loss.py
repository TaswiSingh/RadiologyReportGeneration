import pickle
import torch
import torch.nn as nn

from dataset import ChestXrayDataset, collate_fn
from torch.utils.data import DataLoader
from models.visual_encoder import VisualEncoder
from models.text_decoder import TextDecoder


def main():

    # -----------------------------
    # Load vocabulary
    # -----------------------------

    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab.word2idx)

    print("Vocabulary size:", vocab_size)

    # -----------------------------
    # Load dataset
    # -----------------------------

    dataset = ChestXrayDataset(
        csv_path="data/train.csv",
        img_dir="images/images_normalized",
        vocab_path="data/vocab.pkl",
        max_length=128
    )

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

    # -----------------------------
    # Shift input and target
    # -----------------------------

    decoder_input_ids = input_ids[:, :-1]

    target_ids = input_ids[:, 1:]

    decoder_attention_mask = attention_mask[:, :-1]

    print("\nDecoder input shape:")
    print(decoder_input_ids.shape)

    print("\nTarget shape:")
    print(target_ids.shape)

    # -----------------------------
    # Create models
    # -----------------------------

    print("\nCreating Visual Encoder...")

    encoder = VisualEncoder(
        d_model=512,
        pretrained=True,
        freeze_backbone=True
    )

    encoder.eval()

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

    # -----------------------------
    # Image → visual features
    # -----------------------------

    with torch.no_grad():
        visual_features = encoder(images)

    print("\nVisual features:")
    print(visual_features.shape)

    # -----------------------------
    # Decoder
    # -----------------------------

    with torch.no_grad():
        logits = decoder(
            input_ids=decoder_input_ids,
            memory=visual_features,
            attention_mask=decoder_attention_mask
        )

    print("\nLogits:")
    print(logits.shape)

    # -----------------------------
    # Cross Entropy Loss
    # -----------------------------

    # Ignore <pad> token
    criterion = nn.CrossEntropyLoss(
        ignore_index=0
    )

    loss = criterion(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1)
    )

    print("\nLoss:")
    print(loss.item())

    # -----------------------------
    # Verify loss
    # -----------------------------

    assert loss.ndim == 0
    assert torch.isfinite(loss)

    print("\n================================")
    print("DECODER LOSS TEST PASSED!")
    print("================================")


if __name__ == "__main__":
    main()