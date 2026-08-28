import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset, collate_fn
from models.visual_encoder import VisualEncoder
from models.text_decoder import TextDecoder


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "data/train.csv"
VAL_CSV = "data/val.csv"

IMAGE_DIR = "images/images_normalized"
VOCAB_PATH = "data/vocab.pkl"

CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 2
EPOCHS = 5

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

MAX_LENGTH = 128

DEVICE = torch.device("cpu")

# CPU smoke test
MAX_TRAIN_BATCHES = 300
MAX_VAL_BATCHES = 100


# ============================================================
# Training function
# ============================================================

def train_one_epoch(
    encoder,
    decoder,
    loader,
    criterion,
    optimizer,
    vocab_size
):

    decoder.train()
    encoder.eval()

    total_loss = 0.0
    batches = 0

    for step, batch in enumerate(loader):

        if step >= MAX_TRAIN_BATCHES:
            break

        images = batch["images"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        # Teacher forcing
        decoder_input_ids = input_ids[:, :-1]
        target_ids = input_ids[:, 1:]
        decoder_attention_mask = attention_mask[:, :-1]

        # Image → visual features
        with torch.no_grad():
            visual_features = encoder(images)

        # Visual features → text decoder
        logits = decoder(
            input_ids=decoder_input_ids,
            memory=visual_features,
            attention_mask=decoder_attention_mask
        )

        # Loss
        loss = criterion(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1)
        )

        # Backpropagation
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        batches += 1

        print(
            f"Train Batch [{batches}/{MAX_TRAIN_BATCHES}] "
            f"Loss: {loss.item():.4f}"
        )

    return total_loss / batches


# ============================================================
# Validation function
# ============================================================

def validate(
    encoder,
    decoder,
    loader,
    criterion,
    vocab_size
):

    encoder.eval()
    decoder.eval()

    total_loss = 0.0
    batches = 0

    with torch.no_grad():

        for step, batch in enumerate(loader):

            if step >= MAX_VAL_BATCHES:
                break

            images = batch["images"].to(DEVICE)
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            decoder_input_ids = input_ids[:, :-1]
            target_ids = input_ids[:, 1:]
            decoder_attention_mask = attention_mask[:, :-1]

            visual_features = encoder(images)

            logits = decoder(
                input_ids=decoder_input_ids,
                memory=visual_features,
                attention_mask=decoder_attention_mask
            )

            loss = criterion(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1)
            )

            total_loss += loss.item()
            batches += 1

            print(
                f"Validation Batch [{batches}/{MAX_VAL_BATCHES}] "
                f"Loss: {loss.item():.4f}"
            )

    return total_loss / batches


# ============================================================
# Main
# ============================================================

def main():

    print("========================================")
    print("Radiology Report Generation Training")
    print("========================================")

    print("\nDevice:", DEVICE)

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    print("\nLoading vocabulary...")

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab.word2idx)

    print("Vocabulary size:", vocab_size)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    print("\nLoading training dataset...")

    train_dataset = ChestXrayDataset(
        csv_path=TRAIN_CSV,
        img_dir=IMAGE_DIR,
        vocab_path=VOCAB_PATH,
        max_length=MAX_LENGTH
    )

    print("Training samples:", len(train_dataset))

    print("\nLoading validation dataset...")

    val_dataset = ChestXrayDataset(
        csv_path=VAL_CSV,
        img_dir=IMAGE_DIR,
        vocab_path=VOCAB_PATH,
        max_length=MAX_LENGTH
    )

    print("Validation samples:", len(val_dataset))

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # --------------------------------------------------------
    # Visual Encoder
    # --------------------------------------------------------

    print("\nCreating Visual Encoder...")

    encoder = VisualEncoder(
        d_model=512,
        pretrained=True,
        freeze_backbone=True
    )

    encoder = encoder.to(DEVICE)
    encoder.eval()

    print("Visual Encoder ready.")

    # --------------------------------------------------------
    # Text Decoder
    # --------------------------------------------------------

    print("\nCreating Text Decoder...")

    decoder = TextDecoder(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=4,
        dim_feedforward=2048,
        dropout=0.1,
        max_length=MAX_LENGTH
    )

    decoder = decoder.to(DEVICE)

    print("Text Decoder ready.")

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    pad_id = vocab.word2idx["<pad>"]

    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_id
    )

    print("\nPAD token ID:", pad_id)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        decoder.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_loss = float("inf")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(EPOCHS):

        print("\n========================================")
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("========================================")

        train_loss = train_one_epoch(
            encoder,
            decoder,
            train_loader,
            criterion,
            optimizer,
            vocab_size
        )

        print(
            f"\nTraining Loss: {train_loss:.4f}"
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print("\nRunning validation...")

        val_loss = validate(
            encoder,
            decoder,
            val_loader,
            criterion,
            vocab_size
        )

        print(
            f"\nValidation Loss: {val_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "best_decoder.pt"
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": decoder.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "vocab_size": vocab_size
                },
                checkpoint_path
            )

            print(
                "\nBest model saved to:",
                checkpoint_path
            )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print("\n========================================")
    print("TRAINING + VALIDATION TEST PASSED!")
    print("========================================")

    print("\nBest validation loss:", best_val_loss)

    print(
        "\nCheckpoint:",
        os.path.join(
            CHECKPOINT_DIR,
            "best_decoder.pt"
        )
    )


if __name__ == "__main__":
    main()