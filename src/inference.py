import os
import pickle

import torch
from PIL import Image
from torchvision import transforms

from models.visual_encoder import VisualEncoder
from models.text_decoder import TextDecoder


# ============================================================
# Configuration
# ============================================================

IMAGE_PATH = "images/images_normalized/1019_IM-0015-2001.dcm.png"

VOCAB_PATH = "data/vocab.pkl"

CHECKPOINT_PATH = "checkpoints/best_decoder.pt"

DEVICE = torch.device("cpu")

D_MODEL = 512
NHEAD = 8
DECODER_LAYERS = 4
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
MAX_LENGTH = 128

# Maximum number of tokens to generate
GENERATION_LENGTH = 60


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# Load vocabulary
# ============================================================

print("Loading vocabulary...")

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab.word2idx)

print("Vocabulary size:", vocab_size)

bos_id = vocab.word2idx["<bos>"]
eos_id = vocab.word2idx["<eos>"]


# ============================================================
# Load image
# ============================================================

print("\nLoading X-ray...")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

image = transform(image)

# Add batch dimension
image = image.unsqueeze(0)

image = image.to(DEVICE)

print("Image shape:", image.shape)


# ============================================================
# Create Visual Encoder
# ============================================================

print("\nCreating Visual Encoder...")

encoder = VisualEncoder(
    d_model=D_MODEL,
    pretrained=True,
    freeze_backbone=True
)

encoder = encoder.to(DEVICE)

encoder.eval()

print("Visual Encoder ready.")


# ============================================================
# Create Text Decoder
# ============================================================

print("\nCreating Text Decoder...")

decoder = TextDecoder(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    nhead=NHEAD,
    num_layers=DECODER_LAYERS,
    dim_feedforward=DIM_FEEDFORWARD,
    dropout=DROPOUT,
    max_length=MAX_LENGTH
)

decoder = decoder.to(DEVICE)

print("Text Decoder created.")


# ============================================================
# Load trained decoder checkpoint
# ============================================================

print("\nLoading decoder checkpoint...")

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT_PATH}"
    )

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

decoder.load_state_dict(
    checkpoint["model_state_dict"]
)

decoder.eval()

print("Checkpoint loaded successfully.")

print(
    "Checkpoint validation loss:",
    checkpoint.get("val_loss", "N/A")
)


# ============================================================
# Extract visual features
# ============================================================

print("\nExtracting visual features...")

with torch.no_grad():

    visual_features = encoder(image)

print(
    "Visual feature shape:",
    visual_features.shape
)


# ============================================================
# Generate report
# ============================================================

print("\nGenerating report...")

generated_ids = [bos_id]

for step in range(GENERATION_LENGTH):

    # Convert generated tokens to tensor
    input_ids = torch.tensor(
        [generated_ids],
        dtype=torch.long,
        device=DEVICE
    )

    # All currently generated tokens are real tokens
    attention_mask = torch.ones(
        (1, len(generated_ids)),
        dtype=torch.bool,
        device=DEVICE
    )

    with torch.no_grad():

        logits = decoder(
            input_ids=input_ids,
            memory=visual_features,
            attention_mask=attention_mask
        )

    # Get prediction for the last token
    next_token_logits = logits[:, -1, :]

    # Greedy decoding
    next_token_id = torch.argmax(
        next_token_logits,
        dim=-1
    ).item()

    generated_ids.append(next_token_id)

    # Stop when EOS is generated
    if next_token_id == eos_id:
        break


# ============================================================
# Decode tokens
# ============================================================

generated_words = []

for token_id in generated_ids:

    word = vocab.idx2word.get(
        token_id,
        "<unk>"
    )

    if word == "<bos>":
        continue

    if word == "<eos>":
        break

    generated_words.append(word)


generated_report = " ".join(generated_words)


# ============================================================
# Print result
# ============================================================

print("\n========================================")
print("GENERATED RADIOLOGY REPORT")
print("========================================")

print(generated_report)

print("\n========================================")
print("Inference completed successfully!")
print("========================================")

print("\nGenerated token IDs:")
print(generated_ids)