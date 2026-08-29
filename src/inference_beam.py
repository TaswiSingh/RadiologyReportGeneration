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

# Maximum generated tokens
GENERATION_LENGTH = 60

# Beam search settings
BEAM_SIZE = 3


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
# Load X-ray
# ============================================================

print("\nLoading X-ray...")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

image = Image.open(IMAGE_PATH).convert("RGB")

image = transform(image)

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
# Load checkpoint
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
# Beam Search
# ============================================================

def beam_search(
    decoder,
    memory,
    bos_id,
    eos_id,
    beam_size=3,
    max_length=60
):
    """
    Beam search generation.

    Returns:
        best generated token sequence
    """

    # Each beam is:
    # (token_ids, log_probability)

    beams = [
        ([bos_id], 0.0)
    ]

    completed_beams = []

    for step in range(max_length):

        candidates = []

        for token_ids, score in beams:

            # If this beam already ended,
            # keep it as completed.
            if token_ids[-1] == eos_id:

                completed_beams.append(
                    (token_ids, score)
                )

                continue

            input_ids = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=DEVICE
            )

            attention_mask = torch.ones(
                (1, len(token_ids)),
                dtype=torch.bool,
                device=DEVICE
            )

            with torch.no_grad():

                logits = decoder(
                    input_ids=input_ids,
                    memory=memory,
                    attention_mask=attention_mask
                )

            # Last timestep
            next_token_logits = logits[:, -1, :]

            # Convert logits to log probabilities
            log_probs = torch.log_softmax(
                next_token_logits,
                dim=-1
            )

            # Take best beam_size candidates
            top_log_probs, top_ids = torch.topk(
                log_probs,
                beam_size,
                dim=-1
            )

            for j in range(beam_size):

                next_token = top_ids[0, j].item()

                next_score = (
                    score
                    + top_log_probs[0, j].item()
                )

                new_sequence = (
                    token_ids
                    + [next_token]
                )

                candidates.append(
                    (
                        new_sequence,
                        next_score
                    )
                )

        # If there are no active candidates,
        # generation is finished.
        if not candidates:
            break

        # Keep the best beam_size candidates.
        candidates.sort(
            key=lambda x: x[1],
            reverse=True
        )

        beams = candidates[:beam_size]

        # Stop if every beam has EOS.
        if all(
            sequence[-1] == eos_id
            for sequence, _ in beams
        ):
            completed_beams.extend(beams)
            break

    # Add remaining beams
    completed_beams.extend(beams)

    if not completed_beams:
        return [bos_id]

    # ========================================================
    # Length-normalized scoring
    # ========================================================

    def normalized_score(item):

        sequence, score = item

        length = max(
            len(sequence) - 1,
            1
        )

        return score / length

    completed_beams.sort(
        key=normalized_score,
        reverse=True
    )

    best_sequence = completed_beams[0][0]

    return best_sequence


# ============================================================
# Generate report
# ============================================================

print("\nGenerating report using Beam Search...")

generated_ids = beam_search(
    decoder=decoder,
    memory=visual_features,
    bos_id=bos_id,
    eos_id=eos_id,
    beam_size=BEAM_SIZE,
    max_length=GENERATION_LENGTH
)


# ============================================================
# Decode token IDs
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


generated_report = " ".join(
    generated_words
)


# ============================================================
# Print result
# ============================================================

print("\n========================================")
print("BEAM SEARCH RADIOLOGY REPORT")
print("========================================")

print(generated_report)

print("\n========================================")
print("Beam Search inference completed!")
print("========================================")

print("\nBeam size:")
print(BEAM_SIZE)

print("\nGenerated token IDs:")
print(generated_ids)