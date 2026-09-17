import os
import pickle
import torch
import pandas as pd
from PIL import Image
from torchvision import transforms

from models.visual_encoder import VisualEncoder
from models.text_decoder import TextDecoder


# ============================================================
# CONFIG
# ============================================================

TEST_CSV = "data/test.csv"
IMAGE_DIR = "images/images_normalized"
VOCAB_PATH = "data/vocab.pkl"
CHECKPOINT_PATH = "checkpoints/best_decoder.pt"

DEVICE = torch.device("cpu")

MAX_LENGTH = 128

# Number of test images to evaluate initially.
# Keep this small because you are running on CPU.
NUM_SAMPLES = 100


# ============================================================
# LOAD VOCABULARY
# ============================================================

print("Loading vocabulary...")

with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

vocab_size = len(vocab.word2idx)

print("Vocabulary size:", vocab_size)


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\nLoading test dataset...")

df = pd.read_csv(TEST_CSV)

print("Test samples:", len(df))


# ============================================================
# IMAGE TRANSFORM
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
# CREATE MODELS
# ============================================================

print("\nCreating Visual Encoder...")

encoder = VisualEncoder(
    d_model=512,
    pretrained=True,
    freeze_backbone=True
)

encoder = encoder.to(DEVICE)
encoder.eval()

print("Visual Encoder ready.")


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


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print("\nLoading checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE
)

if "decoder_state_dict" in checkpoint:
    decoder.load_state_dict(
        checkpoint["decoder_state_dict"]
    )
elif "model_state_dict" in checkpoint:
    decoder.load_state_dict(
        checkpoint["model_state_dict"]
    )
else:
    decoder.load_state_dict(checkpoint)

print("Checkpoint loaded successfully.")

if isinstance(checkpoint, dict):
    if "val_loss" in checkpoint:
        print("Checkpoint validation loss:",
              checkpoint["val_loss"])


decoder.eval()


# ============================================================
# TOKEN IDS
# ============================================================

bos_id = vocab.word2idx["<bos>"]
eos_id = vocab.word2idx["<eos>"]


# ============================================================
# GENERATE REPORT
# ============================================================

def violates_no_repeat_ngram(sequence, candidate, ngram_size=3):

    if len(sequence) < ngram_size - 1:
        return False

    new_ngram = tuple(
        sequence[-(ngram_size - 1):] + [candidate]
    )

    for i in range(len(sequence) - ngram_size + 1):

        existing_ngram = tuple(
            sequence[i:i + ngram_size]
        )

        if existing_ngram == new_ngram:
            return True

    return False


def generate_report(memory):

    BEAM_SIZE = 3
    LENGTH_PENALTY = 0.7
    MAX_GENERATION_LENGTH = 128

    # sequence, score, finished
    beams = [
        ([bos_id], 0.0, False)
    ]

    for _ in range(MAX_GENERATION_LENGTH - 1):

        candidates = []

        for sequence, score, finished in beams:

            if finished:

                candidates.append(
                    (sequence, score, True)
                )

                continue

            input_ids = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=DEVICE
            )

            attention_mask = torch.ones(
                (1, len(sequence)),
                dtype=torch.bool,
                device=DEVICE
            )

            logits = decoder(
                input_ids=input_ids,
                memory=memory,
                attention_mask=attention_mask
            )

            next_token_logits = logits[:, -1, :]

            log_probs = torch.log_softmax(
                next_token_logits,
                dim=-1
            )

            top_log_probs, top_ids = torch.topk(
                log_probs,
                BEAM_SIZE
            )

            for j in range(BEAM_SIZE):

                token_id = top_ids[0, j].item()

                token_score = top_log_probs[0, j].item()

                # Prevent repeated 3-grams
                if token_id != eos_id:

                    if violates_no_repeat_ngram(
                        sequence,
                        token_id,
                        3
                    ):
                        continue

                new_sequence = (
                    sequence + [token_id]
                )

                new_score = (
                    score + token_score
                )

                finished_now = (
                    token_id == eos_id
                )

                candidates.append(
                    (
                        new_sequence,
                        new_score,
                        finished_now
                    )
                )

        if not candidates:
            break

        def normalized_score(item):

            sequence = item[0]
            score = item[1]

            length = max(
                1,
                len(sequence) - 1
            )

            return score / (
                length ** LENGTH_PENALTY
            )

        candidates.sort(
            key=normalized_score,
            reverse=True
        )

        beams = candidates[:BEAM_SIZE]

        if all(
            beam[2]
            for beam in beams
        ):
            break

    beams.sort(
        key=lambda item:
            item[1] /
            (
                max(
                    1,
                    len(item[0]) - 1
                )
                ** LENGTH_PENALTY
            ),
        reverse=True
    )

    generated = beams[0][0]

    words = []

    for token_id in generated:

        if token_id == bos_id:
            continue

        if token_id == eos_id:
            break

        word = vocab.idx2word.get(
            token_id,
            "<unk>"
        )

        words.append(word)

    return " ".join(words)

# ============================================================
# EVALUATION
# ============================================================

results = []

num_samples = min(
    NUM_SAMPLES,
    len(df)
)

print("\n========================================")
print("STARTING EVALUATION")
print("========================================")

print("Evaluating", num_samples, "test images...")


with torch.no_grad():

    for i in range(num_samples):

        row = df.iloc[i]

        filename = row["filename"]

        image_path = os.path.join(
            IMAGE_DIR,
            filename
        )

        if not os.path.exists(image_path):

            print(
                f"\nSkipping missing image: {filename}"
            )

            continue

        image = Image.open(
            image_path
        ).convert("RGB")

        image = transform(image)

        image = image.unsqueeze(0)

        image = image.to(DEVICE)

        # --------------------------------------------
        # Visual features
        # --------------------------------------------

        memory = encoder(image)

        # --------------------------------------------
        # Generate report
        # --------------------------------------------

        generated_report = generate_report(
            memory
        )

        reference_report = str(
            row["report"]
        )

        results.append({
            "filename": filename,
            "reference": reference_report,
            "generated": generated_report
        })

        print("\n----------------------------------------")
        print(f"Sample {i + 1}/{num_samples}")
        print("Filename:")
        print(filename)

        print("\nReference:")
        print(reference_report)

        print("\nGenerated:")
        print(generated_report)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

os.makedirs(
    "evaluation",
    exist_ok=True
)

output_path = "evaluation/beam_results.csv"

results_df.to_csv(
    output_path,
    index=False
)

print("\n========================================")
print("EVALUATION COMPLETED")
print("========================================")

print("Results saved to:")
print(output_path)

print("Evaluated samples:", len(results_df))