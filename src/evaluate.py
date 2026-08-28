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

def generate_report(memory):

    generated = [bos_id]

    for _ in range(MAX_LENGTH - 1):

        input_ids = torch.tensor(
            [generated],
            dtype=torch.long,
            device=DEVICE
        )

        attention_mask = torch.ones(
            (1, len(generated)),
            dtype=torch.bool,
            device=DEVICE
        )

        logits = decoder(
            input_ids=input_ids,
            memory=memory,
            attention_mask=attention_mask
        )

        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1
        ).item()

        generated.append(next_token)

        if next_token == eos_id:
            break

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

output_path = "evaluation/results.csv"

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