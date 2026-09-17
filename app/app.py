import os
import sys
import pickle
import importlib.util

import torch
import streamlit as st
from PIL import Image
from torchvision import transforms


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    ROOT_DIR,
    "src"
)

# Add project root and src to Python path
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


# ============================================================
# LOAD VOCABULARY MODULE
# ============================================================
# vocab.pkl was created using the module name "vocabulary".
# We explicitly load src/vocabulary.py with that module name
# before calling pickle.load().

vocabulary_path = os.path.join(
    SRC_DIR,
    "vocabulary.py"
)

spec = importlib.util.spec_from_file_location(
    "vocabulary",
    vocabulary_path
)

vocabulary_module = importlib.util.module_from_spec(spec)

sys.modules["vocabulary"] = vocabulary_module

spec.loader.exec_module(vocabulary_module)


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.models.visual_encoder import VisualEncoder
from src.models.text_decoder import TextDecoder


# ============================================================
# CONFIGURATION
# ============================================================

D_MODEL = 512
NHEAD = 8
DECODER_LAYERS = 4
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
MAX_LENGTH = 128
GENERATION_LENGTH = 60

VOCAB_PATH = os.path.join(
    ROOT_DIR,
    "data",
    "vocab.pkl"
)

CHECKPOINT_PATH = os.path.join(
    ROOT_DIR,
    "checkpoints",
    "best_decoder.pt"
)

DEVICE = torch.device("cpu")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Radiology Report Generator",
    page_icon="🩻",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩻 Radiology Report Generator")

st.write(
    "Upload a chest X-ray image and the trained AI model "
    "will generate a radiology-style report."
)

st.warning(
    "⚠️ Research/Educational Prototype — "
    "This system is not intended for clinical diagnosis "
    "or medical decision-making."
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    # --------------------------------------------------------
    # Load vocabulary
    # --------------------------------------------------------

    with open(VOCAB_PATH, "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab.word2idx)

    bos_id = vocab.word2idx["<bos>"]
    eos_id = vocab.word2idx["<eos>"]

    # --------------------------------------------------------
    # Image preprocessing
    # --------------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # --------------------------------------------------------
    # Visual Encoder
    # --------------------------------------------------------

    encoder = VisualEncoder(
        d_model=D_MODEL,
        pretrained=True,
        freeze_backbone=True
    )

    encoder = encoder.to(DEVICE)
    encoder.eval()

    # --------------------------------------------------------
    # Text Decoder
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Load trained checkpoint
    # --------------------------------------------------------

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    decoder.load_state_dict(
        checkpoint["model_state_dict"]
    )

    decoder.eval()

    return (
        encoder,
        decoder,
        vocab,
        bos_id,
        eos_id,
        transform
    )


# ============================================================
# GENERATE REPORT
# ============================================================

def generate_report(
    image,
    encoder,
    decoder,
    vocab,
    bos_id,
    eos_id,
    transform
):

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    image = image.convert("RGB")

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(DEVICE)

    # --------------------------------------------------------
    # Extract visual features
    # --------------------------------------------------------

    with torch.no_grad():
        visual_features = encoder(image_tensor)

    # --------------------------------------------------------
    # Start with BOS token
    # --------------------------------------------------------

    generated_ids = [bos_id]

    # --------------------------------------------------------
    # Generate tokens
    # --------------------------------------------------------

    with torch.no_grad():

        for _ in range(GENERATION_LENGTH):

            input_ids = torch.tensor(
                [generated_ids],
                dtype=torch.long,
                device=DEVICE
            )

            attention_mask = torch.ones_like(
                input_ids,
                dtype=torch.bool
            )

            logits = decoder(
                input_ids=input_ids,
                memory=visual_features,
                attention_mask=attention_mask
            )

            next_token_logits = (
                logits[:, -1, :]
                .squeeze(0)
            )

            # Prevent immediate repetition
            if len(generated_ids) > 1:

                previous_token = generated_ids[-1]

                next_token_logits[
                    previous_token
                ] = float("-inf")

            # Reduce recent token repetition
            recent_tokens = generated_ids[-8:]

            for token_id in set(recent_tokens):

                if token_id != bos_id:
                    next_token_logits[token_id] *= 0.5

            # Prevent BOS from appearing again
            next_token_logits[bos_id] = float("-inf")

            # Select next token
            next_token = torch.argmax(
                next_token_logits
            ).item()

            generated_ids.append(next_token)

            # Stop at EOS
            if next_token == eos_id:
                break

    # ========================================================
    # Convert IDs to words
    # ========================================================

    words = []

    for token_id in generated_ids:

        if token_id == bos_id:
            continue

        if token_id == eos_id:
            break

        if token_id in vocab.idx2word:
            words.append(
                vocab.idx2word[token_id]
            )

    return " ".join(words)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    with st.spinner("Loading AI model..."):

        (
            encoder,
            decoder,
            vocab,
            bos_id,
            eos_id,
            transform
        ) = load_model()

    st.success("AI model loaded successfully.")

except Exception as e:

    st.error("Could not load the AI model.")

    st.exception(e)

    st.stop()


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a chest X-ray image",
    type=["png", "jpg", "jpeg"]
)


# ============================================================
# DISPLAY IMAGE AND GENERATE REPORT
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded X-ray")

    st.image(
        image,
        caption="Chest X-ray",
        use_container_width=True
    )

    if st.button(
        "🔍 Generate Radiology Report",
        type="primary"
    ):

        with st.spinner(
            "Analyzing X-ray and generating report..."
        ):

            try:

                report = generate_report(
                    image,
                    encoder,
                    decoder,
                    vocab,
                    bos_id,
                    eos_id,
                    transform
                )

                st.subheader("Generated Report")

                st.text_area(
                    "Radiology Report",
                    report,
                    height=250
                )

                st.success(
                    "Report generated successfully."
                )

            except Exception as e:

                st.error(
                    "An error occurred while generating "
                    "the report."
                )

                st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Radiology Report Generation — "
    "AI Research Project | "
    "For educational/research use only"
)