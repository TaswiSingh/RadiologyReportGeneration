import pickle
import torch

from models.text_decoder import TextDecoder


def main():

    # Load vocabulary
    with open("data/vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    vocab_size = len(vocab.word2idx)

    print("Vocabulary size:", vocab_size)

    # Create decoder
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

    print("Text Decoder created successfully.")

    # Fake token IDs
    input_ids = torch.tensor([
        [2, 4, 5, 6, 7],
        [2, 8, 9, 10, 0]
    ])

    # True = actual token
    # False = padding
    attention_mask = torch.tensor([
        [True, True, True, True, True],
        [True, True, True, True, False]
    ])

    # Fake visual features
    # This has the same shape produced by Day 3
    memory = torch.randn(
        2,
        49,
        512
    )

    print("\nInput IDs shape:")
    print(input_ids.shape)

    print("\nVisual memory shape:")
    print(memory.shape)

    with torch.no_grad():

        logits = decoder(
            input_ids=input_ids,
            memory=memory,
            attention_mask=attention_mask
        )

    print("\nOutput logits shape:")
    print(logits.shape)

    print("\nExpected:")
    print(
        f"torch.Size([2, 5, {vocab_size}])"
    )

    assert logits.shape == (
        2,
        5,
        vocab_size
    )

    print("\nText Decoder test PASSED!")


if __name__ == "__main__":
    main()