import torch
from torch.utils.data import DataLoader

from dataset import ChestXrayDataset, collate_fn


def main():

    dataset = ChestXrayDataset(
        csv_path="data/train.csv",
        img_dir="images/images_normalized",
        vocab_path="data/vocab.pkl",
        max_length=128
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    batch = next(iter(loader))

    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]

    print("Original input_ids shape:")
    print(input_ids.shape)

    print("\nOriginal input_ids:")
    print(input_ids)

    # Remove <eos> from decoder input
    decoder_input_ids = input_ids[:, :-1]

    # Remove <bos> from target
    target_ids = input_ids[:, 1:]

    # Corresponding attention mask
    decoder_attention_mask = attention_mask[:, :-1]

    print("\nDecoder input shape:")
    print(decoder_input_ids.shape)

    print("\nTarget shape:")
    print(target_ids.shape)

    print("\nDecoder input:")
    print(decoder_input_ids)

    print("\nTarget:")
    print(target_ids)

    # Verify sequence alignment
    assert decoder_input_ids.shape == target_ids.shape

    # Verify next-token shift
    assert torch.equal(
        decoder_input_ids[:, 1:],
        target_ids[:, :-1]
    )

    print("\nTraining batch preparation PASSED!")


if __name__ == "__main__":
    main()