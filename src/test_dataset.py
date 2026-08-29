import torch

from dataset import ChestXrayDataset, collate_fn
from torch.utils.data import DataLoader


CSV_PATH = "data/train.csv"
IMAGE_DIR = "images/images_normalized"
VOCAB_PATH = "data/vocab.pkl"

MAX_LENGTH = 128


print("Using CSV:", CSV_PATH)
print("Using image directory:", IMAGE_DIR)
print("Using vocabulary:", VOCAB_PATH)


dataset = ChestXrayDataset(
    csv_path=CSV_PATH,
    img_dir=IMAGE_DIR,
    vocab_path=VOCAB_PATH,
    max_length=MAX_LENGTH
)


print("\nDataset Size:", len(dataset))


sample = dataset[0]


print("\nSample keys:")
print(sample.keys())


print("\nFilename:")
print(sample["filename"])


print("\nImage shape:")
print(sample["image"].shape)


print("\nInput IDs shape:")
print(sample["input_ids"].shape)


print("\nReport:")
print(sample["report"])


# ============================================================
# TEST COLLATE FUNCTION
# ============================================================

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


batch = next(iter(loader))


print("\n========================================")
print("COLLATE FUNCTION TEST")
print("========================================")


print("\nBatch keys:")
print(batch.keys())


print("\nBatch image shape:")
print(batch["images"].shape)


print("\nBatch input IDs shape:")
print(batch["input_ids"].shape)


print("\nBatch attention mask shape:")
print(batch["attention_mask"].shape)


print("\n========================================")
print("DATASET TEST PASSED!")
print("========================================")
