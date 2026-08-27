from torch.utils.data import DataLoader

from dataset import ChestXrayDataset, collate_fn


CSV_PATH = "../data/train.csv"
IMG_DIR = "../images/images_normalized"
VOCAB_PATH = "../data/vocab.pkl"


print("Using CSV:", CSV_PATH)
print("Using image directory:", IMG_DIR)
print("Using vocabulary:", VOCAB_PATH)


dataset = ChestXrayDataset(
    csv_path=CSV_PATH,
    img_dir=IMG_DIR,
    vocab_path=VOCAB_PATH,
    max_length=128
)


print("\nDataset Size:", len(dataset))


sample = dataset[0]

print("\n--- Single Sample ---")

print("Filename:")
print(sample["filename"])

print("\nImage shape:")
print(sample["image"].shape)

print("\nInput IDs shape:")
print(sample["input_ids"].shape)

print("\nInput IDs:")
print(sample["input_ids"][:20])

print("\nReport:")
print(sample["report"][:300])


loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


batch = next(iter(loader))


print("\n--- Batch ---")

print("Images shape:")
print(batch["images"].shape)

print("Input IDs shape:")
print(batch["input_ids"].shape)

print("Attention mask shape:")
print(batch["attention_mask"].shape)

print("\nFilenames:")
print(batch["filenames"])

print("\nAttention mask:")
print(batch["attention_mask"])