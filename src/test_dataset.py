from dataset import ChestXrayDataset

IMG_DIR = "images/images_normalized"

print("Using image directory:", IMG_DIR)

dataset = ChestXrayDataset(
    csv_path="data/train.csv",
    img_dir=IMG_DIR
)

print("Dataset Size:", len(dataset))

sample = dataset[0]

print(sample["filename"])
print(sample["image"].shape)
print(sample["report"])