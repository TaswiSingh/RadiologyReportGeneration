import pandas as pd

df = pd.read_csv("data/train.csv")

keys = [
    "nodule",
    "opacity",
    "atelectasis",
    "effusion",
    "pneumonia",
    "consolidation",
    "cardiomegaly",
    "pneumothorax",
    "fracture",
    "mass",
    "edema",
    "infiltrate",
    "granuloma",
    "scoliosis"
]

print("Validation samples:", len(df))
print()

for key in keys:
    count = (
        df["report"]
        .fillna("")
        .str.lower()
        .str.contains(key, regex=False)
        .sum()
    )

    print(f"{key:15s}: {count}")