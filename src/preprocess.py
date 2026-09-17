import os
import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------
# Paths
# -----------------------------
DATA_DIR = "data"

REPORTS_PATH = os.path.join(DATA_DIR, "indiana_reports.csv")
PROJECTIONS_PATH = os.path.join(DATA_DIR, "indiana_projections.csv")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "val.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")


# -----------------------------
# Load CSV Files
# -----------------------------
reports = pd.read_csv(REPORTS_PATH)
projections = pd.read_csv(PROJECTIONS_PATH)

print(f"Reports: {len(reports)}")
print(f"Images : {len(projections)}")


# -----------------------------
# Merge Reports with Images
# -----------------------------
df = pd.merge(
    reports,
    projections,
    on="uid",
    how="inner"
)


# -----------------------------
# Keep Required Columns
# -----------------------------
df = df[
    [
        "uid",
        "filename",
        "projection",
        "findings",
        "impression",
    ]
]


# -----------------------------
# Remove Missing Values
# -----------------------------
df["findings"] = df["findings"].fillna("")
df["impression"] = df["impression"].fillna("")

df = df[
    (df["findings"].str.strip() != "") |
    (df["impression"].str.strip() != "")
]

df = df.reset_index(drop=True)


# -----------------------------
# Combine Report
# -----------------------------
df["report"] = (
    "Findings: "
    + df["findings"]
    + " Impression: "
    + df["impression"]
)


# -----------------------------
# Remove Duplicate Images
# -----------------------------
df = df.drop_duplicates(subset=["filename"])


print("\nFinal Dataset Size:", len(df))


# -----------------------------
# Train / Validation / Test Split
# -----------------------------
train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    shuffle=True
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42
)


# -----------------------------
# Save CSV Files
# -----------------------------
train_df.to_csv(TRAIN_PATH, index=False)
val_df.to_csv(VAL_PATH, index=False)
test_df.to_csv(TEST_PATH, index=False)


print("\nSaved Successfully!")
print(f"Train : {len(train_df)}")
print(f"Validation : {len(val_df)}")
print(f"Test : {len(test_df)}")