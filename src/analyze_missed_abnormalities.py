import pandas as pd
from collections import Counter

RESULTS_PATH = "evaluation/results.csv"

ABNORMALITIES = [
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
    "scoliosis",
]


df = pd.read_csv(RESULTS_PATH)

references = df["reference"].fillna("").str.lower()
generated = df["generated"].fillna("").str.lower()


print("=" * 60)
print("MISSED ABNORMALITY ANALYSIS")
print("=" * 60)

print("\nTotal samples:", len(df))
print()


rows = []

for abnormality in ABNORMALITIES:

    ref_count = references.str.contains(
        abnormality,
        regex=False
    ).sum()

    generated_count = generated.str.contains(
        abnormality,
        regex=False
    ).sum()

    missed = (
        references.str.contains(abnormality, regex=False)
        &
        ~generated.str.contains(abnormality, regex=False)
    ).sum()

    rows.append({
        "abnormality": abnormality,
        "reference": ref_count,
        "generated": generated_count,
        "missed": missed
    })


result = pd.DataFrame(rows)

print(
    result.to_string(index=False)
)

print("\n" + "=" * 60)
print("MOST MISSED ABNORMALITIES")
print("=" * 60)

print(
    result.sort_values(
        "missed",
        ascending=False
    ).to_string(index=False)
)