import pandas as pd
from collections import Counter

RESULTS_PATH = "evaluation/results.csv"

df = pd.read_csv(RESULTS_PATH)

print("=" * 60)
print("RADIOLOGY REPORT ERROR ANALYSIS")
print("=" * 60)

print("\nTotal samples:", len(df))

# --------------------------------------------------
# Report lengths
# --------------------------------------------------

df["reference_words"] = (
    df["reference"]
    .fillna("")
    .str.split()
    .str.len()
)

df["generated_words"] = (
    df["generated"]
    .fillna("")
    .str.split()
    .str.len()
)

print("\nAverage reference length:",
      round(df["reference_words"].mean(), 2))

print("Average generated length:",
      round(df["generated_words"].mean(), 2))

# --------------------------------------------------
# Empty reports
# --------------------------------------------------

empty_reports = (
    df["generated"]
    .fillna("")
    .str.strip()
    .eq("")
    .sum()
)

print("\nEmpty generated reports:", empty_reports)

# --------------------------------------------------
# Duplicate reports
# --------------------------------------------------

duplicate_reports = df["generated"].duplicated().sum()

print("Duplicate generated reports:", duplicate_reports)

# --------------------------------------------------
# Very short reports
# --------------------------------------------------

short_reports = (df["generated_words"] < 8).sum()

print("Very short generated reports:", short_reports)

# --------------------------------------------------
# Most common generated reports
# --------------------------------------------------

print("\nMost common generated reports:")

counts = Counter(df["generated"])

for i, (report, count) in enumerate(counts.most_common(10), 1):
    print(f"\n{i}. Count: {count}")
    print(report)

# --------------------------------------------------
# Common abnormality keywords
# --------------------------------------------------

keywords = [
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

print("\n" + "=" * 60)
print("ABNORMALITY KEYWORD ANALYSIS")
print("=" * 60)

for keyword in keywords:

    reference_count = (
        df["reference"]
        .fillna("")
        .str.lower()
        .str.contains(keyword, regex=False)
        .sum()
    )

    generated_count = (
        df["generated"]
        .fillna("")
        .str.lower()
        .str.contains(keyword, regex=False)
        .sum()
    )

    print(
        f"{keyword:15s} "
        f"Reference: {reference_count:3d} | "
        f"Generated: {generated_count:3d}"
    )

# --------------------------------------------------
# Save analysis
# --------------------------------------------------

df.to_csv(
    "evaluation/analyzed_results.csv",
    index=False
)

print("\nSaved:")
print("evaluation/analyzed_results.csv")

print("\n" + "=" * 60)
print("ERROR ANALYSIS COMPLETED")
print("=" * 60)