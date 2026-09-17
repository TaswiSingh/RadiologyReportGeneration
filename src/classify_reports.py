import pandas as pd

df = pd.read_csv("evaluation/results.csv")

# Words/phrases that usually indicate an abnormal finding.
abnormal_terms = [
    "nodule",
    "mass",
    "opacity",
    "atelectasis",
    "consolidation",
    "pneumonia",
    "cardiomegaly",
    "edema",
    "fracture",
    "granuloma",
    "scoliosis",
]

def contains_abnormality(text):
    text = str(text).lower()

    for term in abnormal_terms:
        if term in text:
            return True

    return False


df["reference_abnormal"] = df["reference"].apply(
    contains_abnormality
)

df["generated_abnormal"] = df["generated"].apply(
    contains_abnormality
)

print("=" * 60)
print("REFERENCE vs GENERATED ABNORMALITY ANALYSIS")
print("=" * 60)

print("\nTotal samples:", len(df))

print(
    "\nReference abnormal:",
    df["reference_abnormal"].sum()
)

print(
    "Reference without detected abnormality:",
    (~df["reference_abnormal"]).sum()
)

print(
    "\nGenerated abnormal:",
    df["generated_abnormal"].sum()
)

print(
    "Generated without detected abnormality:",
    (~df["generated_abnormal"]).sum()
)

# --------------------------------------------------
# Abnormal reference -> normal generated
# --------------------------------------------------

missed = df[
    (df["reference_abnormal"] == True) &
    (df["generated_abnormal"] == False)
]

print(
    "\nAbnormal references missed by model:",
    len(missed)
)

# --------------------------------------------------
# Save
# --------------------------------------------------

missed[
    ["filename", "reference", "generated"]
].to_csv(
    "evaluation/missed_abnormalities.csv",
    index=False
)

print(
    "\nSaved: evaluation/missed_abnormalities.csv"
)

# --------------------------------------------------
# Show examples
# --------------------------------------------------

print("\n" + "=" * 60)
print("EXAMPLES OF MISSED ABNORMALITIES")
print("=" * 60)

for _, row in missed.head(10).iterrows():

    print("\nFILE:", row["filename"])

    print("\nREFERENCE:")
    print(row["reference"])

    print("\nGENERATED:")
    print(row["generated"])

    print("-" * 60)