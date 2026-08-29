import os
import re
import pandas as pd


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
    "scoliosis"
]


NEGATIONS = [
    "no",
    "not",
    "without",
    "negative for",
    "free of",
    "absence of",
    "absent",
    "ruled out"
]


def is_present(text, abnormality):
    text = str(text).lower()

    matches = list(
        re.finditer(
            r"\b" + re.escape(abnormality) + r"\b",
            text
        )
    )

    for match in matches:

        start = max(0, match.start() - 60)

        context = text[start:match.start()]

        # Use the last sentence as local context
        context = re.split(
            r"[.!?]",
            context
        )[-1]

        negated = False

        for neg in NEGATIONS:
            if re.search(
                r"\b" + re.escape(neg) + r"\b",
                context
            ):
                negated = True
                break

        if not negated:
            return True

    return False


# ============================================================
# LOAD RESULTS
# ============================================================

print("=" * 72)
print("NEGATION-AWARE ABNORMALITY ANALYSIS")
print("=" * 72)

df = pd.read_csv(RESULTS_PATH)

print("Samples:", len(df))
print()


rows = []


# ============================================================
# ANALYSIS
# ============================================================

for abnormality in ABNORMALITIES:

    ref_present = df["reference"].apply(
        lambda x: is_present(x, abnormality)
    )

    gen_present = df["generated"].apply(
        lambda x: is_present(x, abnormality)
    )

    tp = (ref_present & gen_present).sum()

    fn = (ref_present & ~gen_present).sum()

    fp = (~ref_present & gen_present).sum()

    tn = (~ref_present & ~gen_present).sum()

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    rows.append({
        "abnormality": abnormality,
        "reference_present": int(ref_present.sum()),
        "generated_present": int(gen_present.sum()),
        "true_positive": int(tp),
        "false_negative": int(fn),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "recall": recall,
        "precision": precision
    })


result = pd.DataFrame(rows)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    f"{'Abnormality':<18}"
    f"{'Ref':>7}"
    f"{'Gen':>7}"
    f"{'TP':>7}"
    f"{'FN':>7}"
    f"{'FP':>7}"
    f"{'Recall':>10}"
    f"{'Precision':>12}"
)

print("-" * 72)


for _, row in result.iterrows():

    print(
        f"{row['abnormality']:<18}"
        f"{int(row['reference_present']):>7}"
        f"{int(row['generated_present']):>7}"
        f"{int(row['true_positive']):>7}"
        f"{int(row['false_negative']):>7}"
        f"{int(row['false_positive']):>7}"
        f"{row['recall'] * 100:>9.1f}%"
        f"{row['precision'] * 100:>11.1f}%"
    )


print("-" * 72)


# ============================================================
# OVERALL METRICS
# ============================================================

macro_recall = result["recall"].mean()

macro_precision = result["precision"].mean()

total_tp = result["true_positive"].sum()
total_fn = result["false_negative"].sum()
total_fp = result["false_positive"].sum()

micro_recall = (
    total_tp / (total_tp + total_fn)
    if total_tp + total_fn > 0
    else 0
)

micro_precision = (
    total_tp / (total_tp + total_fp)
    if total_tp + total_fp > 0
    else 0
)


print()
print("MACRO RECALL     :", f"{macro_recall * 100:.2f}%")
print("MACRO PRECISION  :", f"{macro_precision * 100:.2f}%")
print("MICRO RECALL     :", f"{micro_recall * 100:.2f}%")
print("MICRO PRECISION  :", f"{micro_precision * 100:.2f}%")


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "evaluation",
    exist_ok=True
)

output_path = (
    "evaluation/"
    "abnormality_metrics_negation.csv"
)

result.to_csv(
    output_path,
    index=False
)

print()
print("Saved:", output_path)

print("=" * 72)