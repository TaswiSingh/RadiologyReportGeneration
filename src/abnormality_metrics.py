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


NEGATION_WORDS = [
    "no",
    "without",
    "negative for",
    "absence of",
    "absent",
    "free of",
    "clear of",
    "no evidence of"
]


def has_positive_abnormality(text, abnormality):

    text = str(text).lower()

    if abnormality not in text:
        return False

    sentences = re.split(r"[.!?]", text)

    for sentence in sentences:

        if abnormality not in sentence:
            continue

        pos = sentence.find(abnormality)

        before = sentence[:pos]

        negated = False

        for neg in NEGATION_WORDS:

            if neg in before[-50:]:
                negated = True
                break

        if not negated:
            return True

    return False


df = pd.read_csv(RESULTS_PATH)


print("=" * 70)
print("IMPROVED ABNORMALITY DETECTION ANALYSIS")
print("=" * 70)

print("Samples:", len(df))
print()


rows = []


for abnormality in ABNORMALITIES:

    reference_positive = df["reference"].apply(
        lambda x: has_positive_abnormality(
            x,
            abnormality
        )
    )

    generated_positive = df["generated"].apply(
        lambda x: has_positive_abnormality(
            x,
            abnormality
        )
    )

    reference_count = reference_positive.sum()

    generated_count = generated_positive.sum()

    detected = (
        reference_positive &
        generated_positive
    ).sum()

    missed = (
        reference_positive &
        ~generated_positive
    ).sum()

    false_positive = (
        ~reference_positive &
        generated_positive
    ).sum()

    recall = (
        detected / reference_count
        if reference_count > 0
        else 0
    )

    precision = (
        detected / generated_count
        if generated_count > 0
        else 0
    )

    rows.append({
        "abnormality": abnormality,
        "reference": reference_count,
        "generated": generated_count,
        "detected": detected,
        "missed": missed,
        "false_positive": false_positive,
        "recall": recall,
        "precision": precision
    })


result = pd.DataFrame(rows)


print(
    f"{'Abnormality':<18}"
    f"{'Ref':>7}"
    f"{'Gen':>7}"
    f"{'Detect':>8}"
    f"{'Missed':>8}"
    f"{'FP':>7}"
    f"{'Recall':>10}"
    f"{'Precision':>11}"
)

print("-" * 80)


for _, row in result.iterrows():

    print(
        f"{row['abnormality']:<18}"
        f"{int(row['reference']):>7}"
        f"{int(row['generated']):>7}"
        f"{int(row['detected']):>8}"
        f"{int(row['missed']):>8}"
        f"{int(row['false_positive']):>7}"
        f"{row['recall'] * 100:>9.1f}%"
        f"{row['precision'] * 100:>10.1f}%"
    )


print("-" * 80)

print()

print(
    "Average recall:",
    f"{result['recall'].mean() * 100:.2f}%"
)

print(
    "Average precision:",
    f"{result['precision'].mean() * 100:.2f}%"
)


output_path = "evaluation/abnormality_metrics_improved.csv"

result.to_csv(
    output_path,
    index=False
)


print()
print("Saved:", output_path)

print("=" * 70)