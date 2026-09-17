import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


# ============================================================
# CONFIG
# ============================================================

RESULTS_PATH = "evaluation/results.csv"


# ============================================================
# LOAD RESULTS
# ============================================================

print("Loading evaluation results...")

df = pd.read_csv(RESULTS_PATH)

print("Samples:", len(df))


# ============================================================
# ROUGE
# ============================================================

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)


# ============================================================
# BLEU
# ============================================================

smooth = SmoothingFunction().method1

bleu1_scores = []
bleu2_scores = []
bleu3_scores = []
bleu4_scores = []

rouge_scores = []


# ============================================================
# CALCULATE METRICS
# ============================================================

for _, row in df.iterrows():

    reference = str(row["reference"]).lower().split()
    generated = str(row["generated"]).lower().split()

    if len(generated) == 0:
        continue

    references = [reference]

    bleu1 = sentence_bleu(
        references,
        generated,
        weights=(1, 0, 0, 0),
        smoothing_function=smooth
    )

    bleu2 = sentence_bleu(
        references,
        generated,
        weights=(0.5, 0.5, 0, 0),
        smoothing_function=smooth
    )

    bleu3 = sentence_bleu(
        references,
        generated,
        weights=(1/3, 1/3, 1/3, 0),
        smoothing_function=smooth
    )

    bleu4 = sentence_bleu(
        references,
        generated,
        weights=(0.25, 0.25, 0.25, 0.25),
        smoothing_function=smooth
    )

    rouge_result = rouge.score(
        str(row["reference"]),
        str(row["generated"])
    )

    rouge_l = rouge_result["rougeL"].fmeasure

    bleu1_scores.append(bleu1)
    bleu2_scores.append(bleu2)
    bleu3_scores.append(bleu3)
    bleu4_scores.append(bleu4)

    rouge_scores.append(rouge_l)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n========================================")
print("RADIOLOGY REPORT EVALUATION")
print("========================================")

print(
    f"BLEU-1  : {sum(bleu1_scores) / len(bleu1_scores):.4f}"
)

print(
    f"BLEU-2  : {sum(bleu2_scores) / len(bleu2_scores):.4f}"
)

print(
    f"BLEU-3  : {sum(bleu3_scores) / len(bleu3_scores):.4f}"
)

print(
    f"BLEU-4  : {sum(bleu4_scores) / len(bleu4_scores):.4f}"
)

print(
    f"ROUGE-L : {sum(rouge_scores) / len(rouge_scores):.4f}"
)

print("========================================")
print("Evaluation completed successfully.")
print("========================================")