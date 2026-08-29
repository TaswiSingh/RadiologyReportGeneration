import pandas as pd

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)

from rouge_score import rouge_scorer


GREEDY_PATH = "evaluation/results.csv"
BEAM_PATH = "evaluation/beam_results.csv"


greedy = pd.read_csv(GREEDY_PATH)
beam = pd.read_csv(BEAM_PATH)


smooth = SmoothingFunction().method1

rouge = rouge_scorer.RougeScorer(
    ["rougeL"],
    use_stemmer=True
)


def calculate_metrics(df):

    bleu_scores = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    rouge_scores = []

    for _, row in df.iterrows():

        reference = str(
            row["reference"]
        ).lower().split()

        generated = str(
            row["generated"]
        ).lower().split()

        if not generated:
            continue

        references = [reference]

        bleu_scores[1].append(
            sentence_bleu(
                references,
                generated,
                weights=(1, 0, 0, 0),
                smoothing_function=smooth
            )
        )

        bleu_scores[2].append(
            sentence_bleu(
                references,
                generated,
                weights=(0.5, 0.5, 0, 0),
                smoothing_function=smooth
            )
        )

        bleu_scores[3].append(
            sentence_bleu(
                references,
                generated,
                weights=(1/3, 1/3, 1/3, 0),
                smoothing_function=smooth
            )
        )

        bleu_scores[4].append(
            sentence_bleu(
                references,
                generated,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth
            )
        )

        rouge_result = rouge.score(
            str(row["reference"]),
            str(row["generated"])
        )

        rouge_scores.append(
            rouge_result["rougeL"].fmeasure
        )

    return {
        "BLEU-1": sum(bleu_scores[1]) / len(bleu_scores[1]),
        "BLEU-2": sum(bleu_scores[2]) / len(bleu_scores[2]),
        "BLEU-3": sum(bleu_scores[3]) / len(bleu_scores[3]),
        "BLEU-4": sum(bleu_scores[4]) / len(bleu_scores[4]),
        "ROUGE-L": sum(rouge_scores) / len(rouge_scores)
    }


print("=" * 60)
print("GREEDY vs IMPROVED BEAM SEARCH")
print("=" * 60)

g = calculate_metrics(greedy)
b = calculate_metrics(beam)

print()
print(f"{'Metric':<12} {'Greedy':>12} {'Beam':>12} {'Difference':>12}")
print("-" * 50)

for metric in g:

    difference = b[metric] - g[metric]

    print(
        f"{metric:<12} "
        f"{g[metric]:>12.4f} "
        f"{b[metric]:>12.4f} "
        f"{difference:>+12.4f}"
    )

print("=" * 60)