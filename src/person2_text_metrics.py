import pandas as pd

from nltk.translate.bleu_score import (
    sentence_bleu,
    SmoothingFunction
)

from rouge_score import rouge_scorer


GREEDY_PATH = "evaluation/results.csv"
BEAM_PATH = "evaluation/beam_results.csv"


def calculate_metrics(df):
    smooth = SmoothingFunction().method1

    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    bleu_scores = {
        1: [],
        2: [],
        3: [],
        4: []
    }

    rouge_scores = []

    for _, row in df.iterrows():

        reference = str(row["reference"]).lower().split()
        generated = str(row["generated"]).lower().split()

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


def main():

    print("=" * 70)
    print("PERSON 2 — INDEPENDENT TEXT METRICS")
    print("=" * 70)

    print("\nLoading greedy results...")
    greedy = pd.read_csv(GREEDY_PATH)
    print("Greedy samples:", len(greedy))

    print("\nLoading beam results...")
    beam = pd.read_csv(BEAM_PATH)
    print("Beam samples:", len(beam))

    greedy_metrics = calculate_metrics(greedy)
    beam_metrics = calculate_metrics(beam)

    print("\n" + "=" * 70)
    print("GREEDY vs BEAM")
    print("=" * 70)

    print(
        f"{'Metric':<15}"
        f"{'Greedy':>12}"
        f"{'Beam':>12}"
    )

    print("-" * 40)

    for metric in greedy_metrics:
        print(
            f"{metric:<15}"
            f"{greedy_metrics[metric]:>12.4f}"
            f"{beam_metrics[metric]:>12.4f}"
        )

    results = pd.DataFrame({
        "metric": list(greedy_metrics.keys()),
        "greedy": list(greedy_metrics.values()),
        "beam": list(beam_metrics.values())
    })

    results.to_csv(
        "evaluation/person2_text_metrics.csv",
        index=False
    )

    print("\nSaved:")
    print("evaluation/person2_text_metrics.csv")


if __name__ == "__main__":
    main()