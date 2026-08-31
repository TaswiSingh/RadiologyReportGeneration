import re
import pandas as pd


RESULTS_PATH = "evaluation/results.csv"
BEAM_RESULTS_PATH = "evaluation/beam_results.csv"


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


NEGATIONS = [
    "no",
    "not",
    "without",
    "negative for",
    "free of",
    "absence of",
    "absent",
    "ruled out",
]


def is_present(text, abnormality):
    """
    Detect whether an abnormality is positively mentioned.

    A mention is considered positive if it is not preceded
    by a recognized negation within the same sentence.
    """

    text = str(text).lower()

    matches = list(
        re.finditer(
            r"\b" + re.escape(abnormality) + r"\b",
            text
        )
    )

    for match in matches:

        sentence_start = max(
            text.rfind(".", 0, match.start()),
            text.rfind("!", 0, match.start()),
            text.rfind("?", 0, match.start())
        )

        context = text[
            sentence_start + 1:
            match.start()
        ]

        negated = False

        for negation in NEGATIONS:
            if re.search(
                r"\b" + re.escape(negation) + r"\b",
                context
            ):
                negated = True
                break

        if not negated:
            return True

    return False


def evaluate_abnormalities(df, name):

    rows = []

    for abnormality in ABNORMALITIES:

        reference_present = df["reference"].apply(
            lambda x: is_present(x, abnormality)
        )

        generated_present = df["generated"].apply(
            lambda x: is_present(x, abnormality)
        )

        tp = int(
            (reference_present & generated_present).sum()
        )

        fn = int(
            (reference_present & ~generated_present).sum()
        )

        fp = int(
            (~reference_present & generated_present).sum()
        )

        tn = int(
            (~reference_present & ~generated_present).sum()
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        rows.append({
            "abnormality": abnormality,
            "reference_present": int(
                reference_present.sum()
            ),
            "generated_present": int(
                generated_present.sum()
            ),
            "true_positive": tp,
            "false_negative": fn,
            "false_positive": fp,
            "true_negative": tn,
            "recall": recall,
            "precision": precision,
        })

    result = pd.DataFrame(rows)

    macro_recall = result["recall"].mean()
    macro_precision = result["precision"].mean()

    total_tp = result["true_positive"].sum()
    total_fn = result["false_negative"].sum()
    total_fp = result["false_positive"].sum()

    micro_recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    micro_precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    print()
    print("=" * 72)
    print(name)
    print("=" * 72)

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

    print(
        "Macro Recall    :",
        f"{macro_recall * 100:.2f}%"
    )

    print(
        "Macro Precision :",
        f"{macro_precision * 100:.2f}%"
    )

    print(
        "Micro Recall    :",
        f"{micro_recall * 100:.2f}%"
    )

    print(
        "Micro Precision :",
        f"{micro_precision * 100:.2f}%"
    )

    return result


def evaluate_text_statistics(df, name):

    reference_lengths = df["reference"].astype(str).apply(
        lambda x: len(x.split())
    )

    generated_lengths = df["generated"].astype(str).apply(
        lambda x: len(x.split())
    )

    empty_generations = (
        generated_lengths == 0
    ).sum()

    print()
    print("=" * 72)
    print(name + " TEXT STATISTICS")
    print("=" * 72)

    print(
        "Samples:",
        len(df)
    )

    print(
        "Average reference length:",
        f"{reference_lengths.mean():.2f}"
    )

    print(
        "Average generated length:",
        f"{generated_lengths.mean():.2f}"
    )

    print(
        "Minimum generated length:",
        int(generated_lengths.min())
    )

    print(
        "Maximum generated length:",
        int(generated_lengths.max())
    )

    print(
        "Empty generations:",
        int(empty_generations)
    )


def main():

    print("=" * 72)
    print("PERSON 2 — INDEPENDENT EVALUATION")
    print("=" * 72)

    print()
    print("Loading greedy results...")
    greedy = pd.read_csv(RESULTS_PATH)

    print(
        "Greedy samples:",
        len(greedy)
    )

    print()
    print("Loading beam results...")
    beam = pd.read_csv(BEAM_RESULTS_PATH)

    print(
        "Beam samples:",
        len(beam)
    )

    evaluate_text_statistics(
        greedy,
        "GREEDY"
    )

    evaluate_text_statistics(
        beam,
        "BEAM"
    )

    greedy_metrics = evaluate_abnormalities(
        greedy,
        "GREEDY — NEGATION-AWARE ABNORMALITY EVALUATION"
    )

    beam_metrics = evaluate_abnormalities(
        beam,
        "BEAM — NEGATION-AWARE ABNORMALITY EVALUATION"
    )

    greedy_metrics.to_csv(
        "evaluation/person2_greedy_abnormality_metrics.csv",
        index=False
    )

    beam_metrics.to_csv(
        "evaluation/person2_beam_abnormality_metrics.csv",
        index=False
    )

    print()
    print("Saved:")
    print(
        "evaluation/person2_greedy_abnormality_metrics.csv"
    )
    print(
        "evaluation/person2_beam_abnormality_metrics.csv"
    )


if __name__ == "__main__":
    main()