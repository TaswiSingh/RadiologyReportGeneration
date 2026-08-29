# RadiologyReportGeneration

A Transformer-based deep learning system for **automatic radiology report generation from chest X-ray images**.

The system takes a chest X-ray image as input and generates a structured radiology report containing **Findings** and **Impression** sections.

---

## Project Overview

Radiology report generation combines computer vision and natural language generation.

The implemented pipeline consists of:

```text
Chest X-ray
     ↓
Visual Encoder
     ↓
Visual Feature Extraction
     ↓
Transformer Text Decoder
     ↓
Generated Radiology Report
```

### Main Components

* **Visual Encoder:** CNN-based image feature extractor
* **Text Decoder:** Transformer decoder
* **Vocabulary:** Custom vocabulary built from the radiology reports
* **Training:** Teacher-forcing based sequence generation
* **Inference:** Greedy decoding and Beam Search
* **Evaluation:** BLEU, ROUGE-L and abnormality detection analysis

---

## Dataset

The project uses the **Indiana University Chest X-ray dataset** containing chest X-ray images and corresponding radiology reports.

The preprocessing pipeline produces:

* `data/train.csv`
* `data/val.csv`
* `data/test.csv`

Images are resized and normalized before being passed to the visual encoder.

### Dataset Split

| Split      | Samples |
| ---------- | ------: |
| Training   |   5,940 |
| Validation |     743 |
| Test       |     743 |

Images are processed at:

```text
224 × 224
```

---

## Model Architecture

### 1. Visual Encoder

The visual encoder converts the chest X-ray into visual representations.

Configuration:

```text
Feature dimension: 512
Input size: 224 × 224
Backbone: pretrained CNN
Backbone: frozen during decoder training
```

The encoder produces visual memory with the shape:

```text
[batch_size, 49, 512]
```

This visual representation is provided to the Transformer decoder through cross-attention.

### 2. Transformer Text Decoder

The text decoder generates the report token by token.

Configuration:

```text
Embedding dimension : 512
Attention heads     : 8
Decoder layers      : 4
Feed-forward size   : 2048
Dropout             : 0.1
Maximum length      : 128
```

Special tokens include:

```text
<pad>
<bos>
<eos>
<unk>
```

### 3. Training

The decoder is trained using **teacher forcing**.

For a report:

```text
<bos> the heart is normal <eos>
```

the model receives:

```text
<bos> the heart is normal
```

and learns to predict:

```text
the heart is normal <eos>
```

The loss function is:

```text
CrossEntropyLoss
```

with padding tokens ignored.

---

## Project Structure

```text
RadiologyReportGeneration/
│
├── app/
│
├── configs/
│   └── config.yaml
│
├── data/
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── vocab.pkl
│
├── images/
│   └── images_normalized/
│
├── checkpoints/
│   └── best_decoder.pt
│
├── evaluation/
│   ├── results.csv
│   ├── greedy_results.csv
│   ├── beam_results.csv
│   ├── abnormality_metrics.csv
│   ├── abnormality_metrics_improved.csv
│   ├── abnormality_metrics_negation.csv
│   ├── analyzed_results.csv
│   └── missed_abnormalities.csv
│
├── src/
│   ├── dataset.py
│   ├── train.py
│   ├── inference.py
│   ├── inference_beam.py
│   ├── evaluate.py
│   ├── evaluate_beam.py
│   ├── evaluate_metrics.py
│   ├── abnormality_metrics.py
│   ├── abnormality_metrics_negation.py
│   ├── classify_reports.py
│   ├── analyze_results.py
│   └── analyze_missed_abnormalities.py
│
├── check_distribution.py
├── compare_metrics.py
├── final_evaluation.py
│
└── README.md
```

---

## Training Configuration

The implemented training pipeline uses:

```text
Batch size       : 2
Learning rate    : 1e-4
Weight decay     : 1e-4
Maximum length   : 128
Device           : CPU
```

The visual encoder is frozen and the Transformer decoder is trained.

The available checkpoint is:

```text
checkpoints/best_decoder.pt
```

Because the development environment is CPU-only, the final evaluation was performed using the available trained checkpoint rather than performing a long full retraining experiment.

---

## Inference

### Greedy Decoding

Greedy decoding selects the highest-probability token at every generation step.

Run:

```powershell
.\venv\Scripts\python.exe src\inference.py
```

### Beam Search

Beam Search maintains multiple candidate sequences during generation.

The implemented experiment uses:

```text
Beam size = 3
```

Run:

```powershell
.\venv\Scripts\python.exe src\inference_beam.py
```

---

## Evaluation

The model was evaluated on **100 test samples**.

### Text Generation Metrics

| Metric  |  Score |
| ------- | -----: |
| BLEU-1  | 0.2994 |
| BLEU-2  | 0.1881 |
| BLEU-3  | 0.1254 |
| BLEU-4  | 0.0826 |
| ROUGE-L | 0.3534 |

Run:

```powershell
.\venv\Scripts\python.exe src\evaluate_metrics.py
```

---

## Greedy vs Beam Search

The two decoding strategies were compared on the same evaluation samples.

| Metric  | Greedy |   Beam |
| ------- | -----: | -----: |
| BLEU-1  | 0.2994 | 0.2701 |
| BLEU-2  | 0.1881 | 0.1627 |
| BLEU-3  | 0.1254 | 0.1106 |
| BLEU-4  | 0.0826 | 0.0732 |
| ROUGE-L | 0.3534 | 0.3471 |

### Generation Statistics

| Statistic                     | Greedy |  Beam |
| ----------------------------- | -----: | ----: |
| Average words                 |  29.34 | 24.11 |
| Repeated pneumothorax reports |      2 |     0 |
| Evaluation samples            |    100 |   100 |

In this experiment, **greedy decoding achieved higher BLEU and ROUGE-L scores**.

The improved Beam Search produced shorter reports and reduced the observed repetition problem, but its text-overlap metrics were slightly lower.

---

## Abnormality Analysis

Additional evaluation was performed to determine whether generated reports contained clinically important abnormalities present in the reference reports.

The analysis included:

```text
nodule
opacity
atelectasis
effusion
pneumonia
consolidation
cardiomegaly
pneumothorax
fracture
mass
edema
infiltrate
granuloma
scoliosis
```

The analysis showed that the current model performs substantially better on common findings such as **effusion and pneumothorax** than on several less frequently generated abnormalities.

The negation-aware analysis also showed that the model frequently defaults to normal findings rather than explicitly describing abnormalities.

---

## Important Limitation

A major limitation of the current system is **abnormality under-generation**.

The generated reports frequently contain patterns such as:

```text
the lungs are clear.
no pleural effusion or pneumothorax.
no acute cardiopulmonary abnormalities.
```

even when the reference report contains clinically relevant abnormalities.

This indicates that the current model has learned strong patterns from common/normal radiology reports but has not learned sufficiently robust image-to-abnormality associations.

Therefore, the generated reports should **not be considered clinically reliable diagnostic reports**.

The current implementation is intended as a research and educational project demonstrating multimodal image-to-text generation.

---

## Evaluation Scripts

### General Evaluation

```powershell
.\venv\Scripts\python.exe src\evaluate.py
```

### BLEU / ROUGE

```powershell
.\venv\Scripts\python.exe src\evaluate_metrics.py
```

### Beam Search Evaluation

```powershell
.\venv\Scripts\python.exe src\evaluate_beam.py
```

### Abnormality Analysis

```powershell
.\venv\Scripts\python.exe src\abnormality_metrics.py
```

### Negation-Aware Analysis

```powershell
.\venv\Scripts\python.exe src\abnormality_metrics_negation.py
```

### Final Evaluation Summary

```powershell
.\venv\Scripts\python.exe final_evaluation.py
```

---

## Reproducibility

Install the required Python packages and activate the virtual environment.

Example:

```powershell
.\venv\Scripts\Activate.ps1
```

Then run the required scripts from the project root.

The project currently targets a CPU environment, so inference and evaluation may take longer than on a GPU.

---

## Future Improvements

Potential improvements include:

1. Training for more epochs on GPU hardware.
2. Fine-tuning the visual encoder instead of freezing it.
3. Using a stronger pretrained vision backbone.
4. Adding attention mechanisms specifically optimized for radiology.
5. Addressing class imbalance between normal and abnormal reports.
6. Improving abnormality-aware loss functions.
7. Adding medical concept extraction during training.
8. Using clinically oriented metrics such as CheXbert-based evaluation.
9. Testing on a larger held-out test set.
10. Improving decoding with repetition penalties and constrained generation.
11. Comparing Transformer decoders with pretrained medical language models.
12. Evaluating factual correctness rather than relying only on text-overlap metrics.

---

## Conclusion

The project demonstrates an end-to-end **chest X-ray to radiology report generation pipeline** using a pretrained visual encoder and Transformer-based text decoder.

The implemented system successfully performs:

```text
Image preprocessing
        ↓
Visual feature extraction
        ↓
Transformer-based report generation
        ↓
Greedy / Beam Search inference
        ↓
BLEU / ROUGE evaluation
        ↓
Abnormality analysis
```

The current results demonstrate that the model can generate structured radiology-style text, while the abnormality analysis highlights the main limitation of the current training setup: **difficulty in reliably generating clinically important abnormalities**.

This provides a clear foundation for future work involving better visual-language alignment, abnormality-aware training, stronger pretrained models, and clinically focused evaluation.
