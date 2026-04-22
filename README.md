# Reproducing R-Drop: Consistency Regularization for Improved Generalization in NLP Fine-Tuning

**ECE 570 — Introduction to Machine Learning | Purdue University | Spring 2026**  
**Track 1: TinyReproductions**  
**Paper:** Liang et al., *R-Drop: Regularized Dropout for Neural Networks*, NeurIPS 2021

---

## Overview

This project reproduces the core claim of the R-Drop paper: that penalizing the KL divergence between two dropout-sampled forward passes of the same input reduces the train–validation generalization gap compared to standard fine-tuning, while also improving validation accuracy.

We reproduce this using:
- **Model:** DistilBERT (`distilbert-base-uncased`)
- **Dataset:** SST-2 (binary sentiment classification, GLUE benchmark)
- **Experiments:** Single-epoch comparison (Exp 1), 3-epoch learning curves (Exp 2), α sensitivity sweep (Exp 3)

---

## Repository Structure

```
ECE-570-R-DROP-Project/
│
├── AI_Project.ipynb          # Main notebook — all experiments with outputs
├── README.md                 # This file
└── LICENSE
```

Output figures saved automatically when the notebook runs:
- `experiment2_learning_curves.png` — validation accuracy and generalization gap over 3 epochs
- `experiment3_alpha_sweep.png` — α sensitivity sweep results

---

## Requirements

### Hardware
- **GPU required.** All experiments were run on an NVIDIA T4 GPU (Google Colab).

### Runtime (measured on NVIDIA T4)

| Experiment | Description | Runtime |
|------------|-------------|---------|
| Experiment 1 | 1 epoch × 2 models (baseline + R-Drop) | ~20 min |
| Experiment 2 | 3 epochs × 2 models (baseline + R-Drop α=4) | ~90 min |
| Experiment 3 | 3 epochs × 3 α values (α ∈ {1, 2, 4}) | ~190 min |
| **Total (full notebook)** | All experiments end-to-end | **~6 hours** |

### Software Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | ≥ 2.0 | Core deep learning framework |
| `transformers` | ≥ 4.35 | DistilBERT model and tokenizer |
| `datasets` | ≥ 2.14 | GLUE/SST-2 dataset loading |
| `accelerate` | ≥ 0.24 | Required by transformers training utils |
| `scikit-learn` | ≥ 1.3 | `accuracy_score` metric |
| `matplotlib` | ≥ 3.7 | Result plotting |

---

## How to Run

### Option A — Google Colab (Recommended)

1. Click the "Open in Colab" badge at the top of `AI_Project.ipynb`
2. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
3. Run all cells: **Runtime → Run all**

The notebook installs all dependencies in **Section 1** automatically. Dataset and model weights (~270 MB total) download automatically on first run.

> **Note:** Running the full notebook end-to-end takes approximately **6 hours** on a T4 GPU. If you only want to verify a subset of experiments, you can run Sections 1–3 (Experiment 1, ~20 min) independently before committing to the full run.

### Option B — Local Setup

```bash
# 1. Clone the repository
git clone https://github.com/dohaneedsrest/ECE-570-R-DROP-Project.git
cd ECE-570-R-DROP-Project

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets accelerate scikit-learn matplotlib

# 4. Launch Jupyter
jupyter notebook AI_Project.ipynb
```

---

## Notebook Walkthrough

| Section | Cells | What it does |
|---------|-------|--------------|
| **1 — Setup** | GPU check, pip install, imports | Verifies GPU, installs packages |
| **2 — Data & Baseline** | Dataset loading, tokenization, 1-epoch baseline training | Loads SST-2, trains baseline, defines `evaluate()` |
| **3 — R-Drop Core** | `rdrop_loss`, single-epoch R-Drop, `train_and_track` | Implements loss function and unified training loop |
| **4 — Experiment 2** | 3-epoch baseline vs. R-Drop α=4 | Main multi-epoch comparison with learning curves |
| **5 — Experiment 3** | α sweep (α ∈ {1, 2, 4}) | Hyperparameter sensitivity analysis |
| **6 — Summary** | Consolidated results tables | Final comparison across all experiments |

---

## Results

### Experiment 1 — Single-Epoch Comparison (~20 min)

| Method | Val Accuracy |
|--------|-------------|
| Baseline | 89.45% |
| R-Drop α=4 | **91.28%** (+1.83 pp) |

R-Drop outperforms the baseline after a single epoch, showing an immediate accuracy advantage.

---

### Experiment 2 — 3-Epoch Learning Curves (~90 min)

| Method | Ep1 Val | Ep2 Val | Ep3 Val | Ep1 Gap | Ep2 Gap | Ep3 Gap |
|--------|---------|---------|---------|---------|---------|---------|
| Baseline | 88.99% | 89.68% | 89.45% | 0.027 | 0.064 | 0.080 |
| R-Drop α=4 | **90.25%** | **91.51%** | **90.83%** | **0.012** | **0.043** | **0.063** |

R-Drop reduces the epoch-1 generalization gap by **55% relative** (0.012 vs. 0.027) and outperforms the baseline on validation accuracy at every epoch.

---

### Experiment 3 — α Sensitivity Sweep (~190 min)

| Method | Ep1 Val | Ep2 Val | Ep3 Val | Ep1 Gap | Ep2 Gap | Ep3 Gap |
|--------|---------|---------|---------|---------|---------|---------|
| Baseline | 88.99% | 89.68% | 89.45% | 0.027 | 0.064 | 0.080 |
| R-Drop α=1 | 90.25% | 90.14% | 90.25% | 0.016 | 0.062 | 0.074 |
| R-Drop α=2 | 90.71% | 89.45% | 90.14% | 0.012 | 0.068 | 0.073 |
| R-Drop α=4 | **91.06%** | **91.63%** | **91.74%** | **0.004** | **0.042** | **0.053** |

**Key finding:** α=4 is the optimal regularization weight for SST-2, achieving the highest validation accuracy (91.74% at epoch 3 — best result across all experiments) and the smallest generalization gap at every epoch. All R-Drop variants outperform the baseline.

---

## Code Attribution

| Component | Status | Notes |
|-----------|--------|-------|
| `rdrop_loss()` | Student-written | Implements Eq. 4 from Liang et al. 2021 |
| `train_and_track()` | Student-written | Unified training loop with per-epoch metric tracking |
| `train_one_epoch_rdrop()` | Student-written | Single-epoch two-pass R-Drop training |
| `evaluate()` | Student-written | Inference-mode accuracy computation |
| Dataset loading | Adapted from HuggingFace `datasets` docs | Standard GLUE loading pattern |
| DistilBERT loading | Adapted from HuggingFace `transformers` docs | Standard `AutoModel` pattern |
| Plotting | Student-written | Using matplotlib |

**LLM assistance:** Claude (Anthropic) was used for code structuring suggestions, markdown documentation, and result interpretation. All algorithmic implementations were written and verified by the student. See the final notebook cell for full acknowledgement.

---

## Citation

```bibtex
@inproceedings{liang2021rdrop,
  title     = {R-Drop: Regularized Dropout for Neural Networks},
  author    = {Liang, Xiaobo and Wu, Lijun and Li, Juntao and Wang, Yue and
               Meng, Qi and Qin, Tao and Chen, Wei and Zhang, Min and Liu, Tie-Yan},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2021}
}
```
