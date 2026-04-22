# Reproducing R-Drop: Consistency Regularization for Improved Generalization in NLP Fine-Tuning

**ECE 570 — Introduction to Machine Learning | Purdue University | Spring 2026**  
**Track 1: TinyReproductions**  
**Paper:** Liang et al., *R-Drop: Regularized Dropout for Neural Networks*, NeurIPS 2021

---

## Overview

This project reproduces the core claim of the R-Drop paper: that penalizing the KL divergence between two dropout-sampled forward passes of the same input reduces the train–validation generalization gap compared to standard fine-tuning.

We reproduce this at small scale using:
- **Model:** DistilBERT (`distilbert-base-uncased`)
- **Dataset:** SST-2 (binary sentiment classification, GLUE benchmark)
- **Experiments:** Single-epoch comparison, 3-epoch learning curves, α sensitivity sweep

---

## Repository Structure

```
ECE-570-R-DROP-Project/
│
├── AI_Project.ipynb          # Main notebook — all experiments
├── README.md                 # This file
└── LICENSE
```

All figures are saved automatically when the notebook runs:
- `experiment2_learning_curves.png` — validation accuracy and generalization gap over 3 epochs
- `experiment3_alpha_sweep.png` — α sensitivity sweep results

---

## Requirements

### Hardware
- **GPU required.** All experiments were run on an NVIDIA T4 GPU (Google Colab). Expected runtime:
  - Experiment 1 (1 epoch × 2 runs): ~20 min
  - Experiment 2 (3 epochs × 2 runs): ~90 min
  - Experiment 3 (3 epochs × 3 α values): ~190 min
  - **Total: ~85 minutes on a T4 GPU**

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

1. Open [Google Colab](https://colab.research.google.com/)
2. Go to **File → Open notebook → GitHub**
3. Paste the repo URL: `https://github.com/dohaneedsrest/ECE-570-R-DROP-Project`
4. Select `AI_Project.ipynb`
5. Set runtime to GPU: **Runtime → Change runtime type → T4 GPU**
6. Run all cells: **Runtime → Run all**

The notebook installs all dependencies in **Section 0** automatically — no manual pip installs needed.

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

> **Note:** The dataset (SST-2 via HuggingFace `datasets`) and model weights (DistilBERT via HuggingFace `transformers`) are downloaded automatically on first run. An internet connection is required. Total download size is approximately 270 MB.

---

## Notebook Walkthrough

The notebook is organized into 6 sections, designed to run top-to-bottom:

| Section | Cells | What it does |
|---------|-------|--------------|
| **0 — Setup** | GPU check, pip install, imports | Verifies GPU, installs all packages, sets random seed |
| **1 — Data & Model** | Dataset loading, tokenization, evaluation function | Loads SST-2, tokenizes to length 128, defines `evaluate()` and `make_model()` |
| **2 — R-Drop Core** | `rdrop_loss`, `train_and_track`, `run_experiment` | Implements the R-Drop loss (Eq. 4 from paper) and a unified training loop used by all experiments |
| **3 — Experiment 1** | 1-epoch baseline vs. R-Drop | Replicates the CP1 checkpoint result; confirms implementation is correct |
| **4 — Experiment 2** | 3-epoch learning curves | Main experiment: tracks val accuracy and generalization gap per epoch for both methods |
| **5 — Experiment 3** | α sweep (α ∈ {1, 2, 4}) | Explores sensitivity to the KL regularization weight |
| **6 — Summary** | Final results table + LLM acknowledgement | Consolidates all results and states the core finding |

---

## Expected Results

Based on runs conducted on a Colab T4 GPU:

**Experiment 1 (1 epoch):**
| Method | Val Acc | Gen Gap |
|--------|---------|---------|
| Baseline | ~0.8956 | ~0.021 |
| R-Drop α=4 | ~0.9117 | ~0.004 |

**Experiment 2 (3 epochs):**
| Method | Ep1 Val | Ep2 Val | Ep3 Val | Ep1 Gap | Ep2 Gap | Ep3 Gap |
|--------|---------|---------|---------|---------|---------|---------|
| Baseline | 0.8956 | 0.8991 | 0.8956 | 0.021 | 0.062 | 0.078 |
| R-Drop α=4 | 0.9117 | 0.9117 | 0.9002 | 0.004 | 0.047 | 0.071 |

> Minor variations in accuracy (±0.005) are expected due to GPU non-determinism. The generalization gap trend should be consistent across runs.

**Core finding:** R-Drop reduces the epoch-1 generalization gap by ~80% relative to baseline (0.004 vs. 0.021), reproducing the paper's central regularization claim.

---

## Code Attribution

| Component | Status | Notes |
|-----------|--------|-------|
| `rdrop_loss()` | Student-written | Implements Eq. 4 from Liang et al. 2021 |
| `train_and_track()` | Student-written | Unified training loop with per-epoch metric tracking |
| `run_experiment()` | Student-written | Experiment runner helper |
| `make_model()` | Student-written | Wraps HuggingFace `from_pretrained` for clean experiment isolation |
| `evaluate()` | Student-written | Inference-mode accuracy computation |
| Dataset loading | Adapted from HuggingFace `datasets` documentation | Standard GLUE loading pattern |
| DistilBERT loading | Adapted from HuggingFace `transformers` documentation | Standard `AutoModel` pattern |
| Plotting code | Student-written | Using matplotlib |

**LLM assistance:** Claude (Anthropic) was used for code structuring suggestions and documentation. All algorithmic implementations were written and verified by the student. See the final notebook cell for full acknowledgement.

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
