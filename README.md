# Improving AlexNet on CIFAR-10: Ablations on Data Augmentation, Weight Initialization, and Skip Connections

> **Course:** CSCI-567 Machine Learning · University of Southern California
> **Stack:** PyTorch · NumPy · scikit-learn · Matplotlib · CUDA (NVIDIA L4, RunPod)
> **Highlights:** Built AlexNet from scratch, ran a **90-configuration controlled ablation study** (3 seeds × 10 settings × 3 axes), and pushed test accuracy from a **79.63%** baseline to **83.18%** — a **+3.55 pp** absolute gain — with a single, well-chosen augmentation.

---

## TL;DR

A rigorous, reproducible study of _what actually moves the needle_ when modernizing a classical CNN. We isolate the contribution of three orthogonal training/architecture choices on CIFAR-10:

| Variant                                             | Test Accuracy      | Δ vs. Baseline  |
| --------------------------------------------------- | ------------------ | --------------- |
| **Baseline AlexNet** (random init, no augmentation) | 79.63 ± 1.64 %     | —               |
| + Random Rotation                                   | 82.95 ± 0.34 %     | **+3.32**       |
| **+ Random Flip (best)**                            | **83.18 ± 1.03 %** | **+3.55**       |
| + Color Jitter                                      | 82.34 ± 0.60 %     | +2.71           |
| + Random Sharpness                                  | 82.17 ± 1.18 %     | +2.54           |
| + Random Erasing                                    | 82.82 ± 0.27 %     | +3.19           |
| + All Augmentations (stacked)                       | 82.53 ± 1.23 %     | +2.90           |
| + Skip Connections                                  | 79.71 ± 1.22 %     | +0.08 _(noise)_ |
| + Xavier Init                                       | 82.11 ± 0.45 %     | +2.48           |
| + He Init                                           | 81.33 ± 1.12 %     | +1.70           |

**Key findings.** (1) Lightweight augmentation is the highest-ROI lever — a single horizontal flip beats a full augmentation stack. (2) Stacking augmentations introduces conflicting transformations and _underperforms_ the best individual one. (3) ResNet-style skip connections produce **no statistically meaningful gain** in a shallow 5-conv backbone — a useful negative result. (4) Xavier / He initialization speeds early convergence (≤ 10 epochs) but does _not_ improve final accuracy.

📄 Full write-up: [Improving AlexNet — full report (PDF)](Improving%20AlexNet.%20Evaluating%20the%20Impact%20of%20Data%20Augmentation%2C%20Initialization%2C%20and%20Skip%20Connections%20-%20report.pdf)

---

## What This Project Demonstrates (for AI Engineer reviewers)

- **Experimental rigor.** Every result is reported as **mean ± std over 3 seeds** (`1, 42, 99`) with a clean 9:1 train/val split and a held-out test set. Every modification is studied **in isolation** to attribute effects unambiguously.
- **End-to-end PyTorch engineering.** Custom `nn.Module` implementations, configurable initialization hooks (`model.apply(...)`), pluggable `torchvision.transforms` pipelines, and a CLI-driven training entry point.
- **Reproducibility.** Deterministic seeds, pinned Conda environment ([environment.yml](environment.yml)), shell-script-driven sweeps, and per-run JSON/CSV logs in [logs/](logs/).
- **Compute-cost awareness.** Sweep was parallelized across 4 concurrent shell scripts on a single NVIDIA L4 (24 GB VRAM) — **~10 h wall-clock vs. ~40 h serial**, ~$10 total compute spend on RunPod.
- **Honest reporting.** We publish the _negative_ result for skip connections rather than discarding it — the kind of signal hiring managers want to see in a research-engineering hire.

---

## Methodology Snapshot

### 1. Architecture

Faithful PyTorch reimplementation of AlexNet (5 conv blocks → 3 FC layers, BatchNorm + ReLU + MaxPool, dropout 0.5 in FC). See [src/model.py](src/model.py).

### 2. Skip-Connection Variant

`AlexNetWithSkipConnections` adds two residual paths:

- **Skip 1:** input → output of conv block 1, with a `1×1` channel-projection conv and bilinear spatial alignment.
- **Skip 2:** output of conv block 2 → output of conv block 3, with a `1×1` projection.

### 3. Initialization Hooks

Three swappable initializers wired through `model.apply(...)` — see [src/init_method.py](src/init_method.py):

- **Random** — $W \sim \mathcal{N}(0, 0.02)$
- **Xavier (Glorot uniform)** — $W \sim \mathrm{Uniform}\!\left(-\sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}}, \sqrt{\tfrac{6}{n_{\text{in}}+n_{\text{out}}}}\right)$
- **He (Kaiming uniform, ReLU-tuned)** — variance scaled by $\tfrac{2}{n_{\text{in}}}$

### 4. Data Augmentation Catalog

Seven configurable pipelines compiled at runtime from a declarative dict — see [src/data_augmentation.py](src/data_augmentation.py): resize-only, random rotation (±20°), horizontal flip (p=0.1), color jitter, random sharpness, random erasing (p=0.75), and a stacked combination.

### 5. Training Recipe

SGD, lr=0.005, momentum=0.9, weight decay=0.005, batch size=128, 15 epochs. Best epoch selected by validation accuracy; final number reported on the held-out CIFAR-10 test set.

---

## Repository Layout

```
csci-567-project/
├── src/
│   ├── model.py                  # AlexNet + AlexNetWithSkipConnections
│   ├── init_method.py            # Random / Xavier / He initializers
│   ├── data_augmentation.py      # Declarative transform configs (1–7)
│   ├── run_model.py              # CLI training/eval entry point
│   └── run.sh, run{1..4}.sh      # Parallelizable experiment sweeps
├── logs/
│   ├── trainval/                 # Per-epoch train/val metrics (JSON)
│   └── test/                     # Final test metrics + confusion matrices
├── models/                       # Best checkpoints (per configuration)
├── visualize.py                  # Confusion-matrix & convergence plots
├── arxiv/                        # Reference notebooks (AlexNet baseline)
├── Cifar10/                      # Dataset cache
├── environment.yml               # Conda environment pin
└── Improving AlexNet ... .pdf    # Full technical report
```

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/NicklasHolmberg/csci-567-project.git
cd csci-567-project

# 2. Environment (Miniconda: https://docs.anaconda.com/miniconda/install/)
conda env create -f environment.yml
conda activate cs567

# 3. Run a single configuration
python src/run_model.py \
    --random_seed 42 \
    --init_method he \
    --augment 3 \
    --skip_connection no

# 4. Reproduce the full sweep (90 runs, ~40 h on a single L4)
bash src/run.sh
# …or shard across 4 processes/GPUs:
bash src/run1.sh & bash src/run2.sh & bash src/run3.sh & bash src/run4.sh & wait
```

CLI flags for `src/run_model.py`:

| Flag                | Values                   | Meaning                                          |
| ------------------- | ------------------------ | ------------------------------------------------ |
| `--random_seed`     | `1`, `42`, `99`          | Reproducibility seed                             |
| `--init_method`     | `random`, `xavier`, `he` | Weight initialization                            |
| `--augment`         | `1`–`7`                  | Augmentation config (see `data_augmentation.py`) |
| `--skip_connection` | `yes`, `no`              | Use `AlexNetWithSkipConnections`                 |

Outputs land in `logs/trainval/`, `logs/test/`, and `models/`.

---

## Selected Insights

**Augmentation is the highest-ROI lever.** Random flip alone yields the best single-axis improvement (+3.55 pp). Per-class confusion analysis (Appendix A of the report) shows `cat` and `dog` remain the consistently hardest classes — a clear signal that **class-aware** preprocessing or feature extraction is the next ROI frontier.

**Skip connections don't help shallow networks.** With only 5 conv blocks, AlexNet doesn't suffer from the gradient pathologies that motivated ResNet. Adding skips left accuracy statistically unchanged (79.71 vs. 79.63). Lesson: match architectural priors to network depth, not to hype.

**Initialization is a convergence story, not an accuracy story.** Xavier and He converge measurably faster in the first ~10 epochs (Figure 3 of the report), but by epoch 15 all three initializers land in the same basin. For short training budgets or production fine-tuning, He init still pays for itself; for full convergence it doesn't.

---

## My Contributions

- Authored and ran **Experiment 3 (initialization methods)** end-to-end: implementation, sweep design, convergence-curve analysis, write-up.
- Co-designed the augmentation ablation protocol and the per-class confusion-matrix analysis pipeline.
- Contributed to model code, logging infrastructure, and reproducibility tooling (seeded runs, environment pinning, CLI surface).

---

## Citation

> Merchant, A., **Park, J.**, Fleischer, J., & Holmberg, N. (2024). _Improving AlexNet: Evaluating the Impact of Data Augmentation, Initialization, and Skip Connections._ CSCI-567 Final Project, University of Southern California.

Key references: He et al. 2015/2016 · Glorot & Bengio 2010 · Krizhevsky et al. 2012 · Rebuffi et al. 2021 (full bibliography in the PDF).

---

## License

Academic / educational use.
