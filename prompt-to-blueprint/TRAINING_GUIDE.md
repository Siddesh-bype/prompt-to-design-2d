# Training Guide — Prompt-to-Blueprint (All Models)

All four trainable components can use the **CubiCasa5K** and **Modified Swiss Dwellings** Kaggle datasets.

---

## 1. Prerequisites

```bash
cd prompt-to-blueprint/backend
pip install -r requirements.txt
```

## 2. Dataset Setup

### CubiCasa5K (5000 annotated floor plans)

1. Download from [Kaggle](https://www.kaggle.com/datasets/qmarva/cubicasa5k)
2. Extract to `data/cubicasa5k/`:
   ```
   data/cubicasa5k/
   ├── train.txt / val.txt / test.txt
   └── cubicasa5k/<sample_id>/model.svg
   ```

### Modified Swiss Dwellings (graph-based floor plans)

1. Download from [Kaggle](https://www.kaggle.com/datasets/caspervanengelenburg/modified-swiss-dwellings)
2. Extract to `data/modified_swiss_dwellings/`:
   ```
   data/modified_swiss_dwellings/
   └── graphs/*.pkl
   ```

---

## 3. Train ALL Models (One Command)

```bash
# Train everything on both datasets
python -m app.services.train_all --dataset combined --data-dir data --epochs 100

# Train specific models only
python -m app.services.train_all --dataset combined --models gnn nlp constraints
```

---

## 4. Train Individual Models

### A. GNN Layout Engine (`FloorPlanGNN`)

The graph neural network that generates room bounding boxes from room specifications.

```bash
# CubiCasa5K only
python -m app.services.train_gnn --dataset cubicasa --data-dir data/cubicasa5k --epochs 100

# Swiss Dwellings only
python -m app.services.train_gnn --dataset swiss --data-dir data/modified_swiss_dwellings --epochs 100

# Combined
python -m app.services.train_gnn --dataset combined --data-dir data --epochs 150 --lr 0.0005
```

**Output:** `models/floorplan_gnn.pt`

### B. NLP Parser (Ollama Fine-tuning)

Generates prompt/response training pairs from real layouts for the Ollama SLM.

```bash
# Generate fine-tuning data
python -m app.services.train_nlp --dataset combined --data-dir data --augment 3

# Also output an Ollama Modelfile
python -m app.services.train_nlp --dataset combined --data-dir data --modelfile
```

Then fine-tune Ollama:
```bash
ollama create floorplan-parser -f data/training/Modelfile
```

**Output:** `data/training/nlp_finetune.jsonl`, `data/training/Modelfile`

### C. Constraint Weights Optimiser

Learns optimal soft constraint weights (natural light, corridor area, kitchen distance, etc.) from real layouts.

```bash
python -m app.services.train_constraints --dataset combined --data-dir data --iterations 200
```

**Output:** `models/constraint_weights.json`

### D. Vastu Rule Calibration

Analyses room placement zones across real layouts to calibrate Vastu scoring. Runs automatically via `train_all`.

---

## 5. Recommended Strategy

| Phase | Command | Purpose |
|-------|---------|---------|
| **Pre-train** | `--dataset synthetic --samples 5000 --epochs 50` | Learn basic patterns |
| **Fine-tune** | `--dataset combined --epochs 100 --lr 0.0005 --resume models/floorplan_gnn.pt` | Real-world quality |
| **NLP data** | `python -m app.services.train_nlp --dataset combined --modelfile` | Ollama fine-tuning |
| **Calibrate** | `python -m app.services.train_constraints --dataset combined` | Optimal weights |

```bash
# Full recommended pipeline
python -m app.services.train_gnn --dataset synthetic --samples 5000 --epochs 50
python -m app.services.train_all --dataset combined --data-dir data --epochs 100 --lr 0.0005
```

---

## 6. Output Files

| File | Model | Description |
|------|-------|-------------|
| `models/floorplan_gnn.pt` | GNN | Best model weights |
| `models/checkpoint_epoch_*.pt` | GNN | Periodic checkpoints |
| `models/constraint_weights.json` | Constraints | Optimised weights |
| `models/vastu_calibration.json` | Vastu | Zone frequency stats |
| `data/training/nlp_finetune.jsonl` | NLP | Fine-tuning pairs |
| `data/training/Modelfile` | NLP | Ollama model definition |

---

## 7. CLI Reference

### `train_all.py`
| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | `combined` | `synthetic`, `cubicasa`, `swiss`, `combined` |
| `--data-dir` | `data` | Root data directory |
| `--output` | `models` | Output directory |
| `--epochs` | `50` | GNN epochs |
| `--lr` | `0.001` | GNN learning rate |
| `--models` | all | `gnn nlp constraints vastu` |
| `--max-per-source` | unlimited | Cap samples per source |
