# Table Representation Retrieval

Code for **"Robust Table Retrieval Under Serialization Shift via Centroid-Aligned Adapters"**.

The project measures how serialization format (CSV, TSV, HTML, Markdown, JSON, etc.) shifts table embeddings across retriever families and introduces a lightweight residual bottleneck adapter that transports single-format embeddings toward a centroid target, improving retrieval robustness without retraining the base encoder.

---

## Repository Layout

```
.
├── adapter/
│   ├── train_universal_adapter.py        # adapter training entry point
│   ├── test_universal_adapter.py         # Recall@K evaluation with adapter
│   └── test_rank_universal_adapter.py    # per-question rank export with adapter
├── analysis/
│   ├── create_embedding_visualization.py # PCA plots of serialization orbits
│   └── representation_drift_all.py       # cross-format drift statistics
├── data/
│   └── dataset_all/                      # populated by setup steps below
│       ├── new_all_questions_dict_wtq_nqt_wikisql.pkl
│       └── all_table_transform_table_corpus.pkl
├── figure/
│   ├── figure_adapter.ipynb
│   ├── figure_adapter_ranking.ipynb
│   ├── figure_all_evaluation.ipynb
│   ├── figure_centroid_ranking.ipynb
│   ├── visualize_embedding_adapter.ipynb
│   └── visualize_original_representation.ipynb
├── representations/
│   └── create_representation.py          # generates all 17 serialization formats
├── retrieval/
│   ├── retrieval_all_multiple_dataset.py         # single-format baseline retrieval
│   ├── retrieval_combination_multiple_dataset.py # centroid combination retrieval
│   └── retrieval_rank_export.py                  # per-question gold rank export
├── utils/
│   ├── Adapter.py              # BottleneckResidualAdapter and ResidualCentroidAdapter
│   ├── cache_embedding.py      # cache_exists / load_cache_mmap utilities
│   ├── core.py                 # get_data, get_table_corpus, compute_and_cache_*, batched_topk
│   ├── dataset.py              # CacheMultiViewDatasetGiant, collate_fn
│   ├── losses.py               # invariance, variance, covariance, identity loss terms
│   ├── model.py                # get_model factory (MPNet, BGE-M3, ReasonIR, SPLADE, Jina)
│   ├── table_representation.py # all 17 serialization implementations
│   └── TrainerConfig.py        # TrainCfg dataclass
├── requirements.txt
└── README.md
```

---

## 1. Installation

Python 3.10 or later is recommended.

```bash
git clone https://github.com/KBhandari11/Centroid-Aligned-Table-Retrieval.git
cd Centroid-Aligned-Table-Retrieval
pip install -r requirements.txt
```

---

## 2. Data Setup

The pipeline expects two pre-built pickle files under `data/dataset_all/`. These are derived from WTQ, WikiSQL, and NQ-Tables and contain the questions dict and the serialized table corpus across all 17 formats. [Download these files.](https://drive.google.com/drive/folders/1zEbt6m--XYjoiEGr8D8t0RjYtC_v02Ku?usp=sharing)

```
data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl
data/dataset_all/all_table_transform_table_corpus.pkl
```

Place them at those paths before running any retrieval or training script. The table corpus pickle should map dataset keys (`"WTQ"`, `"WIKISQL"`, `"NQ"`) to dicts of `{table_id: {rep_name: serialized_string}}`. The questions pickle should map dataset keys to dicts of `{question_id: {"question": str, "gold_table_id": str}}`.

To build these pickles from raw benchmark data, run:

```bash
python representations/create_representation.py
```

---

## 3. Build the Embedding Cache

Embeddings are pre-computed once and stored as memory-mapped files via `utils/cache_embedding.py`. Retrieval and training scripts build the cache on demand at the path configured in `utils/core.py` as `CACHE_ROOT`. Set that variable to a directory with sufficient disk space before running.

Retrieval scripts call `compute_and_cache_rep_embeddings` and `compute_and_cache_centroid` internally on cache miss, so no separate cache-build step is required. On the first run for a given model and dataset, expect a significant one-time encoding cost.

---

## 4. Retrieval Baselines

All retrieval scripts take the model name as the first positional argument. Results write to `./data/retrieval_all/{model_name}_results/{dataset}/perturbation_results.json`.

**Single-format retrieval across all 17 serializations and 5 centroid variants:**

```bash
CUDA_VISIBLE_DEVICES=1 python -m retrieval.retrieval_all_multiple_dataset reasonir
CUDA_VISIBLE_DEVICES=1 python -m retrieval.retrieval_all_multiple_dataset bge
CUDA_VISIBLE_DEVICES=1 python -m retrieval.retrieval_all_multiple_dataset mpnet
CUDA_VISIBLE_DEVICES=1 python -m retrieval.retrieval_all_multiple_dataset splade
```

**Custom centroid combination** (any subset of the 17 base representations):

```bash
# Args: model_name rep1 rep2 [rep3 ...]
# Valid rep names are listed in REPRESENTATION inside the script.
CUDA_VISIBLE_DEVICES=1 python -m retrieval.retrieval_combination_multiple_dataset mpnet csv tsv pipe_serialized space_serialized transpose
CUDA_VISIBLE_DEVICES=1 python -m retrieval.retrieval_combination_multiple_dataset splade csv tsv pipe_serialized space_serialized xml latex
```

Results for each combination are appended into the same `perturbation_results.json` under a key formed by joining the rep names with underscores.

**Per-question gold rank export** (used by the analysis notebooks):

```bash
CUDA_VISIBLE_DEVICES=1 python -m retrieval.retrieval_rank_export reasonir
CUDA_VISIBLE_DEVICES=1   python -m retrieval.retrieval_rank_export bge
CUDA_VISIBLE_DEVICES=1   python -m retrieval.retrieval_rank_export mpnet
CUDA_VISIBLE_DEVICES=1   python -m retrieval.retrieval_rank_export splade
```

Per-representation CSV files land in `./data/retrieval_all/{model_name}_results_rank/{dataset}/{rep}_gold_rank_per_question.csv`.

---

## 5. Train the Adapter

`train_universal_adapter.py` takes three positional arguments: model name, a date string used to organise output paths, and `"subset"` or any other value to select the training corpus.

```
python -m adapter.train_universal_adapter <model_name> <date> <"subset"|"joint">
```

**Joint training on WTQ + WikiSQL + NQ:**

```bash
CUDA_VISIBLE_DEVICES=1 python -m adapter.train_universal_adapter mpnet 2026-03-01 joint
CUDA_VISIBLE_DEVICES=1 python -m adapter.train_universal_adapter reasonir 2026-03-01 joint
```

Adapter saved to:
```
/data/Kushal/UniversalRetrieval_data/2026-03-01/centroid_adapter/{model_name}/adapter.pt
```

**Subset training on WTQ + WikiSQL only** (tests cross-dataset transfer to NQ):

```bash
CUDA_VISIBLE_DEVICES=1 python -m adapter.train_universal_adapter mpnet 2026-03-01 subset
```

Adapter saved to:
```
/data/Kushal/UniversalRetrieval_data/2026-03-01/centroid_adapter_subset_dataset/{model_name}/adapter.pt
```

Training logs and checkpoints write every 200 steps. A `trajectory.jsonl` and `trajectory.pkl` in the adapter directory record per-step loss values for plotting.

---

## 6. Evaluate the Adapter

Both evaluation scripts take the same three positional arguments as the training script: model name, date, and `"subset"` or other. They resolve the adapter path from those arguments automatically.

**Recall@K evaluation across all serializations:**

```bash
# joint adapter
python -m adapter.test_universal_adapter mpnet 2026-03-01 joint
# subset adapter
python -m adapter.test_universal_adapter mpnet 2026-03-01 subset
```

Results write to:
```
/data/Kushal/UniversalRetrieval_data/retrieval_all/{model_name}_results/{dataset}/{date}/
  perturbation_results_with_adapter.json       # joint
  perturbation_results_with_adapter_subset.json  # subset
```

**Per-question rank export with adapter** (Δlog-rank analysis):

```bash
python -m adapter.test_rank_universal_adapter mpnet 2026-03-01 joint
python -m adapter.test_rank_universal_adapter mpnet 2026-03-01 subset
```

Results write to:
```
/data/Kushal/UniversalRetrieval_data/retrieval_all/{model_name}_results_rank_with_adapter/{date}/
/data/Kushal/UniversalRetrieval_data/retrieval_all/{model_name}_results_rank_with_adapter_subset/{date}/
```

---

## 7. Figures and Analysis

```bash
python analysis/representation_drift_all.py
python analysis/create_embedding_visualization.py
```

Interactive notebooks live in `figure/`:

```bash
jupyter notebook figure/
```

---

## Retriever Models

| Key | HuggingFace ID | Type |
|---|---|---|
| `mpnet` | `all-mpnet-base-v2` | Dense |
| `bge` | `BAAI/bge-m3` | Dense (multilingual) |
| `reasonir` | `reasonir/ReasonIR-8B` | Dense (reasoning-optimised) |
| `splade` | `naver/splade-v3` | Sparse lexical |
| `jina` | `jinaai/jina-embeddings-v3` | Dense (multilingual) |
| `rank1` | `jhu-clsp/rank1-7b` | Dense |

Models are downloaded automatically from the HuggingFace Hub on first use. `reasonir` and `rank1` load with `torch_dtype=torch.float16` and `device_map="auto"`.

---

## Serialization Formats

| Category | Internal key | Formats |
|---|---|---|
| Popular | `centroid_popular` | `pipe_serialized`, `token_serialized`, `space_serialized` |
| Data | `centroid_data` | `csv`, `tsv`, `html`, `markdown`, `latex`, `dict`, `json`, `xml` |
| Structural | `centroid_structural` | `shuffled_rows`, `shuffled_cols`, `transpose` |
| Schema | `centroid_schema` | `mschema`, `macschema`, `ddl` |
| All | `centroid_all` | all 17 above |

See `utils/table_representation.py` for implementation details and Appendix A.3 of the paper for template examples.

---

## Key Hyperparameters

These are the defaults set directly in `train_universal_adapter.py` via `TrainCfg`. There are no CLI flags for overriding them; edit the `TrainCfg(...)` call in the script.

| Parameter | Value |
|---|---|
| Training steps | 20 000 |
| Batch size | 512 |
| Learning rate | 3 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| λ\_inv | 100.0 |
| λ\_id | 100.0 |
| λ\_var | 25.0 |
| λ\_cov | 1.0 |
| γ\_std | 0.05 |
| Bottleneck rank (r) | 512 |
| Dropout | 0.05 |
| Residual scale (α) | 0.01 |
| Log / checkpoint every | 200 steps |

---

## Citation


```
```