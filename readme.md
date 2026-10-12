# Table Representation Retrieval

Code for **"Robust Table Retrieval Under Serialization Shift via Centroid-Aligned Adapters"**.

The project measures how serialization format (CSV, TSV, HTML, Markdown, JSON, etc.) shifts table embeddings across retriever families and introduces a lightweight **Hierarchical Centroid-Aligned Adapter** that transports single-format embeddings toward a stable centroid target, improving retrieval robustness without retraining the base encoder.

---

## Repository Layout

```
.
├── adapter/
│   ├── train_universal_adapter.py        # HierTableDataset, hier_loss(), train()
│   ├── test_universal_adapter.py         # Recall@K / Log-Rank Delta evaluation with adapter
│   └── test_rank_universal_adapter.py    # per-question rank export with adapter
├── analysis/
│   ├── measure_residual_norm.py          # format drift / residual norm statistics
│   └── visualize_residual_analysis.ipynb # PCA visualization of embedding orbits
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
│   └── retrieval_rank_export.py                  # per-question gold rank export (no adapter)
├── utils/
│   ├── model.py              # get_model() factory for all retrievers
│   ├── Adapter.py            # HierDenseAdapter, HierSparseAdapter (+ legacy adapters)
│   ├── TrainerConfig.py      # TrainCfg (legacy), HierCfg (current)
│   ├── dataset.py            # CacheMultiViewDatasetGiant (legacy batch training)
│   ├── cache_embedding.py    # save/load cache (dense + sparse)
│   └── core.py               # SPLADE utilities, batched_topk, centroid helpers
├── requirements.txt
└── README.md
```

---

## 1. Installation

Python 3.10 or later is recommended.

```bash
git clone <repo-url>
cd Centroid-Aligned-Table-Retrieval
pip install -r requirements.txt
```

---

## 2. Data Setup

The pipeline expects two pre-built pickle files under `data/dataset_all/`. These are derived from WTQ, WikiSQL, and NQ-Tables and contain the questions dict and the serialized table corpus across all 17 formats.

```
data/dataset_all/new_all_questions_dict_wtq_nqt_wikisql.pkl
   → {dataset_key: {question_id: {"question": str, "gold_table_id": str}}}

data/dataset_all/all_table_transform_table_corpus.pkl
   → {dataset_key: {table_id: {rep_name: serialized_string}}}
```

Place them at those paths before running any retrieval or training script.

To build these pickles from raw benchmark data, run:

```bash
python representations/create_representation.py
```

---

## 3. Build the Embedding Cache

Embeddings are pre-computed once and stored as memory-mapped files via `utils/cache_embedding.py`. Set `CACHE_ROOT` in `utils/core.py` to a directory with sufficient disk space.

**Cache layout:**
```
{CACHE_ROOT}/{model_name}/{dataset}/chunked_{rep_name}.emb.npy      (dense)
{CACHE_ROOT}/{model_name}/{dataset}/chunked_{rep_name}.doc_ids.json
{CACHE_ROOT}/{model_name}/{dataset}/chunked_{rep_name}.sparse.npz   (SPLADE)
```

Retrieval scripts call `compute_and_cache_rep_embeddings` and `compute_and_cache_centroid` internally on cache miss, so no separate cache-build step is required. Alternatively, pre-compute explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_all_multiple_dataset mpnet
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_all_multiple_dataset bge
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_all_multiple_dataset splade
# repeat for each model key
```

---

## 4. Retrieval Baselines

All retrieval scripts take the model name as the first positional argument. Results write to `retrieval_all/{model}_results/{dataset}/perturbation_results.json`.

**Single-format retrieval across all 17 serializations and 5 centroid variants:**

```bash
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_all_multiple_dataset mpnet
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_all_multiple_dataset bge
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_all_multiple_dataset reasonir
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_all_multiple_dataset splade
```

**Custom centroid combination** (any subset of the 17 base representations):

```bash
# Args: model_name rep1 rep2 [rep3 ...]
python -m retrieval.retrieval_combination_multiple_dataset mpnet \
    csv tsv pipe_serialized space_serialized transpose
```

**Per-question gold rank export** (used by analysis notebooks):

```bash
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_rank_export mpnet
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_rank_export bge
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_rank_export reasonir
CUDA_VISIBLE_DEVICES=0 python -m retrieval.retrieval_rank_export splade
```

---

## 5. Train the Adapter

`train_universal_adapter.py` takes three positional arguments: model name, a date string used to organise output paths, and `all` (joint: WTQ + WikiSQL + NQ) or `subset` (WTQ + WikiSQL only, zero-shot NQ at evaluation).

```
python -m adapter.train_universal_adapter <model_name> <date> <all|subset>
```

**Joint training on WTQ + WikiSQL + NQ:**

```bash
CUDA_VISIBLE_DEVICES=0 python -m adapter.train_universal_adapter mpnet   2026-09-17 all > log/training_mpnet_joint.txt 2>&1
CUDA_VISIBLE_DEVICES=0 python -m adapter.train_universal_adapter bge     2026-09-17 all > log/training_bge_joint.txt 2>&1
CUDA_VISIBLE_DEVICES=0 python -m adapter.train_universal_adapter splade  2026-09-17 all > log/training_splade_joint.txt 2>&1
```

**Subset training on WTQ + WikiSQL only** (tests cross-dataset transfer to NQ):

```bash
CUDA_VISIBLE_DEVICES=0 python -m adapter.train_universal_adapter mpnet 2026-09-12 subset > log/training_mpnet_subset.txt 2>&1
```

Training logs and checkpoints save every 500 steps. A `trajectory.jsonl` and `trajectory.pkl` in the adapter directory record per-step loss values for plotting.

---

## 6. Evaluate the Adapter

Both evaluation scripts take the same three positional arguments: model name, date, and `all` or `subset`.

**Recall@1 / Log-Rank Delta across all serializations:**

```bash
python -m adapter.test_universal_adapter mpnet 2026-09-17 all      # joint adapter
python -m adapter.test_universal_adapter mpnet 2026-09-12 subset   # subset adapter
```

Results write to:
```
retrieval_all/{model}_results/{dataset}/{date}/perturbation_results_with_adapter.json        # joint
retrieval_all/{model}_results/{dataset}/{date}/perturbation_results_with_adapter_subset.json # subset
```

**Per-question rank export with adapter** (Δlog-rank analysis):

```bash
python -m adapter.test_rank_universal_adapter mpnet 2026-09-17 all
python -m adapter.test_rank_universal_adapter mpnet 2026-09-12 subset
```

---

## 7. Figures and Analysis

```bash
python analysis/measure_residual_norm.py
```

Interactive notebooks for PCA visualization and paper figures:

```bash
jupyter notebook analysis/visualize_residual_analysis.ipynb
jupyter notebook figure/
```

---

## Retriever Models

| Key | HuggingFace ID | Type | Embedding Dim | Precision |
|---|---|---|---|---|
| `mpnet` | `sentence-transformers/all-mpnet-base-v2` | Dense | 768 | float32 |
| `miniLM` | `sentence-transformers/all-MiniLM-L6-v2` | Dense | 384 | float16 |
| `bge` | `BAAI/bge-m3` | Dense (multilingual) | 1024 | float16 |
| `jina` | `jinaai/jina-embeddings-v3` | Dense (multilingual) | 1024 | float16 |
| `mdbr` | `MongoDB/mdbr-leaf-ir` | Dense | 768 | float16 |
| `reasonir` | `reasonir/ReasonIR-8B` | Dense (reasoning-augmented LLM) | 4096 | float16 |
| `rank1` | `jhu-clsp/rank1-7b` | Dense (LLM-based) | 4096 | float16 |
| `qwen0.6B` | `Qwen/Qwen3-Embedding-0.6B` | Dense | 1024 | float16 |
| `qwen4B` | `Qwen/Qwen3-Embedding-4B` | Dense | 2560 | float16 |
| `qwen8B` | `Qwen/Qwen3-Embedding-8B` | Dense | 4096 | float16 |
| `splade` | `naver/splade-v3` | Sparse (lexical) | Vocab (~30K) | float32 |

Dense models produce L2-normalized embeddings; scoring is dot product (equivalent to cosine after normalization). Models are downloaded automatically from the HuggingFace Hub on first use. Large LLM-based encoders (`reasonir`, `rank1`, `qwen4B`, `qwen8B`) load with `torch_dtype=torch.float16` and `device_map="auto"`.

---

## Serialization Formats

17 formats organized into 4 semantic categories:

| Category | Centroid Key | Formats |
|---|---|---|
| Popular (3) | `centroid_popular` | `pipe_serialized`, `token_serialized`, `space_serialized` |
| Data (8) | `centroid_data` | `csv`, `tsv`, `html`, `markdown`, `latex`, `dict`, `json`, `xml` |
| Structural (3) | `centroid_structural` | `shuffled_rows`, `shuffled_cols`, `transpose` |
| Schema (3) | `centroid_schema` | `mschema`, `macschema`, `ddl` |
| All 17 | `centroid_all` | all above (primary retrieval target) |

See `representations/create_representation.py` for implementation details.

---

## Adapter Architecture

The adapter is a two-stage module trained entirely on frozen embeddings (no gradient flows into the base retriever).

**HierDenseAdapter** (all dense models):
- **Stage A** — closed-form format offset removal: subtracts a learnable per-format bias `b_fmt` initialized to the empirically estimated mean offset of each format from the centroid
- **Stage B** — bottleneck residual with FiLM category conditioning: `z = x + Up(Dropout(FiLM_cat(GELU(Down(LN(x))))))`, with `Up` zero-initialized so the adapter starts as an exact identity

**HierSparseAdapter** (SPLADE):
- Uses a multiplicative log-gate mechanism to preserve SPLADE's non-negative sparse structure
- Gate initialized to the empirical log ratio of centroid to per-format activations

At inference, if the serialization format is unknown, the adapter uses a soft blend of per-format biases predicted by an auxiliary format classification head.

---

## Key Hyperparameters (HierCfg)

| Parameter | Value |
|---|---|
| Training steps | 30,000 |
| Tables per batch | 96 (all format views of each emitted) |
| Learning rate | 1 × 10⁻⁴ |
| Format offset LR multiplier | 0.05 (effective LR = 5 × 10⁻⁶) |
| Weight decay | 1 × 10⁻³ |
| Gradient clip (L2) | 5.0 |
| Bottleneck rank r | 1028 |
| Dropout | 0.05 |
| w_all / w_cat / w_prog / w_sep | 1.0 each |
| w_bound | 8.0 |
| w_fmt | 0.1 |
| Anneal fraction (format-align phase) | 0.4 (steps 1–12,000) |
| Centroid proximity target δ_all | 0.01 |
| Category proximity target δ_cat | 0.02 |
| Displacement bound multiplier ρ | 0.5 |
| Log / checkpoint every | 200 / 500 steps |

Hyperparameters are set via `HierCfg` in `utils/TrainerConfig.py`. There are no CLI flags; edit the dataclass directly.

---


## Citation

```
```
