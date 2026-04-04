# chem_mlops

An end-to-end MLOps pipeline that downloads ChEMBL, transforms it into a structured QA dataset, and fine-tunes a Gemma 3 1B language model to answer drug-interaction questions — optimised for Apple Silicon (M1 Pro / M2 / M3).

---

## Overview

```
ChEMBL SQLite (5.6 GB)
        │
        ▼
  collect_data          Download & extract chembl_XX.db
        │
        ▼
  transform_data        Convert all tables → Parquet via DuckDB
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
build_drug_interaction_dataset    create_finetuning_dataset
  23 ChEMBL tables                 activities parquet → JSONL
  17 QA categories
  ~500 K training pairs
        │
        └──────────────┬───────────────────┘
                       ▼
              finetune_lora (MLX LoRA)
              Gemma 3 1B-PT → adapter
```

The pipeline is orchestrated with **Prefect** and runs entirely locally.

---

## Requirements

| Tool | Version |
|------|---------|
| Python | ≥ 3.12 |
| [uv](https://docs.astral.sh/uv/) | any |
| macOS + Apple Silicon | M1 / M2 / M3 |

> **Note:** The fine-tuning step uses `mlx-lm` and requires Apple Silicon. All other steps run on any platform.

---

## Installation

```bash
git clone <repo>
cd chem_mlops
uv sync
```

---

## Pipeline

### Run everything

```bash
python -m app.orchestration.data_transformation
```

This executes the full pipeline via Prefect:

1. Download ChEMBL SQLite archive
2. Convert all tables to Parquet
3. Build the QA JSONL dataset **and** the activity Parquet (in parallel)
4. Fine-tune Gemma 3 1B with LoRA

### Run steps individually

```bash
# 1. Download ChEMBL
python -m app.scripts.flows.initial_data_transformation.collect_data

# 2. Transform to Parquet
python -m app.scripts.flows.initial_data_transformation.transform_data

# 3a. Build the QA finetuning dataset
python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset

# 3b. Build the activity Parquet dataset
python -m app.scripts.flows.llm_finetuning_data.build_finetune_dataset

# 4. Fine-tune
uv run app/scripts/flows/finetuning/finetuning.py
```

---

## QA Dataset

`build_drug_interaction_dataset` reads 23 ChEMBL tables and emits 17 categories of training pairs in `### Question / ### Answer` format:

| # | Category | Source tables |
|---|----------|--------------|
| 1 | Mechanism of action | `drug_mechanism`, `target_dictionary` |
| 2 | Therapeutic indication | `drug_indication` |
| 3 | Metabolic pathways | `metabolism`, `target_dictionary` |
| 4 | Drug-drug interactions | `metabolism` (shared CYP substrates) |
| 5 | Bioactivity potency | `activities` (pChEMBL values) |
| 6 | Drug warnings | `drug_warning` |
| 7 | Drug synonyms | `molecule_synonyms` |
| 8 | Physicochemical properties | `compound_properties` |
| 9 | ATC classification | `atc_classification`, `molecule_atc_classification` |
| 10 | Approved products | `formulations`, `products` |
| 11 | Scientific literature | `docs` |
| 12 | Assay context | `assays`, `activities` |
| 13 | Ligand efficiency | `ligand_eff`, `activities` |
| 14 | Protein target sequences | `component_sequences`, `target_components` |
| 15 | Protein family | `protein_classification`, `component_class`, `target_components` |
| 16 | Biotherapeutics | `biotherapeutics` |
| 17 | Target relations | `target_relations` |

Output files are written to `data/llm_finetune/`:

```
data/llm_finetune/
├── train.jsonl   (90%)
└── valid.jsonl   (10%)
```

Each record:
```json
{"text": "### Question\nWhat does Aspirin target?\n\n### Answer\nAspirin (CHEMBL25) inhibits Cyclooxygenase-1 ..."}
```

### CLI options

```bash
python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset \
  [--data-dir PATH]       # default: data/chembl_transform
  [--output-dir PATH]     # default: data/llm_finetune
  [--row-limit N]         # cap every table at N rows (useful on low-RAM machines)
```

The `--row-limit` flag is useful on memory-constrained machines. For example, `--row-limit 200000` limits each table to 200 K rows. Without it, all rows are loaded (the `activities` table alone has ~19 M rows).

---

## Fine-tuning

Fine-tuning runs `mlx-lm` LoRA on **Gemma 3 1B** (`google/gemma-3-1b-pt`), optimised for Apple Silicon unified memory:

| Parameter | Value |
|-----------|-------|
| Method | LoRA |
| Layers | 16 of 18 |
| Batch size | 4 |
| Iterations | 1 500 |
| Learning rate | 1e-5 |
| Max sequence length | 2 048 |
| Quantisation | 4-bit (q-group 64) |
| Gradient checkpointing | ✓ |

The script:
1. Quantises the base model to 4-bit MLX format
2. Splits any sequences longer than 2 048 tokens
3. Runs LoRA training and saves adapter weights

Artifacts are written to `artifacts/<timestamp>/`:

```
artifacts/20260403_220717/
├── mlx/gemma-3-1b-pt-mlx/        # quantised base model
└── adapters/gemma3-1b-pt-chembl-toon/  # LoRA adapter weights
```

---

## Project structure

```
chem_mlops/
├── app/
│   ├── orchestration/
│   │   └── data_transformation.py      # Prefect pipeline DAG
│   └── scripts/
│       ├── flows/
│       │   ├── initial_data_transformation/
│       │   │   ├── collect_data.py     # Download ChEMBL SQLite
│       │   │   └── transform_data.py   # SQLite → Parquet (DuckDB)
│       │   ├── llm_finetuning_data/
│       │   │   ├── build_drug_interaction_dataset.py  # 17-category QA builder
│       │   │   └── build_finetune_dataset.py          # Activity Parquet → JSONL
│       │   └── finetuning/
│       │       └── finetuning.py       # MLX LoRA fine-tuning
│       └── load_data/
│           └── load_data.py            # ChemblDataLoader helper
├── data/
│   ├── chembl_transform/               # Parquet files (one per table)
│   └── llm_finetune/                   # train.jsonl / valid.jsonl
├── deployments/
│   └── chembl_pipeline_deployments.yaml
├── notebooks/                          # Exploratory data analysis
├── artifacts/                          # Fine-tuning run outputs
└── pyproject.toml
```

---

## Development

```bash
# Run tests
.venv/bin/pytest

# Lint
.venv/bin/ruff check .

# Type check
.venv/bin/ty check
```

All three must pass with zero errors before merging.

---

## Data sources

- **ChEMBL**: [https://www.ebi.ac.uk/chembl/](https://www.ebi.ac.uk/chembl/)
- Schema diagram: [chembl_36_schema.png](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_schema.png)
- Schema documentation: [schema_documentation.html](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/schema_documentation.html)
