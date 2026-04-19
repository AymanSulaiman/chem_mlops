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

The pipeline is orchestrated with **Dagster** and runs entirely locally.

---

## Requirements

| Tool | Version |
|------|---------|
| Python | ≥ 3.13 |
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

### Start the Dagster UI

```bash
dagster dev -w deployments/workspace.yaml
```

Open [http://localhost:3000](http://localhost:3000) to browse ops, trigger runs, and inspect logs. The `chembl_pipeline` job is pre-configured with a daily midnight UTC schedule.

### Run everything (headless)

```bash
uv run python -m app.orchestration.data_transformation
```

This executes the full pipeline via Dagster:

1. Download ChEMBL SQLite archive
2. Convert all tables to Parquet
3. Build the QA JSONL dataset **and** the activity Parquet (in parallel)
4. Fine-tune Gemma 3 1B with LoRA
5. Fuse the LoRA adapter and register the model with Ollama

### Build with the full dataset

To run the complete pipeline end-to-end using **all** available ChEMBL data:

```bash
# Step 1 — Download ChEMBL (~5.6 GB, takes ~5 min on a fast connection)
uv run python -m app.scripts.flows.initial_data_transformation.collect_data

# Step 2 — Convert SQLite → Parquet for all 74 tables (~10–20 min)
uv run python -m app.scripts.flows.initial_data_transformation.transform_data

# Step 3a — Build the QA finetuning dataset from all 23 tables, no row cap
#            (~16–24 GB RAM recommended; produces ~500 K+ training pairs)
uv run python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset

# Step 3b — Build the activity Parquet dataset
uv run python -m app.scripts.flows.llm_finetuning_data.build_finetune_dataset

# Step 4 — Fine-tune Gemma 3 1B on the full dataset (~2–4 hrs on M1 Pro)
uv run app/scripts/flows/finetuning/finetuning.py

# Step 5 — Fuse adapter, export to GGUF, and register with Ollama
uv run python -m app.scripts.flows.finetuning.export_to_ollama
```

Expected disk and time requirements:

| Step | Disk | Time (approx) |
|------|------|---------------|
| Download ChEMBL SQLite | 5.6 GB | ~5 min |
| Convert to Parquet | 8–10 GB | ~15 min |
| Build QA dataset | < 1 GB output | ~30–60 min |
| Fine-tune (1 500 iters) | ~2 GB adapter | ~2–4 hrs |
| Export to Ollama | ~4 GB GGUF | ~5–10 min |

> **Low-RAM machines:** If you have less than 16 GB of RAM, cap the table load with `--row-limit`:
> ```bash
> uv run python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset \
>   --row-limit 200000
> ```

### Run steps individually

```bash
# 1. Download ChEMBL
uv run python -m app.scripts.flows.initial_data_transformation.collect_data

# 2. Transform to Parquet
uv run python -m app.scripts.flows.initial_data_transformation.transform_data

# 3a. Build the QA finetuning dataset
uv run python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset

# 3b. Build the activity Parquet dataset
uv run python -m app.scripts.flows.llm_finetuning_data.build_finetune_dataset

# 4. Fine-tune
uv run app/scripts/flows/finetuning/finetuning.py

# 5. Export to Ollama and start chatting
uv run python -m app.scripts.flows.finetuning.export_to_ollama
ollama run chembl-drug-chat:1b
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
uv run python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset \
  [--data-dir PATH]       # default: data/chembl_transform
  [--output-dir PATH]     # default: data/llm_finetune
  [--row-limit N]         # cap every table at N rows (useful on low-RAM machines)
```

The `--row-limit` flag is useful on memory-constrained machines. For example, `--row-limit 200000` limits each table to 200 K rows. Without it, all rows are loaded (the `activities` table alone has ~19 M rows).

### Using all available data

To build the largest possible dataset from the full ChEMBL database, run without `--row-limit`:

```bash
uv run python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset
```

This loads all rows from all 23 tables. Expected scale:

| Table | Rows |
|-------|------|
| `activities` | ~19 M |
| `docs` | ~99 K |
| `assays` | ~1.5 M |
| `molecule_dictionary` | ~2.4 M |
| `compound_properties` | ~2.3 M |
| other tables | < 200 K each |

> **Memory requirement:** ~16–24 GB of RAM is recommended. On an M1 Pro with 32 GB unified memory this runs comfortably. On machines with less RAM, use `--row-limit` to cap the load (e.g. `--row-limit 500000`).

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

## Loading into Ollama

After fine-tuning, you can serve the model locally via [Ollama](https://ollama.com).

### 1. Install Ollama

```bash
brew install ollama
```

### 2. Export to Ollama

The export script handles everything — it fuses the LoRA adapter into the base model,
converts to GGUF via llama.cpp's conversion script (downloaded automatically on first run),
writes a Modelfile with the system prompt, and registers the model with Ollama.

**Auto-detect the latest run:**

```bash
uv run python -m app.scripts.flows.finetuning.export_to_ollama
```

**Target a specific run:**

```bash
uv run python -m app.scripts.flows.finetuning.export_to_ollama \
  --run-dir artifacts/20260403_220717
```

**Options:**

```
--run-dir PATH        Run directory to export (default: latest in artifacts/)
--model-name NAME     Ollama model name (default: chembl-drug-chat:1b)
--force               Overwrite an existing export
```

The script runs 4 steps:

1. `mlx_lm fuse --save-path` — merge LoRA adapter into base model (HF safetensors)
2. `convert_hf_to_gguf.py` — convert to GGUF F16 (script fetched from llama.cpp on first run)
3. Write `Modelfile` — system prompt + sampling parameters
4. `ollama create` — register with Ollama

The script produces:

```
artifacts/<timestamp>/mlx/ollama/
├── fused_hf/                   # merged HF safetensors (intermediate)
├── chembl-drug-chat.gguf       # final GGUF model
└── Modelfile                   # system prompt + sampling parameters
```

> **Note:** The llama.cpp conversion script is downloaded once to
> `~/.cache/chem_mlops/convert_hf_to_gguf.py` and reused on subsequent runs.

### 3. Chat

```bash
ollama run chembl-drug-chat:1b
```

Example questions:

```
>>> What does Aspirin target?
>>> How is Warfarin metabolised?
>>> What are the black box warnings for Methotrexate?
>>> Which drugs share the CYP2C9 metabolic pathway with Warfarin?
>>> What is the ligand efficiency of Imatinib?
>>> Is Adalimumab a small molecule or a biologic?
```

### Updating after a new fine-tuning run

```bash
uv run python -m app.scripts.flows.finetuning.export_to_ollama --force
```

This re-fuses from the latest artifact and replaces the existing Ollama model.


```
chem_mlops/
├── app/
│   ├── orchestration/
│   │   └── data_transformation.py      # Dagster pipeline (@op / @graph / Definitions)
│   └── scripts/
│       ├── flows/
│       │   ├── initial_data_transformation/
│       │   │   ├── collect_data.py     # Download ChEMBL SQLite
│       │   │   └── transform_data.py   # SQLite → Parquet (DuckDB)
│       │   ├── llm_finetuning_data/
│       │   │   ├── build_drug_interaction_dataset.py  # 17-category QA builder
│       │   │   └── build_finetune_dataset.py          # Activity Parquet → JSONL
│       │   └── finetuning/
│       │       ├── finetuning.py       # MLX LoRA fine-tuning + Ollama export
│       │       └── export_to_ollama.py # Standalone: export any run to Ollama
│       └── load_data/
│           └── load_data.py            # ChemblDataLoader helper
├── data/
│   ├── chembl_transform/               # Parquet files (one per table)
│   └── llm_finetune/                   # train.jsonl / valid.jsonl
├── deployments/
│   └── workspace.yaml                  # Dagster code-location config
├── notebooks/                          # Exploratory data analysis
├── artifacts/                          # Fine-tuning run outputs
└── pyproject.toml
```

---

## Development

```bash
# Run tests
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run ty check
```

All three must pass with zero errors before merging.

---

## Data sources

- **ChEMBL**: [https://www.ebi.ac.uk/chembl/](https://www.ebi.ac.uk/chembl/)
- Schema diagram: [chembl_36_schema.png](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_schema.png)
- Schema documentation: [schema_documentation.html](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/schema_documentation.html)
