# chem_mlops

An end-to-end MLOps pipeline that downloads ChEMBL, builds a 2.85 M-compound vector store for RAG, generates a structured QA dataset, and fine-tunes a Gemma 3 1B language model to answer drug-interaction questions — optimised for Apple Silicon (M1 Pro / M2 / M3).

---

## Overview

```mermaid
flowchart LR
    EBI[(ChEMBL SQLite\n5.6 GB EBI FTP)] --> COL[collect_data\nDownload & extract\nchembl_XX.db]
    COL --> TRF[transform_data\nSQLite → Parquet\nvia DuckDB]

    S3[(TWOSIDES\nFDA FAERS · S3)] --> DLT[download_twosides\nstream-decompress\n→ Parquet]

    TRF --> DDI[build_drug_interaction_dataset\n23 tables · 21 QA categories\n~500K+ training pairs]
    DLT --> DDI
    TRF --> FDS[create_finetuning_dataset\nactivities Parquet → JSONL]
    TRF --> ING[ingest_to_lancedb\n2.85M compound vectors\nMorgan fingerprints · ~6 min]

    ING --> LDB[(LanceDB\nchembl_CHEMBL_36\ncompounds table)]
    DLT --> ING2[ingest_twosides_to_lancedb\nPRR-filtered pairs\n→ polypharmacy table]
    ING2 --> LDB2[(LanceDB\nchembl_CHEMBL_36\npolypharmacy table)]

    DDI --> FT[finetune_lora\nMLX LoRA on Gemma 3 1B-PT\n~1500 iters · Apple Silicon]
    FDS --> FT

    FT --> EXP[export_to_ollama\nfuse adapter → GGUF\nollama create]
    EXP --> OLL[(Ollama\nchembl-drug-chat:1b)]

    USR([User query\nSMILES / drug name]) -->|query_compounds\nget_compound| LDB
    USR -->|query_polypharmacy\nquery_drug_side_effects| LDB2
    LDB -->|top-n similar\ncompounds + metadata| OLL
    LDB2 -->|polypharmacy\nside-effect signals| OLL
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
3. In parallel:
   - Download TWOSIDES polypharmacy dataset (FDA FAERS, Tatonetti et al.)
   - Build the QA JSONL dataset (ChEMBL + TWOSIDES)
   - Build the activity Parquet
   - **Ingest 2.85 M compounds into LanceDB** (vector store for RAG)
   - **Ingest TWOSIDES polypharmacy pairs into LanceDB** (after both ChEMBL ingest and TWOSIDES download complete)
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

# Step 3c — Ingest 2.85 M compounds into LanceDB (~6 min)
uv run python -m app.scripts.flows.vector_store.ingest_to_lancedb

# Step 3d — Download TWOSIDES polypharmacy dataset from Tatonetti Lab S3
#            (streams ~120 MB gzip, decompresses in memory, saves as Parquet)
uv run python -m app.scripts.flows.llm_finetuning_data.download_twosides

# Step 3e — Ingest TWOSIDES into LanceDB polypharmacy table (run after 3c and 3d)
uv run python -m app.scripts.flows.vector_store.ingest_twosides_to_lancedb

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
| **Ingest to LanceDB** | ~15 GB | **~6 min** |
| Download TWOSIDES | ~50 MB Parquet | ~2–3 min |
| Ingest TWOSIDES to LanceDB | < 100 MB | ~1 min |
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

# 3c. Ingest compounds into LanceDB (vector store)
uv run python -m app.scripts.flows.vector_store.ingest_to_lancedb

# 3d. Download TWOSIDES polypharmacy dataset
uv run python -m app.scripts.flows.llm_finetuning_data.download_twosides

# 3e. Ingest TWOSIDES polypharmacy pairs into LanceDB
uv run python -m app.scripts.flows.vector_store.ingest_twosides_to_lancedb

# 4. Fine-tune
uv run app/scripts/flows/finetuning/finetuning.py

# 5. Export to Ollama and start chatting
uv run python -m app.scripts.flows.finetuning.export_to_ollama
ollama run chembl-drug-chat:1b
```

---

## Vector Store (RAG)

The pipeline builds a **LanceDB vector store** alongside fine-tuning — 2,854,996 compounds from ChEMBL, each represented as a 2048-bit Morgan fingerprint (ECFP4, radius 2).

This enables **Retrieval-Augmented Generation (RAG)** at inference time: instead of relying on the model to remember facts, query the vector store first and inject the retrieved record directly into the prompt.

### Why RAG alongside fine-tuning?

Fine-tuning teaches the model to *sound like* a domain expert. It cannot guarantee factual accuracy for specific compounds. RAG grounds answers in real ChEMBL records — mechanisms, indications, warnings, metabolic enzymes — that the model only needs to format.

### Ingest

```bash
# Runs automatically as part of the Dagster pipeline, or standalone:
uv run python -m app.scripts.flows.vector_store.ingest_to_lancedb
```

Re-runs are safe — the table is always overwritten. Output: `data/lancedb/chembl_CHEMBL_36/`.

### Query — compounds

```python
from app.scripts.flows.vector_store.query_lancedb import query_compounds, get_compound

# Similarity search — top 5 compounds most similar to aspirin
hits = query_compounds("CC(=O)Oc1ccccc1C(=O)O", n=5)
# Returns list of dicts with all metadata columns + _distance

# Exact lookup by ChEMBL ID
record = get_compound("CHEMBL25")
# Returns dict or None
```

### Query — polypharmacy (TWOSIDES)

The `polypharmacy` table stores drug-pair adverse-event signals from the TWOSIDES dataset (Tatonetti et al., *Science Translational Medicine* 2012), derived from FDA FAERS co-reporting. Only pairs with PRR ≥ 3.0 and ≥ 5 reported cases are retained. It is a scalar-indexed lookup table with no vector column.

```python
from app.scripts.flows.vector_store.query_lancedb import query_polypharmacy, query_drug_side_effects

# Look up a specific drug pair (order-insensitive, case-insensitive)
pair = query_polypharmacy("Warfarin", "Aspirin")
# Returns dict with side_effects, max_prr, total_cases, n_side_effects — or None

# Find all known polypharmacy partners for a single drug
pairs = query_drug_side_effects("Warfarin", n=20)
# Returns list of dicts sorted by max_prr descending
```

Run `download_twosides` and `ingest_twosides_to_lancedb` before querying this table.

### Internal sanity check

```bash
# Queries the live store with known molecules and asserts correctness
uv run python -m app.scripts.flows.vector_store.query_lancedb
```

Checks: top similarity hit is aspirin itself, exact lookup returns the right record, invalid SMILES raises `ValueError`, unknown ID returns `None`.

Full details: [`app/scripts/flows/vector_store/README.md`](app/scripts/flows/vector_store/README.md)

---

## QA Dataset

`build_drug_interaction_dataset` reads 23 ChEMBL tables plus the TWOSIDES polypharmacy dataset and emits 21 categories of training pairs in `### Question / ### Answer` format:

| # | Category | Source tables |
|---|----------|--------------|
| 1 | Mechanism of action | `drug_mechanism`, `target_dictionary` |
| 2 | Therapeutic indication | `drug_indication` |
| 3 | Metabolic pathways | `metabolism`, `target_dictionary` |
| 4 | Drug-drug interactions (with severity) | `metabolism` (shared CYP substrates, HIGH/MODERATE/LOW severity) |
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
| 18 | CYP inhibition (quantitative) | `activities`, `assays`, `target_dictionary` (IC50/Ki values) |
| 19 | Pharmacodynamic interactions | `drug_mechanism`, `target_dictionary` (shared receptor targets) |
| 20 | P-glycoprotein transport | `activities`, `assays`, `target_dictionary` (ABCB1/MDR1) |
| 21 | Polypharmacy side effects | TWOSIDES (FDA FAERS · PRR-filtered drug-pair adverse events) |

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
  [--workers N]           # parallel generator processes (default: CPU count)
```

The `--row-limit` flag is useful on memory-constrained machines. For example, `--row-limit 200000` limits each table to 200 K rows. Without it, all rows are loaded (the `activities` table alone has ~19 M rows).

Generators run in parallel by default (one process per CPU core). Use `--workers 1` to disable multiprocessing, which is useful for debugging tracebacks.

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


## Project structure

```
chem_mlops/
├── .github/
│   └── workflows/
│       └── ci.yml                      # Lint + typecheck + pytest + bun test on every push/PR
├── app/
│   ├── orchestration/
│   │   ├── data_transformation.py      # Dagster pipeline (@op / @graph / Definitions)
│   │   └── README.md                   # Pipeline architecture and stage docs
│   ├── scripts/
│   │   ├── flows/
│   │   │   ├── initial_data_transformation/
│   │   │   │   ├── collect_data.py     # Download ChEMBL SQLite
│   │   │   │   └── transform_data.py   # SQLite → Parquet (DuckDB)
│   │   │   ├── llm_finetuning_data/
│   │   │   │   ├── build_drug_interaction_dataset.py  # 21-category QA builder (parallel, incl. TWOSIDES)
│   │   │   │   ├── build_finetune_dataset.py          # Activity Parquet → JSONL
│   │   │   │   └── download_twosides.py               # Stream-download TWOSIDES → Parquet
│   │   │   ├── finetuning/
│   │   │   │   ├── finetuning.py       # MLX LoRA fine-tuning
│   │   │   │   └── export_to_ollama.py # Standalone: export any run to Ollama
│   │   │   ├── eval/
│   │   │   │   ├── eval_model.py       # Perplexity + golden benchmark eval
│   │   │   │   └── golden.jsonl        # Golden Q&A benchmark
│   │   │   └── vector_store/
│   │   │       ├── ingest_to_lancedb.py           # Join 13 tables → fingerprint → LanceDB
│   │   │       ├── ingest_twosides_to_lancedb.py  # TWOSIDES → polypharmacy LanceDB table
│   │   │       ├── query_lancedb.py               # query_compounds / get_compound / polypharmacy API
│   │   │       └── README.md                      # Vector store architecture and tradeoffs
│   │   └── load_data/
│   │       └── load_data.py            # ChemblDataLoader helper
│   └── tests/
│       ├── flows/                      # Tests for each pipeline stage
│       └── load_data/                  # Tests for ChemblDataLoader
├── web/                                # Bun chat app (talks to Ollama)
├── data/
│   ├── chembl_transform/               # Parquet files (one per table)
│   ├── llm_finetune/                   # train.jsonl / valid.jsonl
│   ├── lancedb/                        # LanceDB vector store
│   │   └── chembl_CHEMBL_36/           # compounds table (2,854,996 · 2048-bit vectors)
│   │                                   # polypharmacy table (PRR-filtered TWOSIDES pairs)
│   └── twosides/                       # TWOSIDES download cache
│       └── TWOSIDES.parquet            # ~50 MB zstd-compressed (gitignored)
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

All three must pass with zero errors before merging. CI runs the same checks automatically on every push and pull request via `.github/workflows/ci.yml`.

---

## Bun web app

The repository also includes a small Bun chat app in `web/` that talks to the latest Ollama model exported by this project.

Run it with:

```bash
cd web
bun run server.ts
```

Run its Bun-native tests with:

```bash
cd web
bun test
```

Full details live in `web/README.md`.

---

## Data sources

- **ChEMBL**: [https://www.ebi.ac.uk/chembl/](https://www.ebi.ac.uk/chembl/)
  - Schema diagram: [chembl_36_schema.png](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_36_schema.png)
  - Schema documentation: [schema_documentation.html](https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/schema_documentation.html)
- **TWOSIDES**: Tatonetti et al., *Science Translational Medicine* 2012 — drug-pair adverse event signals derived from FDA FAERS co-reporting. Hosted by the Tatonetti Lab at Columbia University.
