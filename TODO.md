# ChEMBL Drug Chat — Project TODO

---

## Pipeline stages

| Stage | Status |
|---|---|
| Data collection (ChEMBL download) | Done |
| Data transformation (SQLite → Parquet) | Done |
| LLM dataset builder (20 QA categories) | Done |
| Vector store (2.85M compounds, LanceDB RAG) | Done |
| Fine-tuning (MLX LoRA, Gemma 3 1B) | Done |
| Ollama export + GGUF conversion | Done |
| Dagster orchestration | Done |
| Bun web chat app | Done |
| Model evaluation | Done |
| CI/CD | Done |
| Monitoring / observability | Missing |

---

## 🔥 Immediate

| # | Task |
|---|---|
| 1 | Regenerate training data with all current fixes |
| 2 | Retrain model on improved dataset |

```bash
# Step 1 — regenerate
uv run python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset

# Step 2 — retrain
uv run python -m app.scripts.flows.finetuning.finetuning
```

---

## 📊 Training Data Improvements

| # | Task | Priority |
|---|---|---|
| ~~3~~ | ~~Add CYP inhibition quantitative QA data~~ | ~~High~~ |
| ~~4~~ | ~~Add pharmacodynamic interaction training data~~ | ~~High~~ |
| ~~5~~ | ~~Add interaction severity scoring~~ | ~~Medium~~ |
| ~~6~~ | ~~Add P-glycoprotein transport interaction data~~ | ~~Medium~~ |
| 7 | Integrate TWOSIDES polypharmacy dataset | Medium |

### Why these matter
- **CYP inhibition :** ✅ IC50/Ki data from `activities` → `assays` → `target_dictionary` join. Generates quantitative inhibition QA with strength (strong/moderate/weak) per pChEMBL.
- **PD interactions :** ✅ Pairs drugs sharing the same `drug_mechanism.tid` target; classifies as additive/synergistic/antagonistic based on action types.
- **Severity scoring :** ✅ CYP3A4/2D6/2C9/2C19 → HIGH, CYP2C8/1A2/2B6 → MODERATE, others → LOW. Included in all DDI answer templates.
- **P-gp :** ✅ Substrate/inhibitor QA from ABCB1/MDR1 assays. Covers absorption and CNS penetration effects.
- **TWOSIDES :** Real-world polypharmacy signals from FDA FAERS — requires external download, not yet integrated.

---

## 🧪 Testing

| # | Task | Priority |
|---|---|---|
| ~~8~~ | ~~Add tests for `build_finetune_dataset.py`~~ | ~~High~~ |
| ~~9~~ | ~~Add tests for `export_to_ollama.py` (mock subprocess calls)~~ | ~~Medium~~ |
| ~~10~~ | ~~Add Dagster op/graph wiring test for `data_transformation.py`~~ | ~~Medium~~ |

---

## 📏 Model Evaluation

| # | Task | Priority |
|---|---|---|
| ~~12~~ | ~~Add perplexity eval on `valid.jsonl` after fine-tuning~~ | ~~High~~ |
| ~~13~~ | ~~Build a small golden benchmark (20–50 known drug questions with expected answers)~~ | ~~High~~ |
| ~~14~~ | ~~Wire eval as a Dagster op that runs after fine-tuning and gates the Ollama export~~ | ~~Medium~~ |
| ~~15~~ | ~~Log eval metrics (perplexity, exact-match %) to a file under `artifacts/<run>/`~~ | ~~Medium~~ |

---

## ⚙️ CI/CD

~~All CI/CD tasks completed.~~ See `.github/workflows/ci.yml`.

| # | Task | Priority |
|---|---|---|
| ~~16~~ | ~~Add GitHub Actions workflow: lint + typecheck + tests on every PR~~ | ~~High~~ |
| ~~17~~ | ~~Cache `uv` environment in CI to keep runs under 2 min~~ | ~~Medium~~ |
| ~~18~~ | ~~Add Bun test step (`bun test` in `web/`) to the same workflow~~ | ~~Medium~~ |

---

## 🔭 Monitoring & Observability

| # | Task | Priority |
|---|---|---|
| 19 | Log inference latency and token counts from the Bun web app | Medium |
| 20 | Add a Dagster asset check that validates LanceDB row count after ingestion | Medium |
| 21 | Persist Dagster run history (currently ephemeral) | Low |

---

## 🤖 Model Variants

Two serving modes are planned, each with a distinct trade-off:

| Variant | How it works | Strength | Weakness |
|---|---|---|---|
| **Fine-tuned** (current) | LoRA-tuned Gemma 3 1B → GGUF → Ollama | Fast, no retrieval step | Relies on memorised training facts |
| **RAG** (planned) | Same model + LanceDB lookup injected into prompt at inference | Grounded in live ChEMBL records | Slower; quality depends on retrieval relevance |

| # | Task | Priority |
|---|---|---|
| 23 | Build a RAG inference wrapper — query LanceDB with the user's drug name / SMILES, format the top-k records into a system-prompt prefix, then call the model | High |
| 24 | Wire the RAG wrapper into the Bun web app as a toggle ("Standard" vs "RAG" mode) | Medium |
| 25 | Benchmark RAG vs fine-tuned on the golden set and record results in `artifacts/` | Medium |
| 26 | Register a second Ollama model (`chembl-drug-chat:1b-rag`) that uses a RAG-aware Modelfile system prompt | Low |

---

## 🏗️ Architecture

| # | Task | Issue | Priority |
|---|---|---|---|
| 22 | Evaluate larger base model (4B or 7B) | [#8](https://github.com/AymanSulaiman/chem_mlops/issues/8) | Low |

---

## ✅ Completed

- [x] Tests for `build_finetune_dataset.py` — `app/tests/flows/build_finetune_dataset_test.py` (13 tests: `filter_activities`, `join_tables`, `create_finetuning_dataset`)
- [x] CYP inhibition quantitative QA — IC50/Ki data via `activities → assays → target_dictionary` join (`generate_cyp_inhibition_qa`)
- [x] Pharmacodynamic interaction QA — shared-receptor drug pairs with additive/synergistic/antagonistic classification (`generate_pd_interaction_qa`)
- [x] Interaction severity scoring — HIGH/MODERATE/LOW labels on all DDI QA pairs based on CYP enzyme clinical importance
- [x] P-glycoprotein transport QA — ABCB1/MDR1 substrate and inhibitor pairs (`generate_pgp_interaction_qa`)
- [x] Parallel dataset generation — `ThreadPoolExecutor` for table loading, `ProcessPoolExecutor` for generators (uses all CPU cores by default)
- [x] RAG vector store — 2.85M compounds ingested into LanceDB (PR #14)
- [x] Bun web chat app wired to Ollama (PR #13)
- [x] Fixed GGUF converter (RMSNorm +1 shift, space prefix, softcap)
- [x] Multi-turn Ollama conversation template
- [x] Anti-repetition tuning (`repeat_penalty 1.5`, `repeat_last_n 512`)
- [x] DDI pairs raised from 5K → 50K
- [x] DDI question templates from 2 → 6 per pair
- [x] Greeting/capability/redirect training examples added
- [x] "I don't have data" examples for unknown drug pairs
- [x] Role-play refusal examples added
- [x] Junk name filter (numeric `pref_name`, `AUTONOM` placeholders)
- [x] Filler phrase removal from training templates
- [x] Assay context questions capped at 5,000 (was 1.93M = 73% of data)
- [x] Model evaluation — perplexity + golden benchmark, gates Ollama export (`app/scripts/flows/eval/eval_model.py`, `data/eval/golden.jsonl`)
- [x] `uv` environment caching in CI (`enable-cache: true`)
- [x] Tests for `export_to_ollama.py` — `app/tests/flows/export_to_ollama_test.py`
- [x] Tests for `data_transformation.py` (Dagster wiring) — `app/tests/flows/transform_data_test.py`
