# ChEMBL Drug Chat — Project TODO

---

## Pipeline stages

| Stage | Status |
|---|---|
| Data collection (ChEMBL download) | Done |
| Data transformation (SQLite → Parquet) | Done |
| LLM dataset builder (17 QA categories) | Done |
| Vector store (2.85M compounds, LanceDB RAG) | Done |
| Fine-tuning (MLX LoRA, Gemma 3 1B) | Done |
| Ollama export + GGUF conversion | Done |
| Dagster orchestration | Done |
| Bun web chat app | Done |
| Model evaluation | Missing |
| CI/CD | Done |
| Monitoring / observability | Missing |

---

## 🔥 Immediate

| # | Task | Issue |
|---|---|---|
| 1 | Regenerate training data with all current fixes | [#9](https://github.com/AymanSulaiman/chem_mlops/issues/9) |
| 2 | Retrain model on improved dataset | [#10](https://github.com/AymanSulaiman/chem_mlops/issues/10) |

```bash
# Step 1 — regenerate
uv run python -m app.scripts.flows.llm_finetuning_data.build_drug_interaction_dataset

# Step 2 — retrain
uv run python -m app.scripts.flows.finetuning.finetuning
```

---

## 📊 Training Data Improvements

| # | Task | Issue | Priority |
|---|---|---|---|
| 3 | Add CYP inhibition quantitative QA data | [#2](https://github.com/AymanSulaiman/chem_mlops/issues/2) | High |
| 4 | Add pharmacodynamic interaction training data | [#5](https://github.com/AymanSulaiman/chem_mlops/issues/5) | High |
| 5 | Add interaction severity scoring | [#7](https://github.com/AymanSulaiman/chem_mlops/issues/7) | Medium |
| 6 | Add P-glycoprotein transport interaction data | [#4](https://github.com/AymanSulaiman/chem_mlops/issues/4) | Medium |
| 7 | Integrate TWOSIDES polypharmacy dataset | [#3](https://github.com/AymanSulaiman/chem_mlops/issues/3) | Medium |

### Why these matter
- **CYP inhibition (#2):** Current DDI inference uses shared substrates (rough proxy). IC50/Ki data tells you *how strongly* a drug inhibits an enzyme — far more clinically meaningful.
- **PD interactions (#5):** We only model pharmacokinetic interactions (metabolism). Pharmacodynamic interactions (same receptor, additive/synergistic/antagonistic effects) are completely missing.
- **Severity scoring (#7):** "May alter plasma levels" is not useful. HIGH/MODERATE/LOW severity with clinical context is.
- **P-gp (#4):** P-glycoprotein transport affects absorption and CNS penetration — a major DDI mechanism not currently covered.
- **TWOSIDES (#3):** Real-world polypharmacy signals from FDA FAERS — captures effects beyond CYP metabolism.

---

## 🧪 Testing

| # | Task | Priority |
|---|---|---|
| 8 | Add tests for `build_finetune_dataset.py` | High |
| ~~9~~ | ~~Add tests for `export_to_ollama.py` (mock subprocess calls)~~ | ~~Medium~~ |
| ~~10~~ | ~~Add Dagster op/graph wiring test for `data_transformation.py`~~ | ~~Medium~~ |

---

## 📏 Model Evaluation

Without an eval step the pipeline produces a model with no automated check that it improved.

| # | Task | Priority |
|---|---|---|
| 12 | Add perplexity eval on `valid.jsonl` after fine-tuning | High |
| 13 | Build a small golden benchmark (20–50 known drug questions with expected answers) | High |
| 14 | Wire eval as a Dagster op that runs after fine-tuning and gates the Ollama export | Medium |
| 15 | Log eval metrics (perplexity, exact-match %) to a file under `artifacts/<run>/` | Medium |

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

## 🏗️ Architecture

| # | Task | Issue | Priority |
|---|---|---|---|
| 22 | Evaluate larger base model (4B or 7B) | [#8](https://github.com/AymanSulaiman/chem_mlops/issues/8) | Low |

---

## ✅ Completed

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
- [x] GitHub Actions CI workflow (lint + typecheck + pytest + bun test) — `.github/workflows/ci.yml`
- [x] `uv` environment caching in CI (`enable-cache: true`)
- [x] Tests for `export_to_ollama.py` — `app/tests/flows/export_to_ollama_test.py`
- [x] Tests for `data_transformation.py` (Dagster wiring) — `app/tests/flows/transform_data_test.py`
