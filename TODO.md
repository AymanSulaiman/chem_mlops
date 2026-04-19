# ChEMBL Drug Chat — Improvement Tasks

Tracked on the [GitHub project board](https://github.com/users/AymanSulaiman/projects/4).

---

## 🔥 Immediate (run these now)

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

## 🏗️ Architecture Improvements

| # | Task | Issue | Priority |
|---|---|---|---|
| 8 | Implement RAG for accurate inference-time lookups | [#6](https://github.com/AymanSulaiman/chem_mlops/issues/6) | High |
| 9 | Evaluate larger base model (4B or 7B) | [#8](https://github.com/AymanSulaiman/chem_mlops/issues/8) | Low |

### RAG vs fine-tuning
Fine-tuning memorises patterns → hallucinates when it doesn't know an answer (e.g. drug name "340156").
RAG retrieves real data at inference time → always grounded in ChEMBL facts.

```
User: "Does fluoxetine interact with tramadol?"
  ↓
Retriever: query metabolism.parquet + activities.parquet
  ↓
Context: "fluoxetine (CHEMBL49) — CYP2D6 inhibitor IC50=24nM
          tramadol (CHEMBL636) — CYP2D6 substrate"
  ↓
Model generates grounded answer
```

---

## ✅ Completed (this session)

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
