# Roadmap

Work items from the project review (2026-09-16). Ranked — stop where it stops
being worth it. Items 1 and 2 are worth doing regardless of the rest.

---

## 1. Fix eval leakage — split by molecule, not by row

**Status:** dataset rebuilt, honest pass rate not yet recorded · **Size:** ~half a day · **Blocks:**
every number in items 2–5

Every question in `app/scripts/flows/eval/golden.jsonl` is a training template
instantiated with a drug that is in the training set. `golden.jsonl:1` is
`"What does Aspirin target?"`; `build_drug_interaction_dataset.py:202` emits
`f"### Question\nWhat does {drug} target?"` for every drug in ChEMBL. The 70%
gate at `eval_finetuned_model.py:50` is measuring memorisation, not capability.

**Scope**
- Pick a holdout set of ~200 ChEMBL molecule IDs before dataset generation.
- Exclude every QA pair mentioning a holdout molecule from `train.jsonl` and
  `valid.jsonl` — the filter is on molecule ID, not on question string, since
  drugs appear across all 21 categories.
- Rebuild `golden.jsonl` from holdout molecules only.
- Re-run the gate and record the honest pass rate. Expect it to drop.

**Acceptance:** no ChEMBL ID in `golden.jsonl` appears anywhere in `train.jsonl`,
verified by a check in the test suite, not by eye.

**Done (2026-09-19)**
- `select_holdout` / `exclude_holdout` / `build_golden_benchmark` in
  `build_drug_interaction_dataset.py`. The holdout is filtered out of
  `molecule_dictionary` before the generators run, so every molregno-keyed
  category drops it at once; TWOSIDES is filtered by name since it carries no
  ChEMBL IDs. Holdout IDs land in `data/llm_finetune/holdout.json`, and
  `golden.jsonl` is rebuilt from those molecules in the same run — the two
  cannot drift apart.
- Drugs hardcoded in `generate_canonical_drug_facts_qa` are excluded from the
  holdout: curating their answers is training, so scoring on them is circular.
- Tests: `TestHoldout` (fixture-level, always runs) plus
  `test_no_golden_molecule_appears_in_train` in `eval_finetuned_model_test.py`,
  which checks the real `train.jsonl` when one exists.

**Remaining:** record the honest pass rate. The rebuild has since run —
`data/llm_finetune/holdout.json`, `train.jsonl`, `valid.jsonl` and the
molecule-keyed `golden.jsonl` are all from the same run, and
`test_no_golden_molecule_appears_in_train` passes against the real files. It was
failing on a substring match (`CHEMBL413` inside `CHEMBL413552`); it now compares
whole IDs. The eval gate itself still has to be run and its number written down.

**Known residual leak:** literature abstracts and assay descriptions are free
text and can name a holdout drug. Filtering on molecule ID cannot catch that;
a text-level filter is a separate item if the honest pass rate looks too high.

**Files:** `build_drug_interaction_dataset.py`, `eval/golden.jsonl`,
`eval/eval_finetuned_model.py`, `app/tests/flows/eval_finetuned_model_test.py`

---

## 2. Collapse to one pane — tools instead of RAG injection

**Status:** done (2026-09-19) · **Size:** ~1–2 days · **Depends on:** nothing

The side-by-side layout asks the user to judge which answer is better and gives
them nothing to judge with. Replace it with a single agentic pane: the
fine-tuned model calls the vector store as tools rather than having context
pre-injected.

**Scope**
- Drop the RAG pane and the context-injection path in `web/src/rag.ts`.
- Expose the four existing functions in `query_lancedb.py` as tool schemas:
  `query_compounds`, `get_compound_by_name`, `query_polypharmacy`,
  `query_drug_side_effects`. They are already the right shape; this is wrapping,
  not rewriting.
- Add `draw_smiles` — RDKit is already a dependency, `Draw.MolToImage` is one call.
- Agent loop is a `while` over tool calls in `web/src/`. No framework.
- Retire `benchmark_rag_vs_finetuned.py` — the head-to-head has no meaning once
  RAG stops being a competing mode.

**Acceptance:** a question needing a lookup produces a visible tool call and a
grounded answer; no LanceDB context is injected unless a tool asked for it.

**Files:** `web/src/rag.ts`, `web/src/app.ts`, `web/src/frontend.ts`,
`web/public/index.html`, `app/scripts/flows/vector_store/query_lancedb.py`,
`app/scripts/flows/eval/benchmark_rag_vs_finetuned.py` (delete)

**Done (2026-09-19)**
- One pane. `rag.ts` and its test are gone; `web/src/tools.ts` holds the tool
  specs, the system prompt, `parseToolCall` and `runTool`. `app.ts` runs the loop
  (max 3 tool steps) and emits NDJSON events — `{tool}`, `{toolResult}`,
  `{message}` — so each lookup renders as its own bubble, image included.
- `app/scripts/flows/vector_store/tools.py` is the bridge: the four
  `query_lancedb.py` functions plus `draw_molecule`, dispatched by name, every
  failure returned as `{"error": ...}` so the model can retry. One subprocess per
  call (~2 s of RDKit/LanceDB import); a stdin worker loop is the upgrade if that
  starts to hurt.
- `benchmark_rag_vs_finetuned.py`, its test, and its Dagster op are deleted;
  export now gates on `eval_finetuned_model_op` alone and the TWOSIDES ingest is
  a terminal op.
- `Bun.serve` needed `idleTimeout: 255` — the 10 s default cut the loop off
  mid-response.
- Prose streams token by token; text from the first `{` is held back until the
  turn ends, since a tool call is only recognisable once its JSON closes. Held
  text that is not a call is released, so nothing is dropped.
- `draw_molecule` (was `draw_smiles`) takes a drug **name** and reads
  `canonical_smiles` from ChEMBL. Models invent SMILES that parse but draw the
  wrong molecule — ibuprofen and paracetamol both came out wrong — so the
  structure is never taken from the model when a name is available.
- `get_compound_by_name` falls back to the `synonyms` column, matching whole
  entries only. ChEMBL stores paracetamol as ACETAMINOPHEN, so the plain lookup
  missed it; "combogesic" still must not match "Ibuprofen component of
  combogesic".

**Tool calls are prompted, not native.** Ollama rejects its `tools` field for
this model outright (`does not support tools` — Gemma 3 has no tool template), so
the system prompt asks for a bare JSON object and `parseToolCall` digs it out of
the reply.

**Verified end to end** with a tool-capable local model: "molecular weight of
Aspirin according to ChEMBL?" → visible `get_compound_by_name` call → CHEMBL25
row → "180.16, C9H8O4". The fine-tune itself does **not** call tools — it answers
from memory and hallucinates (`Aspirin (CHEMBL3984700) ... 69.5 kDa`), and will
not even echo a JSON object when told to. That is item 3, now measured rather
than predicted.

---

## 3. Teach the model to call tools

**Status:** not started · **Size:** ~2–3 days · **Depends on:** 2

This is the load-bearing risk in the agentic plan, and item 2 confirmed it:
Ollama refuses its `tools` field for `chembl-drug-chat:1b` outright, and the
fine-tune ignores a JSON-only instruction even when the question plainly needs a
lookup — it answers from memory with invented ChEMBL IDs. The loop and the tools
work; the caller does not. `parseToolCall` in `web/src/tools.ts` is the seam to
measure against.

**Scope**
- Add a tool-call category to the dataset builder: question → JSON tool call →
  tool result → final answer. **Done (2026-09-19):** `generate_tool_call_qa` in
  `build_drug_interaction_dataset.py`, registered as the "tool calls" category.
  60 K records (15 K per tool) covering `get_compound_by_name`, `draw_molecule`,
  `query_polypharmacy`, `query_drug_side_effects`. `query_compounds` is left out:
  a similarity search needs a SMILES the model does not know, so a truthful
  example is a two-hop chain whose second tool result needs the vector store.
  The record layout is byte-identical to the serve-time prompt (Ollama's
  template plus `formatToolResult`), asserted in both test suites.
- **Open question the run will answer: the ratio.** 60 K tool records sit
  against ~900 K prose records that answer the same question shapes from memory.
  If the fine-tune still skips tools, raising `MAX_TOOL_CALL_PAIRS` alone will
  not fix it — the competing prose categories (compound facts, polypharmacy)
  have to be converted or downsampled.
- Alternative if that underperforms: constrained decoding at serve time, forcing
  valid JSON on the tool-call turn. Cheaper to try first, worth benchmarking
  against the fine-tune approach.
- Measure tool-call validity rate on the item 1 holdout set as its own metric,
  separate from answer quality.

**Temporary bridge (2026-09-19):** `routeDirectToolCall` + `describeDrawResult`
in `web/src/tools.ts` answer an explicit "draw X" deterministically — tool call
and caption both, with no model turn. It exists because the current fine-tune
not only fails to call tools, it cannot use a tool result it is handed: given
one it echoes the JSON shape (`{ CHEMBL1201082 }`, an invented ebi.ac.uk URL).
Delete it and its call site when this item lands; both are marked TEMPORARY.
A bogus tool name from the model is now fed back as an error rather than shown
to the user (`unknownToolName`).

**Acceptance:** tool-call JSON parses on a stated majority of attempts, measured
on held-out drugs. This item decides whether the agent is real — if the number
is bad, items 4 and 5 are deploying something that does not work.

**Files:** `build_drug_interaction_dataset.py`, `finetuning/finetuning.py`,
`eval/eval_finetuned_model.py`

---

## 4. Terraform → Vertex AI, vLLM on the fused safetensors

**Status:** not started · **Size:** ~a weekend · **Depends on:** 3

MLX is Metal-only, so the cloud path needs a separate serving artifact. Note the
reviewer's suggestion to feed GGUF to vLLM is the wrong artifact: vLLM's GGUF
support is experimental and slower than its native path.

**Scope**
- Serve `fused_hf/` — `export_to_ollama.py:120` already writes HF safetensors as
  an intermediate before GGUF conversion. That is the vLLM input. GGUF stays for
  local Ollama; the cloud path skips it entirely.
- Terraform: Vertex AI endpoint, custom container running vLLM, model artifact
  in GCS, IAM so the chat URL authenticates against gcloud.
- Paged attention is the reason for vLLM here — record actual tokens/sec against
  local MLX so the claim is measured, not assumed.
- **Cost guardrail:** an L4 endpoint left running is roughly $400–700/mo. Include
  teardown in the Terraform and decide up front whether this runs on demand or
  stays up.

**Acceptance:** `terraform apply` yields a working authenticated chat URL;
`terraform destroy` leaves no billable resources.

**Files:** new `infra/` directory, `export_to_ollama.py` (split fuse from GGUF
conversion so `fused_hf/` is a first-class output)

---

## 5. ADK for the agent loop on GCP

**Status:** not started · **Size:** ~2–3 days · **Depends on:** 4

Worth it only if "deployed on GCP with Google's own agent framework" is the
point. ADK's real value is Agent Engine deployment and GCP resource
integration — otherwise it wraps the loop already written in item 2.

**Scope**
- Port the item 2 agent loop to ADK, same tool definitions.
- Deploy to Agent Engine, pointed at the item 4 Vertex endpoint.

**Acceptance:** the deployed agent answers a lookup question end to end with no
local machine in the path.

**Decision gate:** skip this if item 2's loop is working and the GCP framing is
not needed. It buys deployment convenience, not capability.

---

## Not doing

- **RAG as a conversational memory store.** The vector store becoming agent tools
  (item 2) is the useful half of that recommendation. Actual cross-session
  conversation memory is a separate feature with no current need for it.
