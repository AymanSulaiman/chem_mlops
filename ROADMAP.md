# Roadmap

Work items from the project review (2026-09-16). Ranked — stop where it stops
being worth it. Items 1 and 2 are worth doing regardless of the rest.

---

## 1. Fix eval leakage — split by molecule, not by row

**Status:** not started · **Size:** ~half a day · **Blocks:** every number in items 2–5

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

**Files:** `build_drug_interaction_dataset.py`, `eval/golden.jsonl`,
`eval/eval_finetuned_model.py`, `app/tests/flows/eval_finetuned_model_test.py`

---

## 2. Collapse to one pane — tools instead of RAG injection

**Status:** not started · **Size:** ~1–2 days · **Depends on:** nothing

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

---

## 3. Teach the model to call tools

**Status:** not started · **Size:** ~2–3 days · **Depends on:** 2

This is the load-bearing risk in the agentic plan. Gemma 3 1B has no native
function calling, and the fine-tune at `finetuning.py` trains it to emit prose
in `### Question / ### Answer` format. A model trained to complete prose is a
poor tool-caller.

**Scope**
- Add a tool-call category to the dataset builder: question → JSON tool call →
  tool result → final answer.
- Alternative if that underperforms: constrained decoding at serve time, forcing
  valid JSON on the tool-call turn. Cheaper to try first, worth benchmarking
  against the fine-tune approach.
- Measure tool-call validity rate on the item 1 holdout set as its own metric,
  separate from answer quality.

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
