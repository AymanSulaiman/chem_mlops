# Roadmap

Work items from the project review (2026-09-16). Ranked — stop where it stops
being worth it. Items 1 and 2 are worth doing regardless of the rest.

---

## 1. Fix eval leakage — split by molecule, not by row

**Status:** dataset rebuilt, honest pass rate recorded (5.0%) · **Size:** ~half a day
· **Remaining:** re-point golden at the agent loop · **Blocks:** every number in items 2–5

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

**Honest pass rate (2026-09-20): 5.0% (2/40), against a 70% gate.** The number
is real and the gate was wrong, not the model. Every golden question is "What
does {drug} target?" about a molecule this item deliberately deleted from
training, so the fine-tune guesses a plausible protein — SIROLIMUS ->
"Insulin receptor substrate 1", BRECANAVIR -> "Adenovirus". The two passes are
EGFR and asparagine, the two most guessable answers in the set. Recalling a
held-out drug's target is a **lookup**, not something weights can supply, so
golden is now **recorded, not gated** (`golden_gated: false`); the tool-call
parse rate gates the export instead. Re-point golden at the agent loop (model +
tools) and it becomes a real gate again — that is the remaining work here.

**Also worth fixing:** `build_golden_benchmark` only ever emits
`mechanism_of_action` questions, so all 40 golden items test one skill. The
benchmark is narrower than it reads.

**Earlier remaining (done):** the rebuild has run —
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

**Status:** routing/grounding/prose all 100% after dropping the system prompt;
bridge removal + self-continuation open · **Size:** ~2–3 days · **Depends on:** 2

This is the load-bearing risk in the agentic plan. Item 2 found the caller
broken outright: Ollama refuses its `tools` field for `chembl-drug-chat:1b`, and
the fine-tune ignored a JSON-only instruction even when the question plainly
needed a lookup, answering from memory with invented ChEMBL IDs.

**That has changed.** A full retrain including the tool-call category
(`20260920_081335`) emits parseable tool-call JSON 87.5% of the time. The open
question is no longer *whether* it calls a tool but *which* — see the measured
rates below. `parseToolCall` / `resolve_tool_call` are the seam to measure
against.

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
  separate from answer quality. **Done (2026-09-20):** `run_tool_call_benchmark`
  in `eval_finetuned_model.py` asks four tool-shaped questions about each of 10
  held-out drugs and reports three rates, because they fail independently and
  the fixes differ:
  `parse_rate` (JSON came back at all — a decoding problem), `known_rate` (it
  named a tool that exists — a naming problem), `correct_rate` (right tool, with
  the expected argument — the routing problem that decides whether the agent is
  real). Written to `data/eval/<run>/finetuned_tool_call_results.jsonl` and
  summarised in the metrics file. **Measured, not gated**: the model this gate
  protects cannot call a tool at all yet, so a threshold would block every
  export. Turn `tool_call_gated` on once a trained model clears it.

**Measured on the exported model (run `20260920_081335`, 2026-09-20).** 40 calls,
10 held-out drugs. Before aliasing / after:

| rate | before | after |
|---|---|---|
| `parse_rate` | 87.5% | 87.5% |
| `known_rate` | 67.5% | **87.5%** |
| `correct_rate` | 42.5% | **55.0%** |

`known_rate` now equals `parse_rate`: every call the model produces reaches a
real tool. That is the ceiling for a name-mapping layer, and it was reached
without training anything.

**Aliasing landed (2026-09-20).** `resolve_tool_call` in
`vector_store/tools.py`, mirrored by `resolveToolCall` in `web/src/tools.ts`,
maps near-miss names (`query_compound`, `query_drugs` -> `get_compound_by_name`)
and argument keys (`drug_name` -> `name`) onto the registry, and rereads a
`query_compounds` call carrying a drug name instead of a SMILES as the by-name
lookup. The benchmark scores *through* the resolver, so the measured rate is the
rate users get; a test asserts the two alias tables match, because a silent
drift between them would make the number a lie. `run_tool` also drops arguments
the tool does not take — models pad calls with `"n": 10` on a single-record
lookup, and a TypeError is worse than an ignored key.

**What aliasing cannot fix — this is what the training run is for:**
- `query_polypharmacy` **0/10.** "Is it safe to take X with warfarin?" gets
  `query_drug_side_effects` with a single drug. The model does not distinguish a
  one-drug interaction scan from a named-pair lookup. Genuinely the wrong tool,
  not the wrong spelling.
- `get_compound_by_name` **5/10.** The misses call `query_compounds` with an
  **invented SMILES** — the same hallucination that made `draw_molecule` take a
  name instead of a structure (item 2). A made-up SMILES parses, so nothing
  downstream can catch it.
- `draw_molecule` 9/10 and `query_drug_side_effects` 8/10 already work.

**The second half of the loop, now measured (2026-09-20).**
`run_tool_result_benchmark` hands the model a tool result and scores whether it
*reads* it. Every expected value is fabricated (`481.27`, `C23H31N5O4`,
`Zalbovir`, `7.43`) on a held-out drug, so a memorised answer cannot score — an
answer containing the marker is proof it read the result. Two rates:
`grounded_rate` 45%, `prose_rate` 60% of 40.

| tool | grounded | prose |
|---|---|---|
| `query_drug_side_effects` | 9/10 | 10/10 |
| `query_polypharmacy` | 5/10 | 4/10 |
| `draw_molecule` | 4/10 | 10/10 |
| `get_compound_by_name` | **0/10** | **0/10** |

- `get_compound_by_name` is fully broken: handed the answer, it re-emits its own
  tool call verbatim — `{"drug_name": "SIROLIMUS", "n": 1, "tool":
  "get_compound_by_name", ...}`. It asks the question again instead of answering
  it. This is the most common lookup shape in the product.
- `query_polypharmacy` fabricates *data* in half the cases: handed one side
  effect it emits a JSON object listing several it was never given
  (`thrombocytopenia`, `platelet count decreased`). Invented interactions in a
  drug-interaction tool is the worst failure mode on this list.
- `draw_molecule` answers in prose every time but pads with invented facts — "a
  molecular weight of 290.0 Da" appears nowhere in the result it was handed.

**Bridge verdict: keep it, for a new reason.** `routeDirectToolCall` was
justified by "handed a tool result it echoes the JSON shape". For draws that is
now false — 10/10 prose. But grounding is 4/10, so deleting the bridge would
swap a deterministic correct caption for a fluent one that invents a molecular
weight more often than not. It goes when `draw_molecule` grounding is high, not
when the echoing stops. The comment in `web/src/tools.ts` still states the old
premise and should be corrected when the bridge is revisited.

**The export gate moved (2026-09-20).** It was the golden pass rate, which after
item 1 measures whether the model can recall facts deliberately withheld from it
— so it blocked every export for a capability the fine-tune is not supposed to
have. Golden is now recorded (`golden_gated: false`) and `tool_call_parse_rate`
gates instead, at `TOOL_CALL_PARSE_THRESHOLD = 0.5` — this item's stated
acceptance criterion. `correct_rate` is printed but not gated: gating it at a
useful level today blocks every export, and it is the number to gate once the
training run lands.

**The continued-training run happened (2026-09-20, `20260920_114710_tools`).**
Mix was balanced — 12 K records per tool against 97 K prose, the 1:2 this item
specified — 600 iters at 1e-5. Head to head with the full retrain, both measured
under the same prompt:

| metric | full retrain `081335` | continued `114710_tools` |
|---|---|---|
| tool-call parse | 85.0% | **90.0%** |
| routing (correct tool) | 42.5% | **50.0%** |
| tool-result grounded | 37.5% | **60.0%** |
| tool-result prose | 52.5% | **77.5%** |
| golden | 5.0% | 2.5% |

**Serve the continued adapter.** It wins on every agent metric. Golden drops by
one question out of 40, which is inside the noise of a 40-item benchmark and is
the prose-forgetting risk this item flagged — worth watching, not acting on yet.

**The ratio hypothesis in the Scope above is falsified.** It predicted that if
the model still skipped tools, the fix was fewer competing prose records. The
mix was rebalanced from ~6% tool records to 33% and **routing did not improve
from the ratio** — what improved was the second half of the loop (grounding
45% -> 60%, prose 60% -> 77.5%). Tool share buys comprehension, not
discrimination. Do not spend another run on the ratio alone.

**The decoy: an untrained tool in the prompt is worse than no tool.**
`query_compounds` is the one tool `generate_tool_call_qa` excludes, but it was
still advertised in `TOOL_SYSTEM_PROMPT`. On the continued model it absorbed
**17 of 40** tool calls, 10 of them copying the prompt's example SMILES
`CC(=O)Oc1ccccc1C(=O)O` verbatim — aspirin, emitted for questions about other
drugs. Unsure which tool to use, the model copies the nearest literal example in
the prompt, and continued training made it more fluent at tool-call syntax
without teaching discrimination, so the decoy got stronger (4 calls -> 17).

Fixed by `advertised: false` in `web/src/tools.ts` (callable, and still in the
in-app manual — only the model prompt drops it), mirrored in the eval's prompt
copy. Effect on the continued model:

| expected tool | with decoy | without |
|---|---|---|
| `query_drug_side_effects` routing | 0/10 | **9/10** |
| `get_compound_by_name` grounding | 0/10 | **10/10** |

`get_compound_by_name` handed a result had been re-emitting its own tool call
instead of answering — 0/10 on both rates, the worst failure found. Deleting one
line from the prompt fixed it outright.

**Read these per-tool, not in aggregate.** Removing the decoy moved the
aggregate grounding rate *down* (77.5% -> 60%) while fixing the worst tool
completely; `draw_molecule` and `query_polypharmacy` regressed and the average
buried both facts. At n=10 per tool the aggregate is the least informative
number produced.

**The system prompt was the bug (2026-09-20). Routing 50% -> 100%.**
`query_polypharmacy` routed 0/10 on every model tried, despite 12,150 training
records using the *exact* benchmark phrasing. It was not phrasing, and not drug
familiarity — "Is it safe to take Warfarin with Aspirin?", both drugs known and
both named in the prompt, failed the same way. Every polypharmacy question
collapsed onto `draw_molecule` with `{"name": "Ibuprofen"}`: the literal
argument of the *last* tool example in the system prompt. Moving
`query_polypharmacy` to last moved the attractor onto it — the failure relocated
rather than resolving, which is what identified the mechanism.

**Training records carry no system prompt at all.** `_tool_call_record` is bare
`### Question` / `### Answer`; its docstring claims "the layout is exactly what
the model sees at inference", and that was false — `app.ts` prepended a
`{role: "system"}` tool list the model had never seen in training. So the tool
list was out-of-distribution text the model copied from rather than reasoned
over. This is the train-serve mismatch this item was always about, sitting in
the one place nobody checked.

Removing it, on `20260920_114710_tools`:

| metric | with system prompt | without |
|---|---|---|
| tool-call parse | 90.0% | **100%** |
| routing (correct tool) | 50.0% | **100%** |
| tool-result grounded | 60.0% | **100%** |
| tool-result prose | 77.5% | **100%** |

40/40 on every tool metric, args correctly lifted from the question
(`{"drug_1": "SIROLIMUS", "drug_2": "warfarin"}`). Landed as: no system message
in `app.ts`, and `system_prompt=""` by default on both benchmarks.
`TOOL_SYSTEM_PROMPT` stays exported and parity-tested for a general
tool-capable model, which does need to be told what the tools are.

**What 100% does not mean.** The rates score one planted marker per case. The
model quotes it correctly every time and then pads with invented detail around
it: handed a result containing a single side effect it answered "TWOSIDES
reports 7 adverse effect(s)", and rendered a planted "tachycardia" as
"tachycardiac arrest". Grounding measures *did it read the value*, not *is the
whole answer faithful*. Faithfulness of the surrounding prose is unmeasured and
is the next thing worth a benchmark.

**Still broken after all of the above:**
- ~~`query_polypharmacy` 0/10 routing~~ — fixed, 10/10.
- ~~`get_compound_by_name` 1/10 routing~~ — fixed, 10/10.
- **Self-continuation persists.** The model still writes its own fabricated
  `### Tool result` after its call, on every tool-call response. Removing the
  system prompt did not touch this — it is the record shape:
  `_tool_call_record` puts call + result + answer in one completion, so the
  model learns to produce the whole transcript. Harmless to scoring and to
  serving (both take the *first* JSON object and discard the tail) but it is
  generated text the streaming path has to suppress. Masking the loss after the
  call turn is the fix, and it is a dataset change.

**A crashed generation used to score as a wrong answer (fixed 2026-09-20).**
`_generate` ignored the subprocess exit code, so when two evals ran at once and
Metal ran out, all 40 completions came back empty and the rates reported
"0.0% routing · 0.0% grounded" — indistinguishable from a real regression. The
only tell was golden dropping to 0.0% in the same run, and golden shares no code
with the tool prompt. It now raises on a non-zero exit or empty stdout: a *bad*
reply is data to score, a *missing* one is a broken run. **Do not run two evals
concurrently** — one exhausts the GPU and the other silently produced garbage
before this fix.

**The benchmarks are deterministic** — two runs give byte-identical responses,
so these before/after deltas are real effects, not sampling noise. Perplexity is
the exception: it samples 50 batches and wobbles ~0.2-0.6 between runs, which is
harmless against a baseline of ~17.

**Two ways to train it, and the cheap one first**
- `continue_tool_training.py` — continue the existing adapter on a tool-heavy
  mix (every tool record plus prose at 1:2) for ~600 iters, into a new run dir
  so the current adapter is untouched. Minutes, not hours, and it attacks the
  ratio problem directly: tool calls are a third of what the model sees instead
  of six percent. Risk is forgetting prose, which is what the prose share and
  the golden benchmark are there to catch.
- A full retrain stays available and unchanged (`finetuning.py`); it is the
  honest end state if continued training drifts.

**Also done:** `num_ctx` 2048 → 8192 in the Modelfile. A 2000-char tool result
plus the system prompt plus history overflowed the old window and silently
dropped the oldest turns — including the tool result the answer depends on.

**Temporary bridge (2026-09-19):** `routeDirectToolCall` + `describeDrawResult`
in `web/src/tools.ts` answer an explicit "draw X" deterministically — tool call
and caption both, with no model turn. It exists because the current fine-tune
not only fails to call tools, it cannot use a tool result it is handed: given
one it echoes the JSON shape (`{ CHEMBL1201082 }`, an invented ebi.ac.uk URL).
Delete it and its call site when this item lands; both are marked TEMPORARY.
A bogus tool name from the model is now fed back as an error rather than shown
to the user (`unknownToolName`).

**Acceptance:** tool-call JSON parses on a stated majority of attempts, measured
on held-out drugs. **Met: 87.5% parse.** But the acceptance criterion turned out
to be the wrong bar — it only covers the first half of the loop. The three
numbers that decide whether the agent is real are routing, grounding and prose.
On `20260920_114710_tools` with no system prompt, all three are **100%** (40/40),
every tool 10/10. The agent is real. What remains is faithfulness of the prose
*around* a correctly-read value, which no benchmark covers yet.

**Files:** `build_drug_interaction_dataset.py`, `finetuning/finetuning.py`,
`finetuning/continue_tool_training.py`, `eval/eval_finetuned_model.py`,
`vector_store/tools.py`, `web/src/tools.ts`

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
