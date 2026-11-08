"""Full fine-tune of Gemma 3 270M on the ChEMBL QA dataset with the Hugging Face Trainer.

Runs wherever PyTorch does: CUDA (a GCP L4 or T4), Apple MPS, or CPU (slow).

    uv run python -m app.scripts.flows.finetuning.finetuning --run-name gemma270m_v1

Re-running with the same --run-name resumes from the last checkpoint, which is
what a preempted or dropped run needs. Output: artifacts/<run>/model/ (Hugging Face
format), which export_to_ollama turns into a GGUF.

ponytail: full fine-tune rather than LoRA. At 270M parameters the whole model
plus AdamW state is ~4.5 GB, so there is no adapter, no merge step and no PEFT
dependency. Revisit (LoRA) if the base model grows past ~1B again.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.scripts.flows.finetuning.export_to_ollama import (
    ARTIFACTS_DIR,
    CHECKPOINTS_SUBDIR,
    MODEL_SUBDIR,
    export_to_ollama,
)

HF_MODEL_ID = "google/gemma-3-270m"
DATA_DIR = Path("data/llm_finetune")

# Sized to fit a 16 GB T4 (no bf16, so fp32), the smallest GPU in deploy/train.sky.yaml. The 262K-token vocabulary
# makes the logits the largest tensor (batch x seq x 262144 floats), so the
# micro-batch stays small and gradient accumulation makes up the rest.
# Gemma's 262k-token vocabulary makes the logits the biggest tensor in training
# (batch x seq x 262k floats). Measured on an M1 Pro: batch 4 peaked at 26 GB and
# ran out of memory on the full data; batch 1 peaks at ~12 GB and is faster
# (less padding). Same effective batch of 16.
BATCH_SIZE = 1
GRAD_ACCUM = 16
ITERS = 3000  # optimizer steps
LEARNING_RATE = 5e-5
MAX_SEQ_LEN = 1024
STEPS_PER_REPORT = 25
STEPS_PER_EVAL = 200
EVAL_RECORDS = 500  # the whole valid split would take longer than the training
SAVE_EVERY = 200  # a disconnect costs at most 200 steps


def split_long_sequences(
    data_dir: Path,
    hf_model_id: str,
    max_seq_len: int = MAX_SEQ_LEN,
) -> None:
    """
    Split JSONL records whose token length exceeds max_seq_len into smaller chunks.

    Operates in-place on train.jsonl / valid.jsonl / test.jsonl.
    Without this the collator truncates long records and the tail is never trained on.
    """
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import AutoTokenizer

    print("Loading tokenizer for sequence pre-splitting...")
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)

    for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
        jsonl_path = data_dir / split
        if not jsonl_path.exists():
            continue

        records = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
        output: list[dict] = []
        long_count = 0

        for record in records:
            text: str = record.get("text", "")
            if not text:
                output.append(record)
                continue

            token_ids = tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) <= max_seq_len:
                output.append(record)
                continue

            long_count += 1
            for start in range(0, len(token_ids), max_seq_len):
                chunk_ids = token_ids[start : start + max_seq_len]
                chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=False)
                output.append({**record, "text": chunk_text})

        if long_count:
            print(f"  {split}: split {long_count} long record(s) -> {len(output)} total")
            jsonl_path.write_text("\n".join(json.dumps(r) for r in output) + "\n")
        else:
            print(f"  {split}: all {len(records)} records within {max_seq_len} tokens")


def _read_texts(path: Path, limit: int | None = None) -> list[str]:
    """The ``text`` field of each JSONL record, optionally the first *limit*."""
    texts: list[str] = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                texts.append(json.loads(line)["text"])
                if limit is not None and len(texts) >= limit:
                    break
    return texts


def collate(tokenizer: Any, texts: list[str], max_seq_len: int = MAX_SEQ_LEN) -> dict[str, Any]:
    """Tokenize a batch for causal LM training, loss on every non-padding token.

    EOS is appended so the model learns to stop after an answer; padding is
    masked out of the loss with -100.
    """
    enc = tokenizer(
        [text + tokenizer.eos_token for text in texts],
        truncation=True,
        max_length=max_seq_len,
        padding=True,
        return_tensors="pt",
    )
    enc["labels"] = enc["input_ids"].masked_fill(enc["attention_mask"] == 0, -100)
    return enc


def finetune(
    base_model: str | Path,
    run_dir: Path,
    data_dir: Path = DATA_DIR,
    iters: int = ITERS,
    learning_rate: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
    max_seq_len: int = MAX_SEQ_LEN,
) -> Path:
    """Fine-tune *base_model* (HF id or local folder) on *data_dir*; return the model folder.

    Resumes from ``run_dir/checkpoints`` when a checkpoint is there.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    model_dir = run_dir / MODEL_SUBDIR
    checkpoint_dir = run_dir / CHECKPOINTS_SUBDIR

    tokenizer = AutoTokenizer.from_pretrained(str(base_model))
    # Eager attention is what Gemma 3 recommends for training.
    model = AutoModelForCausalLM.from_pretrained(
        str(base_model), dtype=torch.float32, attn_implementation="eager"
    )
    model.config.use_cache = False  # incompatible with gradient checkpointing

    train = _read_texts(data_dir / "train.jsonl")
    valid_path = data_dir / "valid.jsonl"
    valid = _read_texts(valid_path, limit=EVAL_RECORDS) if valid_path.exists() else None

    # bf16 needs Ampere or newer; a T4 would only emulate it. Gemma overflows in
    # fp16, so anything older trains in fp32.
    bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8

    args = TrainingArguments(
        output_dir=str(checkpoint_dir),
        max_steps=iters,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=min(100, iters // 10),
        bf16=bf16,
        gradient_checkpointing=True,
        logging_steps=STEPS_PER_REPORT,
        eval_strategy="steps" if valid else "no",
        eval_steps=STEPS_PER_EVAL,
        save_steps=SAVE_EVERY,
        save_total_limit=1,  # a checkpoint is ~3.5 GB with optimizer state
        remove_unused_columns=False,  # the dataset is plain strings
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=valid,
        data_collator=lambda texts: collate(tokenizer, texts, max_seq_len),
    )

    resume = any(checkpoint_dir.glob("checkpoint-*"))
    print(f"Training {base_model} for {iters} steps{' (resuming)' if resume else ''}...")
    trainer.train(resume_from_checkpoint=True if resume else None)

    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    log_path = run_dir / "logs" / "train_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(trainer.state.log_history, indent=2))
    return model_dir


def gemma3_chembl_toon_finetune_flow(
    hf_model_id: str = HF_MODEL_ID,
    data_dir: str = str(DATA_DIR),
    run_name: str | None = None,
    export: bool = True,
    iters: int = ITERS,
) -> Path:
    """
    Fine-tune pipeline:
    1. Pre-split training sequences longer than MAX_SEQ_LEN.
    2. Full fine-tune of the base model (resumable).
    3. Export to Ollama — only when *export* is true.

    Pass ``export=False`` when something downstream evaluates this model first.
    The Dagster pipeline does: it continues this run on a tool-heavy mix,
    evaluates that, and exports only if the gate passes. Exporting here would
    publish an unevaluated model and leave it served if the gate then failed.

    Returns:
        The run directory, so a caller can evaluate or continue this run
        without guessing at "the latest directory in artifacts/".
    """
    run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS_DIR / run_name

    print(f"\n{'=' * 60}")
    print(f"Starting finetuning run: {run_name}")
    print(f"Run directory: {run_dir}")
    print(f"{'=' * 60}\n")

    split_long_sequences(Path(data_dir), hf_model_id)
    model_dir = finetune(hf_model_id, run_dir, Path(data_dir), iters=iters)

    print(f"\nFinetuning complete. Model: {model_dir}\n")

    if export:
        export_to_ollama(run_dir=run_dir, force=True)

    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full fine-tune of Gemma 3 270M.")
    parser.add_argument("--run-name", default=None, help="Reuse a name to resume that run")
    parser.add_argument("--base-model", default=HF_MODEL_ID)
    parser.add_argument("--data-dir", default=str(DATA_DIR))
    parser.add_argument("--iters", type=int, default=ITERS)
    parser.add_argument("--no-export", action="store_true", help="Skip the Ollama export")
    args = parser.parse_args()
    gemma3_chembl_toon_finetune_flow(
        hf_model_id=args.base_model,
        data_dir=args.data_dir,
        run_name=args.run_name,
        export=not args.no_export,
        iters=args.iters,
    )
