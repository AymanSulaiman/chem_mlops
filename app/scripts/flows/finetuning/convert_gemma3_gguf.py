"""Convert a fused Gemma 3 HuggingFace model to GGUF (F16).

Uses ``mlx.core`` to load safetensors (no PyTorch required) and the
``gguf`` package for writing the output file.
"""

from __future__ import annotations

import json
from pathlib import Path

import gguf
import mlx.core as mx
import numpy as np
from typing import cast


def convert(hf_dir: Path, output_path: Path) -> None:
    """Convert *hf_dir* (Gemma 3 safetensors) to a GGUF F16 file at *output_path*."""

    config = json.loads((hf_dir / "config.json").read_text())
    n_layers: int = config["num_hidden_layers"]
    hidden_size: int = config["hidden_size"]
    n_heads: int = config["num_attention_heads"]
    n_kv_heads: int = config["num_key_value_heads"]
    ffn_size: int = config["intermediate_size"]
    rms_eps: float = config["rms_norm_eps"]
    rope_theta: float = config.get("rope_theta", 10_000.0)
    ctx_len: int = config["max_position_embeddings"]
    vocab_size: int = config["vocab_size"]
    head_dim: int = config.get("head_dim", hidden_size // n_heads)
    # Gemma 3 specific
    sliding_window: int | None = config.get("sliding_window")
    sliding_window_pattern: int | None = config.get("sliding_window_pattern")
    rope_local_base_freq: float = config.get("rope_local_base_freq", rope_theta)
    query_pre_attn_scalar: int = config.get("query_pre_attn_scalar", head_dim)

    arch_name = gguf.MODEL_ARCH_NAMES[gguf.MODEL_ARCH.GEMMA3]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(output_path, arch_name)

    # --- Model metadata ---
    writer.add_name("chembl-drug-chat-1b")
    writer.add_block_count(n_layers)
    writer.add_context_length(ctx_len)
    writer.add_embedding_length(hidden_size)
    writer.add_feed_forward_length(ffn_size)
    writer.add_head_count(n_heads)
    writer.add_head_count_kv(n_kv_heads)
    # head_dim (256) differs from hidden_size/n_heads (288) — must be explicit.
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    writer.add_layer_norm_rms_eps(rms_eps)
    # Gemma 3 uses separate RoPE frequencies for global and local attention layers.
    # Ollama/llama.cpp expects the "global"/"local" scoped key names.
    writer.add_float32("gemma3.rope.global.freq_base", rope_theta)
    writer.add_float32("gemma3.rope.local.freq_base", rope_local_base_freq)
    # Only write softcapping when the config enables it (non-null, non-zero).
    # The fused model may have this set to None even if the base model used 30.0;
    # writing 30.0 incorrectly distorts every output logit and produces gibberish.
    softcap: float | None = config.get("final_logit_softcapping")
    if softcap:
        writer.add_float32("gemma3.final_logit_softcapping", float(softcap))
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    # Gemma 3 alternates local (sliding-window) and global attention layers.
    if sliding_window is not None:
        writer.add_sliding_window(sliding_window)

    # --- Vocabulary ---
    import os

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    vocab = gguf.LlamaHfVocab(hf_dir)
    tokens: list[bytes] = []
    scores: list[float] = []
    token_types: list[gguf.TokenType] = []
    for text, score, ttype in vocab.all_tokens():
        tokens.append(text)
        scores.append(score)
        token_types.append(ttype)

    # Truncate to the model's vocab_size — the HF tokenizer may include extra
    # added tokens (e.g. <image_soft_token>) that have no corresponding row in
    # the embedding table, causing an out-of-bounds read and NaN logits.
    tokens = tokens[:vocab_size]
    scores = scores[:vocab_size]
    token_types = token_types[:vocab_size]

    writer.add_tokenizer_model(vocab.tokenizer_model)
    writer.add_string("tokenizer.ggml.pre", "default")
    # Gemma3 HF tokenizer doesn't add a leading space before the first token;
    # tell llama.cpp/Ollama to match that behaviour.
    writer.add_add_space_prefix(False)
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)

    sp_vocab = gguf.SpecialVocab(hf_dir, load_merges=True)
    sp_vocab.add_to_gguf(writer, quiet=True)

    # --- Tensors ---
    name_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.GEMMA3, n_layers)

    # Collect all safetensors shards in order.
    safetensors_files = sorted(hf_dir.glob("*.safetensors"))
    all_weights: dict[str, mx.array] = {}
    for st_file in safetensors_files:
        all_weights.update(cast(dict[str, mx.array], mx.load(str(st_file))))

    # Gemma 3 ties input and output embeddings — do NOT write a separate output.weight.
    # llama.cpp handles weight tying automatically when output.weight is absent.
    SKIP_NAMES = frozenset(["lm_head.weight"])

    skipped: list[str] = []
    for hf_name, tensor in all_weights.items():
        if hf_name in SKIP_NAMES:
            skipped.append(hf_name)
            continue
        # Strip suffix to look up base name in the tensor map.
        if hf_name.endswith(".weight"):
            base = hf_name[: -len(".weight")]
            suffix = ".weight"
        elif hf_name.endswith(".bias"):
            base = hf_name[: -len(".bias")]
            suffix = ".bias"
        else:
            skipped.append(hf_name)
            continue

        gguf_base = name_map.get_name(base)
        if gguf_base is None:
            skipped.append(hf_name)
            continue

        # 1D tensors (RMS-norm scale vectors) must be stored as F32 — ggml's
        # binary-ops (div/mul) require all operands to be the same type, and
        # the norm path uses F32 activations paired with these weight vectors.
        #
        # Gemma3 RMSNorm computes: output * (1.0 + weight)
        # The HF safetensors store only the `weight` offset; llama.cpp applies
        # the stored value directly as a scale, so we must add +1.0 here.
        if tensor.ndim == 1:
            arr = np.array(tensor.astype(mx.float32), dtype=np.float32)
            if hf_name.endswith("norm.weight"):
                arr = arr + 1.0
        else:
            arr = np.array(tensor.astype(mx.float16), dtype=np.float16)
        writer.add_tensor(gguf_base + suffix, arr)

    if skipped:
        print(f"  Note: skipped {len(skipped)} unrecognised tensor(s): {skipped[:5]}")

    print(f"  Writing GGUF to {output_path} …")
    writer.write_header_to_file(output_path)
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()
    print(f"  Done — {output_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert Gemma 3 HF model to GGUF F16")
    parser.add_argument("hf_dir", type=Path, help="HuggingFace safetensors directory")
    parser.add_argument("output", type=Path, help="Output .gguf file path")
    args = parser.parse_args()

    convert(args.hf_dir, args.output)
