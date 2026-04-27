"""Convert a fused Gemma 3/4 HuggingFace model to GGUF (F16)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import gguf
import mlx.core as mx
import numpy as np

VISIBLE_GEMMA4_TOKENS = {
    "<|channel>",
    "<channel|>",
    "<|tool_call>",
    "<tool_call|>",
    "<|tool_response>",
    "<tool_response|>",
    '<|"|>',
}


def _load_model_config(hf_dir: Path) -> tuple[dict, gguf.MODEL_ARCH, bool]:
    config = json.loads((hf_dir / "config.json").read_text())
    if config.get("model_type") == "gemma4":
        return config["text_config"], gguf.MODEL_ARCH.GEMMA4, True
    if "text_config" in config:
        return config["text_config"], gguf.MODEL_ARCH.GEMMA3, False
    return config, gguf.MODEL_ARCH.GEMMA3, False


def _token_candidates(name: str) -> list[str]:
    candidates = [name]
    for prefix in ("model.language_model.", "language_model.", "model."):
        if name.startswith(prefix):
            candidates.append(name.removeprefix(prefix))
    return list(dict.fromkeys(candidates))


def _tensor_suffix(hf_name: str, tensor: mx.array) -> str | None:
    if hf_name.endswith(".weight"):
        return ".weight"
    if hf_name.endswith(".bias"):
        return ".bias"
    if tensor.ndim == 1:
        return ".weight"
    return ".weight"


def _add_vocab(
    writer: gguf.GGUFWriter,
    hf_dir: Path,
    vocab_size: int,
    gemma4: bool,
) -> None:
    vocab = gguf.LlamaHfVocab(hf_dir)
    tokens: list[bytes] = []
    scores: list[float] = []
    token_types: list[gguf.TokenType] = []

    for text, score, toktype in vocab.all_tokens():
        tokens.append(text)
        scores.append(score)
        if gemma4 and text.decode() in VISIBLE_GEMMA4_TOKENS:
            token_types.append(gguf.TokenType.USER_DEFINED)
        else:
            token_types.append(toktype)

    tokens = tokens[:vocab_size]
    scores = scores[:vocab_size]
    token_types = token_types[:vocab_size]

    writer.add_tokenizer_model("gemma4" if gemma4 else vocab.tokenizer_model)
    writer.add_string("tokenizer.ggml.pre", "gemma4" if gemma4 else "default")
    writer.add_add_space_prefix(False)
    writer.add_add_bos_token(True)
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(token_types)

    sp_vocab = gguf.SpecialVocab(hf_dir, load_merges=True)
    sp_vocab.add_to_gguf(writer, quiet=True)


def _write_gemma3_params(writer: gguf.GGUFWriter, hparams: dict) -> None:
    writer.add_context_length(hparams.get("max_position_embeddings", 131072))
    writer.add_embedding_length(hparams["hidden_size"])
    writer.add_feed_forward_length(hparams["intermediate_size"])
    writer.add_head_count(hparams.get("num_attention_heads", 8))
    writer.add_layer_norm_rms_eps(hparams.get("rms_norm_eps", 1e-6))
    head_dim = hparams.get("head_dim") or (hparams["hidden_size"] // hparams["num_attention_heads"])
    writer.add_key_length(head_dim)
    writer.add_value_length(head_dim)
    rope_params = hparams.get("rope_parameters") or {}
    global_rope_theta = float(
        (rope_params.get("full_attention") or {}).get("rope_theta")
        or hparams.get("rope_theta", 1_000_000.0)
    )
    local_rope_theta = float(
        (rope_params.get("sliding_attention") or {}).get("rope_theta") or 10_000.0
    )
    # Ollama/llama.cpp requires separate keys for global and local (SWA) attention.
    writer.add_float32(f"{writer.arch}.rope.global.freq_base", global_rope_theta)
    writer.add_float32(f"{writer.arch}.rope.local.freq_base", local_rope_theta)
    if (final_logit_softcap := hparams.get("final_logit_softcapping")):
        writer.add_final_logit_softcapping(final_logit_softcap)
    if hparams.get("sliding_window_pattern") != 1:
        writer.add_sliding_window(hparams["sliding_window"])
    writer.add_head_count_kv(hparams.get("num_key_value_heads", 4))


def _write_gemma4_params(writer: gguf.GGUFWriter, hparams: dict) -> None:
    num_kv_shared_layers = hparams["num_kv_shared_layers"]
    swa_layers = [layer_type == "sliding_attention" for layer_type in hparams["layer_types"]]

    writer.add_context_length(hparams.get("max_position_embeddings", 131072))
    writer.add_head_count(hparams.get("num_attention_heads", 8))
    writer.add_layer_norm_rms_eps(hparams.get("rms_norm_eps", 1e-6))
    writer.add_key_length(hparams.get("global_head_dim", hparams.get("head_dim", 256)))
    writer.add_value_length(hparams.get("global_head_dim", hparams.get("head_dim", 256)))
    writer.add_key_length_swa(hparams.get("head_dim", 256))
    writer.add_value_length_swa(hparams.get("head_dim", 256))
    writer.add_shared_kv_layers(num_kv_shared_layers)
    writer.add_embedding_length_per_layer_input(hparams.get("hidden_size_per_layer_input", 0))
    writer.add_sliding_window_pattern(swa_layers)

    rope_params = hparams["rope_parameters"]
    writer.add_rope_freq_base(rope_params["full_attention"]["rope_theta"])
    writer.add_rope_freq_base_swa(rope_params["sliding_attention"]["rope_theta"])
    writer.add_rope_dimension_count(hparams.get("global_head_dim", hparams.get("head_dim", 256)))
    partial_rotary_factor_swa = hparams.get("partial_rotary_factor", 1.0)
    writer.add_rope_dimension_count_swa(int(hparams.get("head_dim", 256) * partial_rotary_factor_swa))

    expert_intermediate_size = hparams.get("expert_intermediate_size") or hparams.get(
        "moe_intermediate_size"
    )
    if expert_intermediate_size is not None:
        writer.add_expert_feed_forward_length(expert_intermediate_size)

    if hparams.get("use_double_wide_mlp", False):
        first_kv_shared_layer_idx = hparams["num_hidden_layers"] - num_kv_shared_layers
        n_ff = hparams["intermediate_size"]
        n_ff_arr = [n_ff if il < first_kv_shared_layer_idx else n_ff * 2 for il in range(hparams["num_hidden_layers"])]
        writer.add_feed_forward_length(n_ff_arr)
    else:
        writer.add_feed_forward_length(hparams["intermediate_size"])

    num_key_value_heads_full = hparams.get("num_global_key_value_heads")
    num_key_value_heads_swa = hparams.get("num_key_value_heads")
    if num_key_value_heads_full is not None and num_key_value_heads_swa is not None:
        writer.add_head_count_kv(
            [num_key_value_heads_swa if is_swa else num_key_value_heads_full for is_swa in swa_layers]
        )

    if (final_logit_softcap := hparams.get("final_logit_softcapping")):
        writer.add_final_logit_softcapping(final_logit_softcap)


def _gemma4_rope_freqs(hparams: dict) -> np.ndarray:
    rope_params_full = hparams["rope_parameters"]["full_attention"]
    partial_rotary_factor_full = rope_params_full["partial_rotary_factor"]
    head_dim_full = hparams["global_head_dim"]
    n_rot_full = int(head_dim_full * partial_rotary_factor_full / 2)
    n_unrot_full = int(head_dim_full / 2) - n_rot_full
    values = [1.0] * n_rot_full + [1e30] * n_unrot_full
    return np.array(values, dtype=np.float32)


def convert(hf_dir: Path, output_path: Path) -> None:
    """Convert a fused HF model to a GGUF F16 file at *output_path*."""

    hparams, arch, is_gemma4 = _load_model_config(hf_dir)
    n_layers: int = hparams["num_hidden_layers"]
    vocab_size: int = hparams["vocab_size"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = gguf.GGUFWriter(output_path, gguf.MODEL_ARCH_NAMES[arch])
    writer.add_name(f"chembl-drug-chat-{'gemma4' if is_gemma4 else 'gemma3'}")
    writer.add_block_count(n_layers)
    writer.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    if is_gemma4:
        _write_gemma4_params(writer, hparams)
    else:
        _write_gemma3_params(writer, hparams)

    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    _add_vocab(writer, hf_dir, vocab_size, is_gemma4)

    name_map = gguf.get_tensor_name_map(arch, n_layers)

    if is_gemma4:
        rope_tensor_name = name_map.get_name("rope_freqs") or "rope_freqs"
        writer.add_tensor(rope_tensor_name, _gemma4_rope_freqs(hparams))

    safetensors_files = sorted(hf_dir.glob("*.safetensors"))
    all_weights: dict[str, mx.array] = {}
    for st_file in safetensors_files:
        all_weights.update(cast(dict[str, mx.array], mx.load(str(st_file))))

    skipped: list[str] = []
    written_gguf_names: set[str] = set()
    embed_weights: np.ndarray | None = None

    for hf_name, tensor in all_weights.items():
        suffix = _tensor_suffix(hf_name, tensor)
        if suffix is None:
            skipped.append(hf_name)
            continue

        gguf_base = None
        for candidate in _token_candidates(hf_name.removesuffix(suffix)):
            gguf_base = name_map.get_name(candidate)
            if gguf_base is not None:
                break

        if gguf_base is None:
            skipped.append(hf_name)
            continue

        if tensor.ndim == 1:
            arr = np.array(tensor.astype(mx.float32), dtype=np.float32)
            if hf_name.endswith("norm.weight"):
                arr = arr + 1.0
        else:
            arr = np.array(tensor.astype(mx.float16), dtype=np.float16)

        gguf_name = gguf_base + suffix
        writer.add_tensor(gguf_name, arr)
        written_gguf_names.add(gguf_name)

        if gguf_name == "token_embd.weight":
            embed_weights = arr

    # Gemma uses tied embeddings: mlx_lm fuse never writes lm_head.weight as a
    # separate tensor, so output.weight is absent from the safetensors.
    # llama.cpp requires it explicitly; write a copy of the embedding matrix.
    if "output.weight" not in written_gguf_names and embed_weights is not None:
        print("  Note: output.weight missing — writing tied copy of token_embd.weight")
        writer.add_tensor("output.weight", embed_weights)

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

    parser = argparse.ArgumentParser(description="Convert Gemma HF model to GGUF F16")
    parser.add_argument("hf_dir", type=Path, help="HuggingFace safetensors directory")
    parser.add_argument("output", type=Path, help="Output .gguf file path")
    args = parser.parse_args()

    convert(args.hf_dir, args.output)
