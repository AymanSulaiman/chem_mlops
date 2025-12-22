import polars as pl
from pathlib import Path
import subprocess
from datetime import datetime

HF_MODEL_ID = "google/gemma-3-1b-pt"
DATA_DIR = Path("data/llm_finetune")

BATCH_SIZE = 2
NUM_LAYERS = 8
ITERS = 15
LEARNING_RATE = 1e-5


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    display_cwd = str(cwd) if cwd else "(current)"
    print(f"\n>>> Running: {' '.join(cmd)}")
    print(f"    in: {display_cwd}\n")

    result = subprocess.run(
        cmd,
        check=False,
        cwd=cwd,
    )

    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)



def convert_to_mlx(hf_model_id: str, mlx_model_dir: Path) -> Path:
    """Convert HF model to MLX format."""
    mlx_model_dir.parent.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "python",
            "-m",
            "mlx_lm",
            "convert",
            "--hf-path",
            hf_model_id,
            "--mlx-path",
            str(mlx_model_dir),
            "--q-bits",
            "4",
            "--q-group-size",
            "64",
        ]
    )

    return mlx_model_dir


def finetune_lora(
    mlx_model_dir: Path,
    adapter_dir: Path,
    data_dir: Path = DATA_DIR,
    batch_size: int = BATCH_SIZE,
    num_layers: int = NUM_LAYERS,
    iters: int = ITERS,
    learning_rate: float = LEARNING_RATE,
):
    #TODO: Find a way to save the logs to analyse the loss, tokens/sec etc. example of logs below
    """
    Iter 393: Train loss 0.568, Learning Rate 1.000e-05, It/sec 0.061, Tokens/sec 185.796, Trained Tokens 682568, Peak mem 25.933 GB
    Iter 394: Train loss 0.603, Learning Rate 1.000e-05, It/sec 0.175, Tokens/sec 218.927, Trained Tokens 683816, Peak mem 25.933 GB
    Iter 395: Train loss 0.415, Learning Rate 1.000e-05, It/sec 0.195, Tokens/sec 249.194, Trained Tokens 685091, Peak mem 25.933 GB
    Iter 396: Train loss 0.573, Learning Rate 1.000e-05, It/sec 0.078, Tokens/sec 193.704, Trained Tokens 687569, Peak mem 25.933 GB
    Iter 397: Train loss 0.593, Learning Rate 1.000e-05, It/sec 0.111, Tokens/sec 196.816, Trained Tokens 689347, Peak mem 25.933 GB
    Iter 398: Train loss 0.616, Learning Rate 1.000e-05, It/sec 0.162, Tokens/sec 215.451, Trained Tokens 690679, Peak mem 25.933 GB
    Iter 399: Train loss 0.594, Learning Rate 1.000e-05, It/sec 0.125, Tokens/sec 227.212, Trained Tokens 692493, Peak mem 25.933 GB
    """

    #TODO [WARNING] Some sequences are longer than 2048 tokens. The longest sentence 5161 will be truncated to 2048. Consider pre-splitting your data to save memory. <- figure this out

    """
    Launch LoRA fine-tuning using updated mlx_lm CLI.
    """
    adapter_dir.mkdir(parents=True, exist_ok=True)

    _run(
        [
            "python",
            "-m",
            "mlx_lm",
            "lora",
            "--model",
            str(mlx_model_dir),
            "--train",
            "--data",
            str(data_dir),
            "--fine-tune-type",
            "lora",
            "--batch-size",
            str(batch_size),
            "--num-layers",
            str(num_layers),
            "--iters",
            str(iters),
            "--learning-rate",
            str(learning_rate),
            "--adapter-path",
            str(adapter_dir),
            "--steps-per-report", "1",
        ]
    )

    return adapter_dir

def save_to_ollama(
    mlx_model_dir: Path,
    adapter_dir: Path,
    model_name: str,
    llama_cpp_dir: Path = Path("llama.cpp"),
) -> None:
    """Convert fused MLX model to Ollama-compatible GGUF format."""
    import shutil
    import sys
    from mlx_lm import load
    from safetensors.torch import save_file
    import torch
    import json
    
    fused_dir = adapter_dir.parent / "fused_model"
    hf_dir = adapter_dir.parent / "fused_model_hf"
    
    # Clean up any existing directories from previous runs
    for dir_path in [fused_dir, hf_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    
    fused_dir.mkdir(parents=True, exist_ok=True)
    hf_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Fuse adapter with base model in MLX format
    print("\n==> Fusing LoRA adapter with base model...")
    sys.stdout.flush()
    _run([
        "python", "-m", "mlx_lm", "fuse",
        "--model", str(mlx_model_dir),
        "--adapter-path", str(adapter_dir),
        "--save-path", str(fused_dir),
        "--de-quantize",
    ])
    
    # 2. Load fused MLX model and export to HuggingFace format
    print("\n==> Converting MLX weights to HuggingFace format...")
    sys.stdout.flush()
    model, tokenizer = load(str(fused_dir))
    
    # Flatten nested parameters dict and convert to PyTorch
    def flatten_params(params, prefix=""):
        """Recursively flatten nested parameter dictionaries and lists."""
        flat = {}
        
        if isinstance(params, dict):
            for key, value in params.items():
                full_key = f"{prefix}.{key}" if prefix else key
                flat.update(flatten_params(value, full_key))
        elif isinstance(params, list):
            for idx, value in enumerate(params):
                full_key = f"{prefix}.{idx}" if prefix else str(idx)
                flat.update(flatten_params(value, full_key))
        elif hasattr(params, 'tolist'):
            # It's an MLX array - convert to torch tensor
            numpy_array = params.tolist()
            flat[prefix] = torch.tensor(numpy_array, dtype=torch.bfloat16)
        
        return flat
    
    state_dict = flatten_params(model.parameters())
    
    # Save as safetensors (HF format)
    print(f"Saving {len(state_dict)} parameters to safetensors...")
    sys.stdout.flush()
    save_file(state_dict, hf_dir / "model.safetensors")
    
    # Copy config and tokenizer files
    print("\n==> Copying config files...")
    sys.stdout.flush()
    for file in ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]:
        src = fused_dir / file
        if src.exists():
            shutil.copy(src, hf_dir / file)
            print(f"  ✓ Copied {file}")
            sys.stdout.flush()
        else:
            print(f"  ✗ Missing {file}")
            sys.stdout.flush()
    
    # Fix vocab_size in config.json to match tokenizer
    print("\n==> Fixing vocab_size in config.json...")
    sys.stdout.flush()
    
    config_path = hf_dir / "config.json"
    tokenizer_path = hf_dir / "tokenizer.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found at {config_path}")
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found at {tokenizer_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    with open(tokenizer_path, 'r') as f:
        tokenizer_data = json.load(f)
    
    # Get max token ID from vocab
    vocab = tokenizer_data.get('model', {}).get('vocab', {})
    if vocab:
        max_token_id = max(vocab.values())
        required_vocab_size = max_token_id + 1
        
        print(f"  Current vocab_size: {config.get('vocab_size')}")
        print(f"  Max token ID: {max_token_id}")
        print(f"  Required vocab_size: {required_vocab_size}")
        sys.stdout.flush()
        
        config['vocab_size'] = required_vocab_size
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"  ✓ Updated vocab_size to {required_vocab_size}")
        sys.stdout.flush()
    else:
        raise ValueError("Could not find vocab in tokenizer.json!")
    
    # 3. Clone llama.cpp if needed
    if not llama_cpp_dir.exists():
        print("\n==> Cloning llama.cpp...")
        sys.stdout.flush()
        _run(["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llama_cpp_dir)])
    
    # 4. Convert HF format to GGUF
    gguf_path = adapter_dir.parent / "model-f16.gguf"
    print("\n==> Converting to GGUF format...")
    sys.stdout.flush()
    _run([
        "python", str(llama_cpp_dir / "convert_hf_to_gguf.py"),
        str(hf_dir),
        "--outfile", str(gguf_path),
        "--outtype", "f16"
    ])
    
    # 5. Quantize to Q4_K_M
    quantized_path = adapter_dir.parent / "model-q4_k_m.gguf"
    quantize_bin = llama_cpp_dir / "llama-quantize"
    
    if not quantize_bin.exists():
        print("\n==> Building llama-quantize...")
        sys.stdout.flush()
        _run(["make", "llama-quantize"], cwd=llama_cpp_dir)
    
    print("\n==> Quantizing model to Q4_K_M...")
    sys.stdout.flush()
    _run([
        str(quantize_bin),
        str(gguf_path),
        str(quantized_path),
        "Q4_K_M"
    ])
    
    # 6. Create Modelfile
    modelfile_path = adapter_dir.parent / "Modelfile"
    modelfile_path.write_text(f"""FROM {quantized_path}
TEMPLATE \"\"\"{{{{ if .System }}}}<|im_start|>system
{{{{ .System }}}}<|im_end|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
{{{{ end }}}}<|im_start|>assistant
\"\"\"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
""")
    
    # 7. Import to Ollama
    print(f"\n==> Creating Ollama model '{model_name}'...")
    sys.stdout.flush()
    _run([
        "ollama", "create", model_name,
        "-f", str(modelfile_path)
    ])
    
    print(f"\n{'=' * 60}")
    print(f"✓ Model '{model_name}' successfully created in Ollama!")
    print(f"  F16 GGUF: {gguf_path}")
    print(f"  Q4_K_M GGUF: {quantized_path}")
    print(f"{'=' * 60}\n")
    sys.stdout.flush()
    
def gemma3_chembl_toon_finetune_flow(
    hf_model_id: str = HF_MODEL_ID,
    data_dir: str = str(DATA_DIR),
    run_name: str | None = None,
) -> None:
    """
    Minimal finetuning pipeline:

    1. Convert Gemma 3 HF model -> MLX format (4-bit quantized).
    2. Run LoRA fine-tuning on the ChEMBL→TOON JSONL dataset.

    Args:
        hf_model_id: HuggingFace model ID
        data_dir: Path to training data
        run_name: Optional custom run name, defaults to timestamp
    """
    # Generate timestamped run directory
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = Path("artifacts") / run_name
    mlx_model_dir = run_dir / "mlx" / "gemma-3-4b-pt-mlx"
    adapter_dir = run_dir / "adapters" / "gemma3-4b-pt-chembl-toon"

    print(f"\n{'=' * 60}")
    print(f"Starting finetuning run: {run_name}")
    print(f"Run directory: {run_dir}")
    print(f"{'=' * 60}\n")

    mlx_model_dir = convert_to_mlx(hf_model_id, mlx_model_dir)
    adapter_dir = finetune_lora(mlx_model_dir, adapter_dir, Path(data_dir))

    print(f"\n{'=' * 60}")
    print(f"Finetuning complete!")
    print(f"MLX model: {mlx_model_dir}")
    print(f"LoRA adapter: {adapter_dir}")
    print(f"{'=' * 60}\n")
    
    save_to_ollama(
        mlx_model_dir=mlx_model_dir,
        adapter_dir=adapter_dir,
        model_name=f"gemma3-chembl-{run_name}"
    )

if __name__ == "__main__":
    gemma3_chembl_toon_finetune_flow()
