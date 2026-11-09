#!/usr/bin/env bash
# One command, data to running app:
#   ./install.sh
# installs tools → downloads ChEMBL + TWOSIDES → builds datasets and LanceDB →
# fine-tunes Gemma 3 270M → exports Q4_K_M to Ollama → starts the Docker containers.
# Safe to re-run: finished downloads are reused.
set -euo pipefail

if [[ "$(uname)" != "Darwin" ]]; then
  echo "ERROR: This installer uses Homebrew and targets macOS." >&2
  exit 1
fi
if ! docker info &>/dev/null; then
  echo "ERROR: Docker isn't running. Start Docker Desktop and re-run." >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
CHEMBL_VERSION=37

echo "==> ChEMBL Drug Chat: $REPO_DIR"

# ── tools ─────────────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo "==> Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi
echo "==> Installing uv, ollama, llama.cpp, aria2..."
brew install uv ollama llama.cpp aria2
uv sync

# Gemma is gated on Hugging Face: accept the license on its model page first.
if ! uv run hf auth whoami &>/dev/null; then
  echo "==> Log in to Hugging Face (needed to download Gemma 3)."
  uv run hf auth login
fi

# The export registers the model with the local Ollama.
if ! pgrep -x ollama &>/dev/null; then
  ollama serve &>/tmp/ollama-install.log &
  sleep 3
fi

# ── ChEMBL download ───────────────────────────────────────────────────────────
# EBI throttles each connection (~50 kB/s), so fetch with 16 in parallel;
# collect_data then extracts this archive instead of downloading it again.
if [[ ! -d "data/chembl_$CHEMBL_VERSION/chembl_${CHEMBL_VERSION}_sqlite" ]]; then
  echo "==> Downloading ChEMBL $CHEMBL_VERSION (5.8 GB)..."
  aria2c -x16 -s16 -c -d data \
    "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/chembl_${CHEMBL_VERSION}_sqlite.tar.gz"
fi

# ── pipeline ──────────────────────────────────────────────────────────────────
echo "==> Running the pipeline: transform → datasets → LanceDB → fine-tune → export"
uv run python -m app.orchestration.chembl_drug_chat_pipeline

# ── deploy ────────────────────────────────────────────────────────────────────
run_dir="$(ls -d artifacts/*_tools | sort | tail -1)"
echo "==> Starting the containers with $run_dir"
MODEL_DIR="$run_dir/ollama" docker compose up -d --build --wait

echo
echo "✓ Done. Chat at http://localhost:3000"
echo "  Logs: docker compose logs -f    Stop: docker compose down"
