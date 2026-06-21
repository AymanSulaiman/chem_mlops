#!/usr/bin/env bash
set -euo pipefail

# ── guards ────────────────────────────────────────────────────────────────────
if [[ "$(uname)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
  echo "ERROR: This installer requires macOS on Apple Silicon (arm64)." >&2
  exit 1
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> ChEMBL Drug Chat — installer"
echo "    Repo: $REPO_DIR"
echo

# ── Homebrew ──────────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
  echo "==> Installing Homebrew..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
fi

# ── system deps ───────────────────────────────────────────────────────────────
echo "==> Installing uv, bun, ollama via Homebrew..."
brew install uv bun ollama

# ── Python env ────────────────────────────────────────────────────────────────
echo "==> Installing Python dependencies (uv sync)..."
cd "$REPO_DIR"
uv sync

# ── Bun web app ───────────────────────────────────────────────────────────────
echo "==> Installing Bun dependencies (web/)..."
cd "$REPO_DIR/web"
bun install

cd "$REPO_DIR"

# ── Ollama base model ─────────────────────────────────────────────────────────
echo "==> Pulling gemma3:1b base model (needed for RAG mode)..."
# Start ollama serve in background if not already running
if ! pgrep -x ollama &>/dev/null; then
  ollama serve &>/tmp/ollama-install.log &
  OLLAMA_PID=$!
  sleep 3  # give it a moment to bind
fi
ollama pull gemma3:1b

# ── Pipeline ──────────────────────────────────────────────────────────────────
echo
echo "==> Running the full pipeline (this will take 2–3 hours on first run)..."
echo "    Stages: download ChEMBL → transform → build datasets → ingest LanceDB"
echo "            → fine-tune Gemma 3 1B → eval → export to Ollama"
echo
uv run python -m app.orchestration.chembl_drug_chat_pipeline

# ── done ──────────────────────────────────────────────────────────────────────
echo
echo "✓ Installation complete."
echo
echo "  Start the web app:"
echo "    ollama serve &"
echo "    cd web && bun run dev"
echo
echo "  Or open the Dagster UI:"
echo "    dagster dev -w deployments/workspace.yaml"
