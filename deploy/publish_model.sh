#!/usr/bin/env bash
# Publish a fine-tuned model (GGUF + Modelfile + manifest) to gs://$BUCKET/models/<run>/.
#
#   BUCKET=my-bucket ./deploy/publish_model.sh [artifacts/<run>]
#
# Defaults to the latest run in artifacts/. Re-run after each new fine-tune +
# export_to_ollama; then rebuild the Ollama image from that prefix.
# Rebuild on any machine with:  ollama create chembl-drug-chat:270m -f Modelfile
set -euo pipefail

: "${BUCKET:?set BUCKET to the GCS bucket name}"
run_dir="${1:-$(ls -d artifacts/*/ | sort | tail -1)}"
run_dir="${run_dir%/}"
run="$(basename "$run_dir")"
src="$run_dir/ollama"
eval_metrics="data/eval/$run/finetuned_eval_metrics.json"

[[ -f "$src/chembl-drug-chat.gguf" && -f "$src/Modelfile" ]] \
  || { echo "No GGUF/Modelfile in $src. Run export_to_ollama first." >&2; exit 1; }
head -1 "$src/Modelfile" | grep -q '^FROM \./' \
  || { echo "$src/Modelfile has an absolute FROM; re-export or fix it to 'FROM ./chembl-drug-chat.gguf'." >&2; exit 1; }

# Where the model came from. Eval metrics are embedded if the eval was run.
manifest="$(mktemp)"
trap 'rm -f "$manifest"' EXIT
cat > "$manifest" <<EOF
{
  "run": "$run",
  "git_commit": "$(git rev-parse HEAD)",
  "gguf_sha256": "$(shasum -a 256 "$src/chembl-drug-chat.gguf" | cut -d' ' -f1)",
  "published_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "eval": $(cat "$eval_metrics" 2>/dev/null || echo null)
}
EOF

dest="gs://$BUCKET/models/$run"
gcloud storage cp "$src/chembl-drug-chat.gguf" "$src/Modelfile" "$dest/"
gcloud storage cp "$manifest" "$dest/manifest.json"
echo "Published $dest"
