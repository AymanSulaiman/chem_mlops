#!/usr/bin/env bash
# Create (first run) and deploy the chat app to a GCE VM (e2-medium), from your
# working copy: uncommitted changes included, nothing pushed to GitHub.
#
#   ./deploy/vm.sh               # the chem-demo VM
#
# Re-run to redeploy: code is copied fresh, the model and LanceDB sync from the
# bucket (unchanged files are skipped), and only changed images rebuild.
# Public address: if deploy/<vm>.env exists (gitignored) with DOMAIN and
# GOOGLE_CLIENT_ID, it is copied to the VM, ports 80/443 are opened, and Caddy
# serves https://$DOMAIN with sign-in. Without it the VM opens no ports beyond
# SSH; open the app through a tunnel:
#   gcloud compute ssh chem-demo --zone us-central1-a -- -N -L 3000:localhost:3000
#
# The model must be published first: BUCKET=chem-mlops-train ./deploy/publish_model.sh artifacts/<run>
set -euo pipefail

VM="${1:-chem-demo}"
ZONE="${ZONE:-us-central1-a}"         # the bucket's region: copying from it is free
BUCKET="${BUCKET:-chem-mlops-train}"
RUN="${RUN:-gcp_v1_tools}"            # gs://$BUCKET/models/$RUN
# ponytail: 50 GB because LanceDB is 24 GB; 30 GB (the free tier) once its
# fingerprints are bit-packed.
DISK_GB="${DISK_GB:-50}"
# 4 GB and 1 sustained core (bursting to 2): room for 3 chats at once
# (MAX_USERS in deploy/<vm>.env). Changing it resizes the VM, a few minutes down.
MACHINE_TYPE="${MACHINE_TYPE:-e2-medium}"
# The account on the VM whose home holds the model, LanceDB and the checkout.
# gcloud would use the local username ("runner" in CI): a fresh home that
# re-downloads 24 GB and fills the disk. Pin it so every deploy shares one.
VM_USER="${VM_USER:-$USER}"

cd "$(dirname "$0")/.."
# Keep-alives notice a dropped connection (laptop asleep, network blip) instead of hanging.
ssh_vm() { gcloud compute ssh "$VM_USER@$VM" --zone "$ZONE" --quiet -- -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "$@"; }
SETTINGS="deploy/$VM.env"
DOMAIN="$( [[ -f "$SETTINGS" ]] && sed -n 's/^DOMAIN=//p' "$SETTINGS" || true)"

if ! gcloud compute instances describe "$VM" --zone "$ZONE" &>/dev/null; then
  echo "==> Creating $VM ($MACHINE_TYPE, ${DISK_GB} GB disk, $ZONE)"
  gcloud compute instances create "$VM" --zone "$ZONE" \
    --machine-type "$MACHINE_TYPE" \
    --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-standard \
    --scopes storage-rw \
    --metadata-from-file startup-script=<(cat <<'EOF'
#!/bin/bash
# Docker + compose, and 2 GB swap: 1 GB RAM is tight for the model plus a tool call.
command -v docker >/dev/null || curl -fsSL https://get.docker.com | sh
if [ ! -f /swapfile ]; then
  fallocate -l 2G /swapfile && chmod 600 /swapfile && mkswap /swapfile
  echo "/swapfile none swap sw 0 0" >> /etc/fstab
fi
swapon -a
EOF
)
else
  if [[ "$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(machineType.basename())')" != "$MACHINE_TYPE" ]]; then
    echo "==> Resizing $VM to $MACHINE_TYPE (stops it for a few minutes)"
    gcloud compute instances stop "$VM" --zone "$ZONE"
    gcloud compute instances set-machine-type "$VM" --zone "$ZONE" --machine-type "$MACHINE_TYPE"
  fi
  if [[ "$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)')" != RUNNING ]]; then
    echo "==> Starting $VM"
    gcloud compute instances start "$VM" --zone "$ZONE"
  fi
fi

if [[ -n "$DOMAIN" ]]; then
  echo "==> Opening ports 80 and 443 for $DOMAIN"
  gcloud compute firewall-rules describe chem-web &>/dev/null \
    || gcloud compute firewall-rules create chem-web --network default \
         --allow tcp:80,tcp:443 --target-tags chem-web --source-ranges 0.0.0.0/0
  gcloud compute instances add-tags "$VM" --zone "$ZONE" --tags chem-web
fi

echo "==> Waiting for SSH and Docker (first boot installs Docker, ~2 min)"
until ssh_vm 'docker compose version' &>/dev/null; do sleep 10; done

echo "==> Copying the working copy"
git ls-files -z --cached --others --exclude-standard \
  | COPYFILE_DISABLE=1 tar -czf - --null -T - \
  | ssh_vm 'rm -rf ~/chem_mlops.new && mkdir ~/chem_mlops.new && tar -xzf - -C ~/chem_mlops.new \
            && rm -rf ~/chem_mlops && mv ~/chem_mlops.new ~/chem_mlops'
if [[ -f "$SETTINGS" ]]; then
  # docker compose reads .env from the project directory.
  ssh_vm 'umask 077 && cat > ~/chem_mlops/.env' < "$SETTINGS"
fi
PROFILE="$( [[ -n "$DOMAIN" ]] && echo "--profile public" || true)"

echo "==> Uploading the usage log to gs://$BUCKET/usage/ every 10 minutes"
# The web container appends to the usage_logs volume; this timer copies the file
# to the bucket so DuckDB can read it (README "Usage log"). Needs the VM's
# storage-rw scope.
ssh_vm "sudo tee /etc/systemd/system/chem-usage-upload.service >/dev/null <<'EOF'
[Unit]
Description=Copy the chat usage log to gs://$BUCKET/usage/

[Service]
Type=oneshot
ExecStart=/bin/sh -c '[ ! -f /var/lib/docker/volumes/chem_mlops_usage_logs/_data/usage.jsonl ] || gcloud storage cp --quiet /var/lib/docker/volumes/chem_mlops_usage_logs/_data/usage.jsonl gs://$BUCKET/usage/usage.jsonl'
EOF
sudo tee /etc/systemd/system/chem-usage-upload.timer >/dev/null <<'EOF'
[Unit]
Description=Copy the chat usage log to GCS every 10 minutes

[Timer]
OnBootSec=5min
OnUnitActiveSec=10min

[Install]
WantedBy=timers.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now chem-usage-upload.timer"

# The sync and the image builds run detached on the VM (they take a while on
# a small VM), so a dropped connection doesn't kill them. This script follows
# ~/deploy.log; if it gets cut off, check on it with:
#   gcloud compute ssh chem-demo --zone us-central1-a -- tail -f deploy.log
echo "==> Syncing the model ($RUN) and LanceDB from gs://$BUCKET, then building and starting the app"
# The settings go in as a header; the rest is a quoted heredoc, so it runs on
# the VM exactly as written here.
{
  printf 'BUCKET=%q\nRUN=%q\nPROFILE=%q\n' "$BUCKET" "$RUN" "$PROFILE"
  cat <<'EOF'
set -e
mkdir -p ~/model ~/data/lancedb
gcloud storage rsync --recursive --delete-unmatched-destination-objects "gs://$BUCKET/models/$RUN" ~/model
gcloud storage rsync --recursive --delete-unmatched-destination-objects "gs://$BUCKET/data/lancedb" ~/data/lancedb
cd ~/chem_mlops
compose() { sudo env PWD="$PWD" MODEL_DIR="$HOME/model" LANCEDB_HOST_DIR="$HOME/data/lancedb" docker compose $PROFILE "$@"; }
# The Ollama image only changes with the model; rebuilding it unpacks a 3 GB
# base image, which a small disk can't always fit. Rebuild it only when needed.
model_sha="$(sha256sum ~/model/chembl-drug-chat.gguf | cut -d' ' -f1)"
if ! sudo docker image inspect chem_mlops-ollama >/dev/null 2>&1 || [ "$model_sha" != "$(cat ~/.model.sha 2>/dev/null)" ]; then
  compose build ollama web
  echo "$model_sha" > ~/.model.sha
else
  compose build web
fi
compose up -d --no-build --wait
sudo docker builder prune -f   # build leftovers, several GB on a 50 GB disk
sudo docker image prune -f
sudo docker ps --format '{{.Names}}  {{.Status}}'
EOF
} | ssh_vm 'cat > ~/deploy.sh'
ssh_vm 'rm -f ~/deploy.log; setsid nohup bash -c "bash ~/deploy.sh; echo DEPLOY_EXIT=\$?" > ~/deploy.log 2>&1 < /dev/null &'
ssh_vm 'tail -n +1 -F ~/deploy.log 2>/dev/null | sed -u "/^DEPLOY_EXIT=/q"'
ssh_vm 'grep -q "^DEPLOY_EXIT=0" ~/deploy.log' || { echo "Deploy failed: see ~/deploy.log on $VM" >&2; exit 1; }

echo
if [[ -n "$DOMAIN" ]]; then
  echo "Deployed to $VM: https://$DOMAIN"
else
  echo "Deployed to $VM. Open it with:"
  echo "  gcloud compute ssh $VM --zone $ZONE -- -N -L 3000:localhost:3000"
  echo "  open http://localhost:3000"
fi
