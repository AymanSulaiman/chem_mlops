#!/usr/bin/env bash
# Create (first run) and deploy the chat app to a GCE e2-micro VM, from your
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

cd "$(dirname "$0")/.."
# Keep-alives notice a dropped connection (laptop asleep, network blip) instead of hanging.
ssh_vm() { gcloud compute ssh "$VM" --zone "$ZONE" --quiet -- -o ServerAliveInterval=30 -o ServerAliveCountMax=4 "$@"; }
SETTINGS="deploy/$VM.env"
DOMAIN="$( [[ -f "$SETTINGS" ]] && sed -n 's/^DOMAIN=//p' "$SETTINGS" || true)"

if ! gcloud compute instances describe "$VM" --zone "$ZONE" &>/dev/null; then
  echo "==> Creating $VM (e2-micro, ${DISK_GB} GB disk, $ZONE)"
  gcloud compute instances create "$VM" --zone "$ZONE" \
    --machine-type e2-micro \
    --image-family debian-12 --image-project debian-cloud \
    --boot-disk-size "${DISK_GB}GB" --boot-disk-type pd-standard \
    --scopes storage-ro \
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
elif [[ "$(gcloud compute instances describe "$VM" --zone "$ZONE" --format='value(status)')" != RUNNING ]]; then
  echo "==> Starting $VM"
  gcloud compute instances start "$VM" --zone "$ZONE"
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

# The sync and the image builds run detached on the VM (they take a while on
# an e2-micro), so a dropped connection doesn't kill them. This script follows
# ~/deploy.log; if it gets cut off, check on it with:
#   gcloud compute ssh chem-demo --zone us-central1-a -- tail -f deploy.log
echo "==> Syncing the model ($RUN) and LanceDB from gs://$BUCKET, then building and starting the app"
ssh_vm "cat > ~/deploy.sh <<EOF
set -e
mkdir -p ~/model ~/data/lancedb
gcloud storage rsync --recursive --delete-unmatched-destination-objects gs://$BUCKET/models/$RUN ~/model
gcloud storage rsync --recursive --delete-unmatched-destination-objects gs://$BUCKET/data/lancedb ~/data/lancedb
cd ~/chem_mlops
sudo env PWD=\\\$PWD MODEL_DIR=\\\$HOME/model LANCEDB_HOST_DIR=\\\$HOME/data/lancedb \\
  docker compose $PROFILE up -d --build --wait
sudo docker ps --format "{{.Names}}  {{.Status}}"
EOF
rm -f ~/deploy.log
setsid nohup bash -c 'bash ~/deploy.sh; echo DEPLOY_EXIT=\$?' > ~/deploy.log 2>&1 < /dev/null &"
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
