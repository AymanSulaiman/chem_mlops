#!/usr/bin/env bash
# One-time setup so GitHub Actions can run deploy/vm.sh (the "release" job in
# .github/workflows/ci.yml). Keyless: GitHub's OIDC token is exchanged for a
# short-lived token of a deployer service account, and only runs from this
# repo's main branch may do that. Re-running is safe.
#
#   ./deploy/setup_ci.sh
#
# Needs gcloud (as a project owner) and gh (logged in to the repo).
set -euo pipefail

PROJECT="${PROJECT:-chem-mlops}"
REPO="${REPO:-AymanSulaiman/chem_mlops}"
VM="${VM:-chem-demo}"
ZONE="${ZONE:-us-central1-a}"
POOL=github
PROVIDER=github
SA_NAME=github-deployer
SA="$SA_NAME@$PROJECT.iam.gserviceaccount.com"

cd "$(dirname "$0")/.."
PROJECT_NUMBER="$(gcloud projects describe "$PROJECT" --format='value(projectNumber)')"
VM_SA="$(gcloud compute instances describe "$VM" --zone "$ZONE" --project "$PROJECT" --format='value(serviceAccounts[0].email)')"

gcloud services enable iamcredentials.googleapis.com sts.googleapis.com --project "$PROJECT"

gcloud iam service-accounts describe "$SA" --project "$PROJECT" &>/dev/null \
  || gcloud iam service-accounts create "$SA_NAME" --project "$PROJECT" --display-name "GitHub Actions deployer"

# vm.sh starts the VM, tags it, reads the firewall rule, and SSHes in with a
# key it adds to project metadata: all in instanceAdmin.v1. SSH to a VM that
# runs as a service account also needs serviceAccountUser on that account.
gcloud projects add-iam-policy-binding "$PROJECT" --condition=None --quiet \
  --member "serviceAccount:$SA" --role roles/compute.instanceAdmin.v1 >/dev/null
gcloud iam service-accounts add-iam-policy-binding "$VM_SA" --project "$PROJECT" --quiet \
  --member "serviceAccount:$SA" --role roles/iam.serviceAccountUser >/dev/null

gcloud iam workload-identity-pools describe "$POOL" --location global --project "$PROJECT" &>/dev/null \
  || gcloud iam workload-identity-pools create "$POOL" --location global --project "$PROJECT" \
       --display-name "GitHub Actions"
gcloud iam workload-identity-pools providers describe "$PROVIDER" --workload-identity-pool "$POOL" \
    --location global --project "$PROJECT" &>/dev/null \
  || gcloud iam workload-identity-pools providers create-oidc "$PROVIDER" \
       --workload-identity-pool "$POOL" --location global --project "$PROJECT" \
       --issuer-uri https://token.actions.githubusercontent.com \
       --attribute-mapping 'google.subject=assertion.sub,attribute.repository=assertion.repository,attribute.ref=assertion.ref' \
       --attribute-condition "assertion.repository == '$REPO' && assertion.ref == 'refs/heads/main'"

gcloud iam service-accounts add-iam-policy-binding "$SA" --project "$PROJECT" --quiet \
  --role roles/iam.workloadIdentityUser \
  --member "principalSet://iam.googleapis.com/projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/attribute.repository/$REPO" >/dev/null

# Nothing here is a secret (the client ID ships in the page), so repo variables.
gh variable set GCP_WIF_PROVIDER --repo "$REPO" \
  --body "projects/$PROJECT_NUMBER/locations/global/workloadIdentityPools/$POOL/providers/$PROVIDER"
gh variable set GCP_DEPLOY_SA --repo "$REPO" --body "$SA"
gh variable set GCP_PROJECT --repo "$REPO" --body "$PROJECT"
gh variable set DEMO_DOMAIN --repo "$REPO" --body "$(sed -n 's/^DOMAIN=//p' "deploy/$VM.env")"
gh variable set GOOGLE_CLIENT_ID --repo "$REPO" --body "$(sed -n 's/^GOOGLE_CLIENT_ID=//p' "deploy/$VM.env")"

echo "Done. Pushes to main now deploy $VM after CI passes."
