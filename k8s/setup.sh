#!/usr/bin/env bash
# Create a 3-worker kind cluster, install Kueue and enable cohort fair sharing.
# Requires: docker, kind, kubectl. The host should have >= cohort CPU quota free cores.
set -euo pipefail
KUEUE_VERSION="${KUEUE_VERSION:-v0.10.1}"     # pin the release you cite in the paper
cd "$(dirname "$0")"
kind get clusters | grep -q '^fairattack$' || kind create cluster --name fairattack --config kind-config.yaml
kubectl config use-context kind-fairattack
kubectl apply --server-side -f "https://github.com/kubernetes-sigs/kueue/releases/download/${KUEUE_VERSION}/manifests.yaml"
kubectl -n kueue-system wait deploy/kueue-controller-manager --for=condition=Available --timeout=180s

# Enable fair sharing in the controller config (the ConfigMap holds controller_manager_config.yaml)
kubectl -n kueue-system get cm kueue-manager-config -o jsonpath='{.data.controller_manager_config\.yaml}' > /tmp/kueue-cfg.yaml
if ! grep -q '^fairSharing:' /tmp/kueue-cfg.yaml; then
  cat >> /tmp/kueue-cfg.yaml <<'YAML'
fairSharing:
  enable: true
  preemptionStrategies: [LessThanOrEqualToFinalShare, LessThanInitialShare]
YAML
  kubectl -n kueue-system create cm kueue-manager-config \
     --from-file=controller_manager_config.yaml=/tmp/kueue-cfg.yaml --dry-run=client -o yaml | kubectl apply -f -
  kubectl -n kueue-system rollout restart deploy/kueue-controller-manager
  kubectl -n kueue-system rollout status deploy/kueue-controller-manager --timeout=180s
fi
# Pre-pull the payload image into every node so image pulls do not distort timings.
docker pull python:3.12-slim
kind load docker-image python:3.12-slim --name fairattack
echo "ready. Verify: kubectl -n kueue-system get cm kueue-manager-config -o yaml | grep -A3 fairSharing"
