#!/usr/bin/env bash
# Smoke-test a built BankShield image the way it should be run in production:
# read-only root filesystem, no capabilities, state on a named volume, the
# JWT signing key supplied at run time. Then restart the container and check
# that users, cases and the audit trail survived.
#
# Usage: scripts/docker_smoke.sh [IMAGE]          (default bankshield-ai:ci)
set -euo pipefail

IMAGE="${1:-bankshield-ai:ci}"
NAME="bankshield-smoke"
VOLUME="bankshield-smoke-data"
PORT="${SMOKE_PORT:-8000}"
cd "$(dirname "$0")/.."

export BANKSHIELD_SMOKE_PASSWORD="smoke-$(head -c 18 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')"
JWT_SECRET="$(head -c 48 /dev/urandom | base64 | tr -dc 'A-Za-z0-9')"

cleanup() {
  docker logs "$NAME" > "${SMOKE_LOG:-/dev/null}" 2>&1 || true
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker volume rm "$VOLUME" >/dev/null 2>&1 || true
}
trap cleanup EXIT
cleanup

wait_healthy() {
  for _ in $(seq 1 45); do
    status=$(docker inspect -f '{{.State.Health.Status}}' "$NAME")
    [ "$status" = healthy ] && { echo "container healthy"; return 0; }
    [ "$status" = unhealthy ] && break
    sleep 2
  done
  echo "container did not become healthy"; docker logs "$NAME" | tail -50; return 1
}

docker volume create "$VOLUME" >/dev/null
docker run -d --name "$NAME" -p "$PORT:8000" \
  --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  -v "$VOLUME:/app/var" -e BANKSHIELD_JWT_SECRET="$JWT_SECRET" \
  "$IMAGE" >/dev/null
wait_healthy

# Accounts can only be provisioned with operator access to the container.
for role in analyst reviewer agent admin; do
  echo "$BANKSHIELD_SMOKE_PASSWORD" | docker exec -i "$NAME" \
    python -m bankshield.auth.cli create-user "smoke.$role" --role "$role" --password-stdin
done

python3 scripts/smoke_test_api.py "http://127.0.0.1:$PORT"

echo; echo "--- restarting the container (same volume, same signing key) ---"
docker restart "$NAME" >/dev/null
wait_healthy
python3 scripts/smoke_test_api.py "http://127.0.0.1:$PORT" --verify-persistence
