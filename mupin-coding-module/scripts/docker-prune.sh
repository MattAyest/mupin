#!/usr/bin/env bash
# Docker maintenance script for the Coding Module.
#
# Safe cleanup of build-up artifacts without touching running services or task
# workspaces. Run manually when build cache / exited containers grow.
#
# Usage:
#   ./scripts/docker-prune.sh
#   ./scripts/docker-prune.sh --dry-run    # show what would be removed

set -euo pipefail

DRY_RUN=${DRY_RUN:-}
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

log() {
  echo "[docker-prune] $*"
}

run() {
  if [ -n "$DRY_RUN" ]; then
    echo "  (dry-run) $*"
  else
    "$@"
  fi
}

# 1. Remove exited containers, EXCEPT persistent infra.
#    We keep containers whose names match core Mupin services.
log "Removing exited containers..."
EXCLUDE_NAMES="mupin_|opencode-|red-music-bot|cloud-ide|coding_workers|lavalink|n8n-|cloudflared|sandbox_memory|sandbox_proxy"
mapfile -t OLD_EXITED < <(docker ps -a --filter "status=exited" --format '{{.Names}}' | grep -vE "^($EXCLUDE_NAMES)" || true)
if [ ${#OLD_EXITED[@]} -gt 0 ]; then
  run docker rm -f "${OLD_EXITED[@]}"
else
  log "  none found"
fi

# 2. Remove dangling build cache (the main disk hog).
log "Removing dangling build cache..."
run docker builder prune -f --filter type=regular

# 3. Remove dangling images and unused images older than 24 hours.
log "Removing dangling / unused images..."
run docker image prune -f --filter "until=24h"

# 4. Remove unused networks (usually tiny, but tidy).
log "Pruning unused networks..."
run docker network prune -f

# 5. Report remaining disk usage.
log "Current docker disk usage:"
docker system df
