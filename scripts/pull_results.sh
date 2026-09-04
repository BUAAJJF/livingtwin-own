#!/usr/bin/env bash
# Bring the tracked result files back from the training server.
#
#   scripts/pull_results.sh [subdir]
#
# Excludes what .gitignore excludes -- the raw per-run sweep JSONs and the
# multi-gigabyte session tensors -- so that what lands here is exactly what
# gets committed.
set -Eeuo pipefail
SUB=${1:-d455_heavy_dr}
cd "$(dirname "$0")/.."
# --delete, because the server is the source of truth: a result quarantined
# there must not survive here and quietly re-enter an analysis.  runs/ is
# generated on this side and is excluded from the deletion.
rsync -az --info=stats1 --delete --filter 'protect runs/***' \
  --exclude '*.pt' --exclude 's1/' --exclude 's2/' --exclude 's3/' \
  -e "ssh -o ClearAllForwardings=yes" \
  "shen-teacher:/home/yunfan/work/piper-push/LivingTwin/results/$SUB/" \
  "results/$SUB/"
