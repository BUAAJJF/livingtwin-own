#!/usr/bin/env bash
# Put the current branch on the server without touching its working tree.
#
#   scripts/sync_server.sh
#
# A plain push is rejected: the server's receive.denyCurrentBranch is
# updateInstead, which refuses whenever the incoming commits would overwrite an
# untracked file, and the server's untracked files are the raw result JSONs
# that later got committed from this side.  So the commits go to a side branch,
# the checked-out ref is moved by hand, and the index is reset -- none of which
# writes over anything in the working tree -- and only the source trees are
# checked out.
set -Eeuo pipefail
REMOTE=shen-teacher:/home/yunfan/work/piper-push/LivingTwin
BRANCH=$(git rev-parse --abbrev-ref HEAD)
GIT_SSH_COMMAND='ssh -o ClearAllForwardings=yes' \
  git push -q "$REMOTE" "+$BRANCH:refs/heads/wm1-sync" --tags
ssh shen-teacher "cd /home/yunfan/work/piper-push/LivingTwin \
  && git update-ref refs/heads/$BRANCH refs/heads/wm1-sync \
  && git reset -q \
  && git checkout -- .gitignore docs src tests scripts pyproject.toml README.md \
  && git log --oneline -1"
