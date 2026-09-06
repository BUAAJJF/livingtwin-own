"""Assemble a deployment bundle from a finished route directory.

    python scripts/pc/bundle.py results/pc/routes/pc_final_P1B_<utc> --spec results/pc/specs/obs_spec_P1B.json \\
        --out hardware/deploy/policies/pc_P1B_<utc>

Copies the final checkpoint, the exported graphs, the observation spec, the
D455 calibration snapshot and every hash into one directory with a
manifest.json, so a shadow or real run can name a single path and everything
it ran with is recorded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess
import time


def sha(path):
  h = hashlib.sha256()
  with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
      h.update(chunk)
  return h.hexdigest()


def main():
  p = argparse.ArgumentParser()
  p.add_argument("route_dir")
  p.add_argument("--spec", required=True)
  p.add_argument("--out", required=True)
  p.add_argument("--checkpoint", default=None, help="override: the local copy of the final checkpoint")
  p.add_argument("--rig", default="hardware/deploy/rig_d455.json")
  a = p.parse_args()
  rd = pathlib.Path(a.route_dir)
  out = pathlib.Path(a.out)
  man = json.loads((rd / "manifest.json").read_text())
  from piper_push.pc import routes as pc_routes
  try:
    pc_routes.check_deployable(str(man.get("route")))
  except ValueError as e:
    raise SystemExit(f"refusing to bundle {rd}: {e}")
  if man.get("oracle_only"):
    raise SystemExit(f"refusing to bundle {rd}: manifest says oracle_only")
  if out.exists():
    raise SystemExit(f"{out} exists")
  out.mkdir(parents=True)
  export = rd / "export"
  files = {}
  for name in ("policy.onnx", "policy.pt"):
    src = export / name
    if not src.exists():
      raise SystemExit(f"{src} missing: the route's export stage did not produce it")
    shutil.copy2(src, out / name)
    files[name] = sha(out / name)
  ck = pathlib.Path(a.checkpoint) if a.checkpoint else None
  if ck is not None and ck.exists():
    shutil.copy2(ck, out / "checkpoint.pt")
    files["checkpoint.pt"] = sha(out / "checkpoint.pt")
  shutil.copy2(a.spec, out / "obs_spec.json")
  files["obs_spec.json"] = sha(out / "obs_spec.json")
  shutil.copy2(a.rig, out / "rig_d455.json")
  files["rig_d455.json"] = sha(out / "rig_d455.json")
  spec = json.loads((out / "obs_spec.json").read_text())
  commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
  bundle = {
    "route": man.get("route"), "built": time.strftime("%Y-%m-%dT%H:%M:%S"), "route_dir": str(rd),
    "teacher": man.get("teacher"), "teacher_sha256": man.get("teacher_sha256"),
    "final_checkpoint_remote": (rd / "final_checkpoint.txt").read_text().strip() if (rd / "final_checkpoint.txt").exists() else None,
    "code_commit_training": man.get("code_commit_local"), "code_commit_bundle": commit,
    "action_api": spec.get("action_api"), "action_spec_hash": None,
    "actor_groups": spec.get("actor_groups"), "files_sha256": files,
    "rollback": "hardware/deploy/policies/d455_v4_final (Action API v1; needs -V1 ids and --allow-legacy-action-api; NOT this pipeline)",
  }
  from piper_push import action_api
  if spec.get("action_spec"):
    bundle["action_spec_hash"] = action_api.spec_hash(spec["action_spec"])
  (out / "manifest.json").write_text(json.dumps(bundle, indent=2) + "\n")
  print(json.dumps(bundle, indent=2))


if __name__ == "__main__":
  main()
