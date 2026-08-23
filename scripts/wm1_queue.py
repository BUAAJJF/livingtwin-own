"""Run an adaptation plan on whichever GPUs are free, as they become free.

    python scripts/wm1_queue.py --plan results/wm1_latency/adapt/formal.json

`wm1_adapt.sh` shards by modulo and pins shard *k* to GPU *k*, which is right
when the box is yours. It is wrong when it is not: another project took five of
the eight cards between one wave of this phase and the next, and five pinned
shards then sat waiting for cards that were busy for hours while three cards
went idle with eight jobs still queued.

So: no pinning. A job goes to whatever card has room, one job per card, and a
card that frees up gets the next pending job. Each job is handed to the same
`wm1_adapt.sh` as a one-job plan, so the skip-if-done, lock, checkpoint
assertion and evaluation logic are the ones already tested rather than a second
copy of them.

Restartable. A job counts as done when its checkpoint pointer and all six
evaluations exist, so killing this and starting it again picks up where it
stopped.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ADAPT = Path("results/wm1_latency/adapt")


def gpu_free_mib() -> dict[int, int]:
  try:
    out = subprocess.run(
      ["nvidia-smi", "--query-gpu=index,memory.free",
       "--format=csv,noheader,nounits"],
      capture_output=True, text=True, timeout=30).stdout
  except Exception:
    return {}
  free = {}
  for line in out.strip().splitlines():
    try:
      i, m = (x.strip() for x in line.split(","))
      free[int(i)] = int(m)
    except ValueError:
      continue
  return free


def busy_elsewhere(tags) -> set[str]:
  """Tags some other process is already working on.

  Not only training.  Three trainings were orphaned when their pinned shards
  were stopped -- the shell was killed, the training was not -- and a queue
  that checks only for a finished checkpoint starts a second copy of one on
  another card.  It did, once.  Restarting this queue creates the same hazard
  a second way: its previous children keep running, and between finishing a
  training and finishing its evaluations there is no `finetune.py` to find.

  So the test is whether the tag appears anywhere in the command line of a
  running training or evaluation, which covers both -- `--run-name wm1_<tag>`
  and `--label <tag>_target__r0`.
  """
  try:
    out = subprocess.run(["ps", "-eo", "cmd"], capture_output=True,
                         text=True, timeout=30).stdout
  except Exception:
    return set()
  lines = [ln for ln in out.splitlines()
           if "scripts/finetune.py" in ln or "scripts/accept_s1.py" in ln
           or "scripts/wm1_adapt.sh" in ln]
  return {t for t in tags if any(t in ln for ln in lines)}


def is_done(tag: str) -> bool:
  if not (ADAPT / f"{tag}.ckpt").exists():
    return False
  return all((ADAPT / f"{tag}_{kind}__r{r}.json").exists()
             for kind in ("target", "retention") for r in range(3))


def main() -> int:
  p = argparse.ArgumentParser()
  p.add_argument("--plan", required=True)
  p.add_argument("--min-free-mib", type=int, default=70000)
  p.add_argument("--poll", type=int, default=60)
  p.add_argument("--max-concurrent", type=int, default=8)
  p.add_argument("--gpus", default=None,
                 help="comma-separated allow-list; default every visible card")
  a = p.parse_args()

  plan = json.loads(Path(a.plan).read_text())
  jobs = plan["jobs"]
  allowed = ([int(x) for x in a.gpus.split(",")] if a.gpus
             else sorted(gpu_free_mib()))
  print(f"  {len(jobs)} jobs, cards {allowed}, "
        f"need {a.min_free_mib} MiB free", flush=True)

  running: dict[int, tuple] = {}      # gpu -> (Popen, tag, started)
  t0 = time.time()
  while True:
    for gpu in list(running):
      proc, tag, started = running[gpu]
      if proc.poll() is None:
        continue
      mark = "ok" if proc.returncode == 0 and is_done(tag) else "FAILED"
      print(f"  [{(time.time() - t0) / 60:6.1f} min] cuda:{gpu} {tag} {mark} "
            f"(rc={proc.returncode}, {(time.time() - started) / 60:.1f} min)",
            flush=True)
      del running[gpu]

    mine = {t for _, t, _ in running.values()}
    busy = mine | busy_elsewhere({j["tag"] for j in jobs} - mine)
    pending = [j for j in jobs if not is_done(j["tag"])
               and j["tag"] not in busy]
    if not pending and not running:
      print(f"  all {len(jobs)} jobs done in "
            f"{(time.time() - t0) / 60:.1f} min", flush=True)
      return 0

    if len(running) < a.max_concurrent:
      free = gpu_free_mib()
      for job in pending:
        cand = [g for g in allowed
                if g not in running and free.get(g, 0) >= a.min_free_mib]
        if not cand:
          break
        gpu = max(cand, key=lambda g: free[g])
        one = ADAPT / f"queue_{job['tag']}.json"
        one.write_text(json.dumps({**plan, "jobs": [job], "n_runs": 1}, indent=1))
        proc = subprocess.Popen(
          ["bash", "scripts/wm1_adapt.sh", str(gpu), "0", "1", str(one)],
          stdout=open(f"logs/wm1_adapt/queue_{job['tag']}.log", "w"),
          stderr=subprocess.STDOUT)
        running[gpu] = (proc, job["tag"], time.time())
        print(f"  [{(time.time() - t0) / 60:6.1f} min] cuda:{gpu} <- "
              f"{job['tag']} ({'+'.join(job['methods'])}), "
              f"{free[gpu]} MiB free", flush=True)
        # One job per card, and the freshly launched one has not allocated
        # yet, so do not hand out this card again on this pass.
        free[gpu] = 0
        if len(running) >= a.max_concurrent:
          break

    if not running:
      print(f"  [{(time.time() - t0) / 60:6.1f} min] {len(pending)} pending, "
            f"no card with {a.min_free_mib} MiB free", flush=True)
    time.sleep(a.poll)


if __name__ == "__main__":
  raise SystemExit(main())
