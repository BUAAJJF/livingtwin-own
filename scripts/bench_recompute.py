"""What does re-deriving the model constants every control step cost?

Redrawing an object's shape as it is replaced means writing per-world model
fields mid-episode, and a write to those is only visible after
``recompute_constants``.  At 8192 environments a placement happens somewhere on
almost every step, so the question is not whether the call is cheap but whether
paying it every step is affordable.
"""
import argparse, time
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.event_manager import RecomputeLevel
from mjlab.tasks.registry import load_env_cfg

p = argparse.ArgumentParser()
p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Vision")
p.add_argument("--num-envs", type=int, default=1024)
p.add_argument("--steps", type=int, default=120)
p.add_argument("--warmup", type=int, default=30)
p.add_argument("--device", default="cuda:0")
a = p.parse_args()

cfg = load_env_cfg(a.task)
cfg.scene.num_envs = a.num_envs
env = ManagerBasedRlEnv(cfg=cfg, device=a.device, render_mode=None)
env.reset()
act = torch.zeros(a.num_envs, env.action_space.shape[-1], device=a.device)


def run(with_recompute: bool) -> float:
    with torch.inference_mode():
        for _ in range(a.warmup):
            env.step(act)
            if with_recompute:
                env.sim.recompute_constants(RecomputeLevel.set_const)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(a.steps):
            env.step(act)
            if with_recompute:
                env.sim.recompute_constants(RecomputeLevel.set_const)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / a.steps


plain = run(False)
heavy = run(True)
print(f"  {a.num_envs} envs")
print(f"  plain                    {1000*plain:7.2f} ms/step")
print(f"  + recompute every step   {1000*heavy:7.2f} ms/step")
print(f"  overhead                 {100*(heavy-plain)/plain:7.1f}%")
