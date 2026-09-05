"""Export the recurrent vision policy and check the export still acts the same.

Exporting is only useful if the exported thing agrees with the thing that was
trained, and a recurrent policy can disagree in a way a single forward pass
would never show: the state has to advance identically too.  So this steps both
for a few frames, feeding ONNX its own previous hidden state, and compares.
"""
import argparse
from dataclasses import asdict

import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--task", default="Mjlab-Pick-Place-PiperX-Vision")
p.add_argument("--device", default="cuda:0")
p.add_argument("--steps", type=int, default=8)
p.add_argument("--out", default="/tmp/vision_policy")
from piper_push import evalcfg as _evalcfg  # noqa: E402
_evalcfg.add_action_api_arg(p)
a = p.parse_args()

_evalcfg.apply_action_api_arg(a)
env_cfg = load_env_cfg(a.task, play=True)
agent_cfg = load_rl_cfg(a.task)
env_cfg.scene.num_envs = 1
env = ManagerBasedRlEnv(cfg=env_cfg, device=a.device, render_mode=None)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = (load_runner_cls(a.task) or MjlabOnPolicyRunner)(env, asdict(agent_cfg), None, a.device)
runner.load(a.checkpoint, load_cfg={"actor": True}, strict=True, map_location=a.device)

runner.export_policy_to_jit(a.out, "policy.pt")
runner.export_policy_to_onnx(a.out, "policy.onnx")
print("exported jit + onnx")

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(f"{a.out}/policy.onnx", providers=["CPUExecutionProvider"])
jit = torch.jit.load(f"{a.out}/policy.pt")
jit.reset()

policy = runner.alg.get_policy()
policy.reset()
names = [i.name for i in sess.get_inputs()]
print("onnx inputs :", names)
print("onnx outputs:", [o.name for o in sess.get_outputs()])

h = np.zeros((policy.rnn.rnn.num_layers, 1, policy.rnn.rnn.hidden_size), dtype=np.float32)
obs = env.get_observations()
if isinstance(obs, tuple):
    obs = obs[0]
worst_onnx = worst_jit = 0.0
with torch.inference_mode():
    for i in range(a.steps):
        ref = policy(obs.to(a.device))

        obs_1d = torch.cat([obs[g] for g in policy.obs_groups], dim=-1).cpu()
        images = [obs[g].cpu() for g in policy.obs_groups_2d]

        feed = {names[0]: obs_1d.numpy()}
        for n, img in zip(names[1:-1], images):
            feed[n] = img.numpy()
        feed[names[-1]] = h
        act_onnx, h = sess.run(None, feed)

        act_jit = jit(obs_1d, images)

        worst_onnx = max(worst_onnx, float((ref.cpu() - torch.from_numpy(act_onnx)).abs().max()))
        worst_jit = max(worst_jit, float((ref.cpu() - act_jit).abs().max()))
        obs = env.step(ref.to(env.device))[0]

print(f"max |torch - onnx| over {a.steps} steps: {worst_onnx:.3e}")
print(f"max |torch - jit | over {a.steps} steps: {worst_jit:.3e}")
print("OK" if max(worst_onnx, worst_jit) < 1e-4 else "MISMATCH")
