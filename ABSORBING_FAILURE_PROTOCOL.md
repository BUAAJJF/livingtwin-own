# Frozen protocol: Stage 1 absorbing-failure diagnostic

This is an independent diagnostic of the accepted-but-failed position+yaw Stage 1.
It never overwrites seeds 2201--2203. Only the out-of-bounds termination semantics
change; observation dimensionality, action, ordinary reward weights, PPO settings,
reset distributions, success definition, and the 200-step horizon remain frozen.

## Original-return audit

The audit uses deterministic final policies and common evaluation states from the
frozen failed Stage 1. Representative trajectories are the median gamma=0.99
discounted-return members of three categories across those evaluations: sustained
success, valid timeout with measurable progress, and out-of-bounds failure.

## Frozen failure reward

Within the legal cube and target bounds, the largest possible position error is
`sqrt(0.32^2 + 0.23^2) = 0.394081... m`. The most negative legal ordinary reward is

```text
-2.0 * 0.394081... - 0.15 * pi - 0.001 * (1^2 + 1^2)
= -1.261401...
```

The absorbing failure reward is frozen before training at `-1.262` per control
step. It is slightly more negative than that legal-state lower bound. Because all
gamma weights are positive, replacing every remaining step by -1.262 makes leaving
the legal state weakly worse than any legal continuation with the same prefix.

## Absorbing state

On the first out-of-bounds transition:

- mark the episode as failed and store the absorption step;
- replace that transition reward and every remaining reward with `-1.262`;
- do not set done before step 200;
- freeze MuJoCo state and ignore subsequent actions;
- return the fixed 15-dimensional all-zero raw observation. This is deliberately
  out of manifold because valid sine/cosine pairs cannot both be zero;
- at step 200 return `truncated=True`, reset normally, and use zero bootstrap.

Thus GAE uses a nonterminal mask through absorbing steps, propagates their penalties,
and uses a terminal mask only at the fixed horizon. Evaluation follows exactly the
same semantics.

## Execution boundary

After unit tests, return audit, absorbing-state checks, yaw-bin reachability audit,
LR=0, and checkpoint round-trip pass, run exactly one 64-environment, 65,536-step
smoke from scratch with seed 1201. Do not run formal seeds or any later stage.
