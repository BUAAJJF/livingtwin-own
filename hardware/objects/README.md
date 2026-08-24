# The physical object set

Eight objects, fixed. The sizes are not a wish list — each one is placed at a
named percentile of the distribution the policy was actually trained on, drawn
by sampling `piper_push.shapes._compose` 400 000 times at `variety=1.0`.

Buy to these sizes. Weigh what you find. Mass is the one thing not fixed here,
and the reason is in **What mass is not decided yet** below.

## The set

| # | shape class | long × short × height | aspect | W %ile | H %ile | aspect %ile |
|---|---|---|---|---|---|---|
| 1 | box | 28 × 28 × 30 | 1.07 | 7 | 20 | 32 |
| 2 | box | 40 × 40 × 40 | 1.00 | 62 | 38 | 28 |
| 3 | box | 45 × 45 × 25 | 0.56 | 85 | 11 | **0** |
| 4 | cylinder | Ø34 × 62 | 1.82 | 34 | 76 | 78 |
| 5 | cylinder | Ø32 × 66 | 2.06 | 25 | 81 | **93** |
| 6 | stepped | 42 × 42 × 50 | 1.19 | 71 | 56 | 39 |
| 7 | L | 40 × 40 × 40 | 1.00 | 62 | 38 | 28 |
| 8 | capped | Ø38 × 60 | 1.58 | 53 | 73 | 63 |

Three and five are the corners on purpose. **#3** sits at the height floor:
24 mm is where the fingertip reaches the table before the pad reaches the
object, so a flatter object is not a harder task, it is one this gripper
cannot do. **#5** is the tallest aspect the sampler draws.

The class shares come out at box 3/8, cylinder 2/8, stepped 1/8, L 1/8, capped
1/8, against the trained 34 / 22 / 16 / 16 / 12 %.

## What the composite shapes mean

The sim builds these out of a box, a cylinder and a second box, so the
proportions are fixed by `shapes._compose` and not free:

* **#6 stepped** — base `42 × 42 × 30`, block on top `23 × 23 × 20`.
  A nut on a bolt, a turned step, a squat jar with a smaller lid.
* **#7 L** — an upright arm `20 × 40 × 40` with a foot `20 × 40 × 18` at the
  bottom, meeting at the body origin. A steel angle bracket standing up.
* **#8 capped** — base `Ø38 × 33` with a cylinder `Ø28 × 27` on top.
  A small jar or bottle with its cap on.

## Hard limits — a miss here is not a data point, it is a broken run

| | limit | why |
|---|---|---|
| widest horizontal extent | **≤ 50 mm** | ≥ 55 mm rolls out of the 29.6 mm pads |
| height | **≥ 24 mm** | the fingertip hangs 12.3 mm below the grasp site |
| height | **≤ 90 mm** | a 98 mm object on a 25 mm base tips before the pads close |
| plan view | **long/short ≤ 1.4** | the sampler's anisotropy p90; it draws blocks, not bars |
| mass | **≤ 400 g** | the top of the trained range; the gripper itself is good to 600 |

Tolerance on the table above: **±3 mm** on each dimension, ±4 mm on the
composites. The camera pose randomisation is 20 mm, so a few millimetres of
object is not what will decide anything.

## Surfaces — pick these deliberately

The bench measured what this camera does to a surface, and it is the largest
effect in the whole vision stage: a printed pattern fills 99.6% of its pixels
and a blank white patch fills 88% on average and **41.9%** in the worst shot,
at twice the noise. So the set has to contain that case rather than avoid it.

* at least one **matte white, untextured** — the measured worst case
* at least one **matt black** — 98.9% fill but 1.7× the noise
* at least one **specular** (bare machined metal, glossy paint) — the failure
  mode the bench did not measure and the one no model here covers
* the rest **textured** — labels, printing, grain, machining marks

The cheapest controlled experiment in the set: buy **#2 twice** and spray one
matte white. Same shape, same mass, same everything, one variable.

## Friction

The trained range is 0.4 to 1.0 against the table. Polished metal on a smooth
table can sit below that, and an object outside the range will be dropped in a
way that looks exactly like a policy failure. If an object slides too freely,
put a strip of tape on its base rather than throwing the object away — and
write down that you did.

## What mass is not decided yet

Because the current distribution cannot be met by real objects and the fix
depends on what you find.

Mass is drawn uniformly from 50–400 g **independently of size**, so the implied
bounding-box density comes out at a median of **3.8 g/cm³**, denser than
aluminium; 19% of draws are denser than steel and **1% are denser than gold**,
peaking at 26.5 g/cm³. No material does that. Meanwhile the objects you will
actually find at 28–45 mm are wood, plastic and glass at 0.5–1.4 g/cm³, which
is below the 5th percentile of what the policy has seen.

So: weigh everything, put the numbers in `measured.json`, and run

```bash
micromamba run -n mjlab python scripts/fit_object_distribution.py
```

It places each object in the trained distribution and prints what to change.
There are two candidate changes and the numbers decide between them:

1. **widen `OBJECT_MASS_RANGE`** downwards to cover what you have — one line,
   keeps everything else;
2. **sample density instead of mass** and compute mass from the composed
   volume — a coherent joint distribution instead of one containing objects
   denser than gold, and the right fix if the set spans a wide size range.

Either way it is a retrain, and either way it is one change. Do not guess the
masses now; the whole point of buying the objects is that you no longer have
to.
