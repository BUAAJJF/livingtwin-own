"""A streaming SAM2.1 video predictor, shaped like ``sam_tracker.Predictor``.

``sam_tracker.SamTargetTracker`` holds the policy -- when to trust a mask, when
to withhold it -- and takes its pixels from an injected ``Predictor`` so that
policy is testable without a GPU.  This is the real one.

It exists because SAM2's video API cannot be driven live as shipped.
``init_state`` takes a *directory of JPEGs*, decodes all of them up front, and
``propagate_in_video`` is a generator over a video that already exists.  The
TwinSight audit worked around that by writing every frame to disk as a JPEG and
re-initialising per sequence, and paid for it: core inference measured
11.2 ms p50 while end-to-end came out at 53.4 ms/frame.  The handoff is
explicit that the JPEG round-trip and the per-sequence re-initialisation are
audit conveniences that "should not be an acceptable live path".

So this builds the inference state itself and grows it one frame at a time:

* ``images`` is a **list** the class appends to, not a preallocated tensor.
  ``_get_image_feature`` only ever indexes ``images[frame_idx]``, so a list
  works and nothing has to know the length of a video that does not exist yet.
* preprocessing is a GPU resize of the frame already in memory -- no encode, no
  decode, no filesystem.
* one frame is advanced per call via ``max_frame_num_to_track=0``, which makes
  ``propagate_in_video``'s ``processing_order`` exactly one index long.
* frames older than the model's own memory window are dropped, so a ten-minute
  episode costs the same as a ten-second one.

What it does **not** do is decide anything.  It returns the mask SAM produced
and nothing else; every rejection lives in ``sam_tracker``.  Keeping those
apart is what lets the 13% wrong-target rate be handled by code with tests.

Compiled ``vos_optimized`` was tested and rejected on the deployment GPU:
the installed torch 2.13/SAM2.1 stack fails on its first propagated frame.
``run.py`` refuses it before opening hardware; the constructor flag remains
only for isolated compatibility tests after that stack changes.  Eager BF16 is
the measured live path.
"""

from __future__ import annotations

import dataclasses
import pathlib
import time
from collections import OrderedDict

import numpy as np

CHECKPOINT = pathlib.Path(__file__).parent / "sam2_assets" / "sam2.1_hiera_small.pt"
MODEL_CFG = "configs/sam2.1/sam2.1_hiera_s.yaml"

IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)
"""SAM2's own normalisation, copied from ``sam2.utils.misc.load_video_frames``.
Not a choice -- the backbone was trained with it."""

MAX_ANCHORS = 3
"""How many past anchors stay in the conditioning set.

SAM2 ships ``max_cond_frames_in_attn = -1``: memory attention cross-attends to
**every** conditioning frame there has ever been.  Off a fixed-length video
that is a constant; live, with the depth stack re-anchoring whenever it and SAM
agree, it is a leak.  Measured over 1200 frames with 37 re-anchors, inference
p50 went 26.5 ms to 41.4 ms and p99 to 69 ms -- entirely from attending to
anchors the tracker had long since superseded.  The most recent few are what
the target currently looks like; the rest are history."""

MEMORY_FRAMES = 48
"""How many past frames to keep.

The model looks back ``num_maskmem - 1 = 6`` frames for mask memory and up to
``max_obj_ptrs_in_encoder - 1 = 15`` for object pointers, both at stride 1.
48 is that with room to spare; the anchor frame is exempt because it is a
conditioning frame and is what the target *is*."""


@dataclasses.dataclass
class Timing:
  """Wall clock, split the way the handoff's gate is written.

  Reported, never asserted on.  A number measured on this machine inside a
  simulation loop is not a deployment latency and this file will not pretend
  otherwise."""

  preprocess_ms: list = dataclasses.field(default_factory=list)
  anchor_ms: list = dataclasses.field(default_factory=list)
  infer_ms: list = dataclasses.field(default_factory=list)

  def summary(self) -> dict:
    def pct(xs):
      if not xs:
        return None
      a = np.asarray(xs)
      return {"p50": float(np.percentile(a, 50)),
              "p90": float(np.percentile(a, 90)),
              "p95": float(np.percentile(a, 95)),
              "p99": float(np.percentile(a, 99)),
              "max": float(a.max()), "n": len(xs)}
    return {"preprocess_ms": pct(self.preprocess_ms),
            "anchor_ms": pct(self.anchor_ms),
            "infer_ms": pct(self.infer_ms)}


class Sam2StreamingPredictor:
  """SAM2.1 driven one frame at a time, with no video on disk."""

  def __init__(self, checkpoint=CHECKPOINT, model_cfg: str = MODEL_CFG,
               device: str = "cuda", bf16: bool = True,
               vos_optimized: bool = False,
               memory_frames: int = MEMORY_FRAMES,
               max_anchors: int = MAX_ANCHORS) -> None:
    import torch
    from sam2.build_sam import build_sam2_video_predictor
    import sam2.sam2_video_predictor as vp

    # ``propagate_in_video`` wraps its (length-one, here) processing order in a
    # tqdm bar.  At 30 Hz that is a progress bar per frame on stderr.  Nothing
    # else in the module uses the name.
    vp.tqdm = lambda x, **kw: x

    self.torch = torch
    self.device = torch.device(device)
    self.bf16 = bool(bf16) and self.device.type == "cuda"
    self.memory_frames = int(memory_frames)
    self.max_anchors = max(1, int(max_anchors))
    self.checkpoint = pathlib.Path(checkpoint)
    self.model_cfg = str(model_cfg)
    t0 = time.perf_counter()
    self.model = build_sam2_video_predictor(
      model_cfg, str(checkpoint), device=device, vos_optimized=vos_optimized)
    self.load_s = time.perf_counter() - t0
    # This wheel does not contain SAM2's optional CUDA connected-components
    # extension.  Upstream otherwise retries it on the first mask, emits a
    # traceback-like warning, then skips the operation.  Disable only that
    # unavailable operation up front; all other video post-processing stays on.
    self.hole_filling = bool(getattr(self.model, "fill_hole_area", 0) > 0)
    if self.hole_filling:
      try:
        from sam2 import _C as _sam2_cuda_extension  # noqa: F401
      except ImportError:
        self.model.fill_hole_area = 0
        self.hole_filling = False
        print("SAM2: optional CUDA hole filling unavailable; disabled")
    # A refresh on an already-tracked frame is a conditioning correction.  It
    # lets us re-use that frame's cached backbone instead of appending and
    # processing the same image a second time merely to make it "initial".
    self.model.add_all_frames_to_correct_as_cond = True

    self.mean = torch.tensor(IMG_MEAN, device=self.device).view(3, 1, 1)
    self.std = torch.tensor(IMG_STD, device=self.device).view(3, 1, 1)
    self.size = int(self.model.image_size)
    self.timing = Timing()
    self.state = None
    self._anchor_idx = -1
    self.vos_optimized = bool(vos_optimized)
    self.warmup_ms: float | None = None

  # -- the Predictor protocol ---------------------------------------------

  def anchor(self, image: np.ndarray, mask: np.ndarray) -> None:
    """Take a mask the depth pipeline chose as the definition of the target."""
    if self.state is None:
      self.state = self._new_state(*image.shape[:2])
    idx = self._push(image)
    self._anchor_at(idx, mask)

  def reanchor(self, mask: np.ndarray) -> None:
    """Correct the most recently propagated frame without pushing it twice."""
    if self.state is None or not self.state["images"]:
      raise RuntimeError("cannot re-anchor SAM before a frame has been pushed")
    self._anchor_at(self.state["num_frames"] - 1, mask)

  def warmup(self, image_shape: tuple[int, int]) -> None:
    """Pay CUDA's one-time cost before a real target starts the policy.

    The first real ``add_new_mask`` on this rig takes about 245 ms while later
    anchors take only a few milliseconds.  That cold start used to land after
    the control loop had begun, so the first usable observation immediately
    tripped the stale-frame hold.  Exercise both the prompt and propagation
    paths on an empty synthetic image, reset every tracking datum, and reset
    the timing counters so the run report contains real frames only.
    """
    h, w = map(int, image_shape)
    if h <= 0 or w <= 0:
      raise ValueError("SAM warmup image dimensions must be positive")
    image = np.zeros((h, w), dtype=np.uint8)
    target = np.zeros((h, w), dtype=bool)
    cy, cx = h // 2, w // 2
    target[max(0, cy - 4):min(h, cy + 5),
           max(0, cx - 4):min(w, cx + 5)] = True
    t0 = time.perf_counter()
    self.anchor(image, target)
    self.propagate(image)
    self.reset()
    self.warmup_ms = (time.perf_counter() - t0) * 1000.0
    self.timing = Timing()

  def _anchor_at(self, idx: int, mask: np.ndarray) -> None:
    t0 = time.perf_counter()
    m = self.torch.as_tensor(np.ascontiguousarray(mask.astype(bool)))
    with self._amp():
      self.model.add_new_mask(self.state, frame_idx=idx, obj_id=1, mask=m)
    if self.device.type == "cuda":
      self.torch.cuda.synchronize()
    self.timing.anchor_ms.append((time.perf_counter() - t0) * 1000.0)
    self._anchor_idx = idx
    self._trim_anchors()

  def propagate(self, image: np.ndarray) -> np.ndarray | None:
    """Advance the target one frame.  ``None`` before anything is anchored."""
    if self.state is None or self._anchor_idx < 0:
      # Still record the frame so indices stay dense; SAM's temporal positions
      # are frame *indices*, and a gap in them is a gap in time it will believe.
      if self.state is not None:
        self._push(image)
      return None
    idx = self._push(image)
    t0 = time.perf_counter()
    with self._amp():
      gen = self.model.propagate_in_video(self.state, start_frame_idx=idx,
                                          max_frame_num_to_track=0)
      try:
        _, _, logits = next(gen)
      except StopIteration:
        return None
      finally:
        gen.close()
      mask = (logits[0, 0] > 0)
      # Reduced on GPU; only the packed bool crosses the bus.  The handoff
      # names the audit's full-logit CPU transfer as one of its cost centres.
      out = mask.cpu().numpy()
    if self.device.type == "cuda":
      self.torch.cuda.synchronize()
    self.timing.infer_ms.append((time.perf_counter() - t0) * 1000.0)
    self._prune(idx)
    return out

  def reset(self) -> None:
    """Drop the target.  Called by the tracker on place, loss, episode reset."""
    if self.state is not None:
      self.model.reset_state(self.state)
      self.state["images"] = []
      self.state["num_frames"] = 0
      self.state["cached_features"] = {}
    self._anchor_idx = -1

  # -- internals -----------------------------------------------------------

  def _amp(self):
    import contextlib
    if not self.bf16:
      return contextlib.nullcontext()
    return self.torch.autocast("cuda", dtype=self.torch.bfloat16)

  def _new_state(self, h: int, w: int) -> dict:
    """``init_state``'s dictionary, minus the video.

    Every key here is read by name somewhere in ``sam2_video_predictor``; this
    is that structure with ``images`` growable and ``num_frames`` counted as
    frames arrive.  The warm-up call ``init_state`` makes on frame 0 is not
    possible before a frame exists, and happens on the first ``anchor``.
    """
    return {
      "images": [], "num_frames": 0,
      "offload_video_to_cpu": False, "offload_state_to_cpu": False,
      "video_height": int(h), "video_width": int(w),
      "device": self.device, "storage_device": self.device,
      "point_inputs_per_obj": {}, "mask_inputs_per_obj": {},
      "cached_features": {}, "constants": {},
      "obj_id_to_idx": OrderedDict(), "obj_idx_to_id": OrderedDict(),
      "obj_ids": [], "output_dict_per_obj": {}, "temp_output_dict_per_obj": {},
      "frames_tracked_per_obj": {},
    }

  def _push(self, image: np.ndarray) -> int:
    """Normalise one frame onto the model's grid and append it."""
    torch = self.torch
    t0 = time.perf_counter()
    if image.ndim == 2:
      image = np.repeat(image[..., None], 3, axis=2)
    if image.shape[2] == 1:
      image = np.repeat(image, 3, axis=2)
    x = torch.as_tensor(np.ascontiguousarray(image[..., :3]),
                        device=self.device)
    x = x.permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
    # Bicubic to the square grid, matching PIL's default in
    # ``_load_img_as_tensor``.  848x480 -> 1024x1024 is an upscale on both
    # axes, so there is nothing to antialias.
    x = torch.nn.functional.interpolate(
      x, size=(self.size, self.size), mode="bicubic", align_corners=False)
    x = x.clamp_(0.0, 1.0)[0].sub_(self.mean).div_(self.std)
    self.state["images"].append(x)
    self.state["num_frames"] = len(self.state["images"])
    if self.device.type == "cuda":
      torch.cuda.synchronize()
    self.timing.preprocess_ms.append((time.perf_counter() - t0) * 1000.0)
    return self.state["num_frames"] - 1

  def _trim_anchors(self) -> None:
    """Keep outputs *and their full-resolution prompts* for recent anchors."""
    for obj_idx, obj in self.state["output_dict_per_obj"].items():
      output_cond = obj["cond_frame_outputs"]
      temp_cond = self.state["temp_output_dict_per_obj"][obj_idx][
        "cond_frame_outputs"]
      keys = sorted(set(output_cond) | set(temp_cond))
      drop = keys[:-self.max_anchors]
      for k in drop:
        output_cond.pop(k, None)
        temp_cond.pop(k, None)
        # Each prompt is a 1x1x1024x1024 float tensor (~4 MiB).
        self.state["mask_inputs_per_obj"].get(obj_idx, {}).pop(k, None)
        self.state["point_inputs_per_obj"].get(obj_idx, {}).pop(k, None)

  def _prune(self, idx: int) -> None:
    """Forget frames the model can no longer reach.

    Without this the state keeps a 1024x1024x3 tensor and a memory-encoder
    output for every frame of the episode.  With it, cost is flat in episode
    length -- which is the difference between a benchmark and a deployment.
    """
    cut = idx - self.memory_frames
    if cut < 0:
      return
    imgs = self.state["images"]
    for i in range(max(0, cut - 8), cut + 1):
      imgs[i] = None                     # index stays valid, tensor is freed
    for obj in self.state["output_dict_per_obj"].values():
      for k in [k for k in obj["non_cond_frame_outputs"] if k <= cut]:
        obj["non_cond_frame_outputs"].pop(k, None)
    for tracked in self.state["frames_tracked_per_obj"].values():
      for k in [k for k in tracked if k <= cut]:
        tracked.pop(k, None)

  # -- reporting -----------------------------------------------------------

  def report(self) -> dict:
    prompts = (sum(len(v) for v in self.state["mask_inputs_per_obj"].values())
               if self.state is not None else 0)
    return {"checkpoint": str(self.checkpoint), "model_cfg": self.model_cfg,
            "vos_optimized": self.vos_optimized, "bf16": self.bf16,
            "hole_filling": self.hole_filling,
            "load_s": round(self.load_s, 2), "image_size": self.size,
            "warmup_ms": self.warmup_ms,
            "memory_frames": self.memory_frames,
            "max_anchors": self.max_anchors,
            "retained_prompt_frames": prompts,
            "timing": self.timing.summary()}
