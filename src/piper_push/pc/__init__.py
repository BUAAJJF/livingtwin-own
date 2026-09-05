"""The point-cloud line (branch ``yf/pc``, 2026-09-06).

No mask, no target channel, no detector: the policy sees the workspace as the
depth camera measures it -- a metric depth image (P0) or a point cloud in the
robot base frame (P1/P2) -- at the D455's own 30 Hz cadence, with the age of
what it is looking at as an input.  Everything here is shared by the
simulator and the deployment so that the two build the observation the same
way; the modules say where they differ.

  cloud.py      unprojection, workspace crop, 30-of-50 Hz cadence, latency
  encoders.py   PointNet, the point-patch transformer, the depth ResNet
  models.py     the recurrent policy that takes point sets and images
  grasp.py      analytic top-K grasp candidates and the candidate lock (P2)
"""
