"""Generate the rig build sheet, in English or Chinese, from the task itself.

Every dimension on the drawings is imported from the modules the simulation
uses, so a sheet handed to somebody building the real thing cannot disagree
with what the policy was trained against.  Change the camera pose in
``piper_push.camera`` and regenerate; there is no second copy of the number to
forget.

    python scripts/make_rig_sheet.py --lang en --out docs/rig_build_sheet.html
    python scripts/make_rig_sheet.py --lang zh --out docs/rig_build_sheet.zh.html

The camera frames are rendered separately (they need a GPU) and embedded from
a PNG; pass --frames to point at one.
"""

from __future__ import annotations

import argparse
import base64
import math
import pathlib

# mjlab loads registered task packages on import.  Importing the task
# module directly first would register every task, then let mjlab try to
# register them again and refuse.
import mjlab  # noqa: F401

from piper_push import camera, objects
from piper_push.tasks.pick_place import env_cfg as pc

MM = 1000.0
TABLE = (-0.25, 0.98, -0.42, 0.70)

CAM = camera.CAMERA_POS
AIM = camera.CAMERA_AIM
FOVY = camera.FOVY_DEG
FOVX = camera.fovx_deg()
W, H = camera.WIDTH, camera.HEIGHT

GROUND = math.hypot(CAM[0], CAM[1])
BEARING = math.degrees(math.atan2(CAM[1], CAM[0]))
VIEW = (AIM[0] - CAM[0], AIM[1] - CAM[1])
VIEW_LEN = math.hypot(*VIEW)
TILT = math.degrees(math.atan2(CAM[2] - AIM[2], VIEW_LEN))
RANGE = math.dist(CAM, AIM)
BIN_R = math.hypot(*objects.BIN_CENTER)
BIN_A = math.degrees(math.atan2(objects.BIN_CENTER[1], objects.BIN_CENTER[0]))
NEAR_GROUND = CAM[2] / math.tan(math.radians(TILT + FOVY / 2))
NEAR_RANGE = math.hypot(NEAR_GROUND, CAM[2])
BASE_RANGE = math.hypot(GROUND, CAM[2])
ARM_CLEARANCE = 0.271   # measured: 64 arms x 400 steps of random actions

T = {
  "en": {
    "lang": "en",
    "title": "PiPER-X tidying rig &mdash; camera build sheet",
    "eyebrow": "S2 &middot; vision stage &middot; rig geometry",
    "h1": "Where the camera goes",
    "lede": (f"One fixed third-person depth camera, on the robot&rsquo;s left, "
             f"{CAM[2] * MM:.0f}&#8202;mm above the table, tilted {TILT:.0f}&deg; "
             "down. Every number below was chosen by measurement against eight "
             "candidate viewpoints, not by eye."),
    "k_run": "Along the table", "k_run_sub": "mm from base centre",
    "k_height": "Height", "k_height_sub": "mm above table",
    "k_bearing": "Bearing", "k_bearing_sub": "to the robot&rsquo;s left",
    "k_tilt": "Tilt", "k_tilt_sub": "below level",
    "k_fov": "Field of view", "k_fov_sub": f"vertical, {FOVX:.0f}&deg; horizontal",
    "k_clear": "Arm clearance", "k_clear_sub": "mm at worst",
    "h_plan": "Plan",
    "cap_plan": ("The <span class=\"tag zone\">object area</span> is where things "
                 "are put for the robot to tidy; the <span class=\"tag cam\">camera "
                 f"cone</span> is its {FOVX:.0f}&deg; horizontal field of view, "
                 "clipped to the table. The bin sits "
                 f"{abs(BIN_A):.0f}&deg; to the <em>right</em>, outside the object "
                 "area, so nothing is put down on top of it."),
    "h_elev": "Elevation",
    "cap_elev": (f"Section taken through the camera&rsquo;s line of sight, not "
                 "through the base &mdash; the bin is 36&deg; off that line and a "
                 "section through the base would draw it on top of the robot. The "
                 f"camera looks at a point {RANGE * MM:.0f}&#8202;mm away, "
                 f"{AIM[2] * MM:.0f}&#8202;mm above the table."),
    "h_view": "What it should see",
    "cap_view": ("Six frames from the simulated camera at this pose: near is "
                 "bright, far is dark, the target object is tinted. Use these to "
                 "check the real mounting &mdash; the arm should sit in the upper "
                 "middle, the bin should be fully inside the left edge with a "
                 "margin, and the near table should fill the bottom third. If the "
                 "bin is clipped, the camera is aimed too far left."),
    "h_geom": "Full geometry",
    "th": ("Item", "Quantity", "Value"),
    "h_hw": "What the camera has to be",
    "th_hw": ("Requirement", "Value", "Why"),
    "h_build": "Building it",
    "h_notes": "Two things worth knowing before you mount anything",
    "note1_t": "Precision is not the goal.",
    "note1": (f"The policy is trained with the camera pose randomised around this "
              f"nominal &mdash; &plusmn;{camera.POS_JITTER_M * MM:.0f}&#8202;mm and "
              f"&plusmn;{math.degrees(camera.ROT_JITTER_RAD):.0f}&deg; &mdash; so "
              "calibration only has to land inside that envelope, not hit the "
              f"number. That is also why the field of view is {FOVY:.0f}&deg; "
              "rather than the 45&deg; that would just barely fit: the extra "
              "margin is what a mis-mounted camera spends. Mount it solidly, "
              "though; a camera that moves after calibration is a camera that is "
              "wrong, and nothing in the system will notice."),
    "note2_t": "The bin side is the one place it must not go.",
    "note2": ("Of eight candidate viewpoints, the one placed near the bin was the "
              "only failure: the target object was visible in 58% of frames "
              "against 100% everywhere else, because the bin walls occlude the "
              "object area and the view runs nearly along the table. It is the "
              "position that feels right &mdash; close to where the placing "
              "happens &mdash; and it is the one that does not work."),
    "footer": ("Not specified yet, on purpose: depth noise and dropout, to be "
               "measured on the real sensor and put back into simulation rather "
               "than guessed; the table plane relative to the base; and the mask "
               "source. Those come after the rig exists."),
    "svg": {
      "plan": "PLAN &#183; millimetres from the centre of the robot base "
              "&#183; +x is the direction the base faces",
      "elev": "ELEVATION &#183; section along the camera&#8217;s line of sight, "
              "not through the base &#183; millimetres",
      "base": "BASE", "bin": "BIN", "cam": "CAMERA", "zone": "OBJECT AREA",
      "above": "above the table", "wide": f"{FOVX:.0f}&#176; wide on the table",
      "vfov": f"{FOVY:.0f}&#176; vertical", "inner": "inner",
      "aim": "aim point, %d from the camera" % round(RANGE * MM),
      "nearest": "nearest table seen: %d out" % round(NEAR_GROUND * MM),
    },
  },
  "zh": {
    "lang": "zh-CN",
    "title": "PiPER-X 收纳工作台 &mdash; 相机搭建图纸",
    "eyebrow": "S2 &middot; 视觉阶段 &middot; 工作台几何",
    "h1": "相机装在哪里",
    "lede": (f"一台固定的第三视角深度相机，位于机器人左前方，桌面上方 "
             f"{CAM[2] * MM:.0f}&#8202;mm，俯角 {TILT:.0f}&deg;。"
             "下面每一个数字都是在八个候选视角之间量出来的，不是看着定的。"),
    "k_run": "沿桌面距离", "k_run_sub": "mm，自底座中心",
    "k_height": "高度", "k_height_sub": "mm，桌面以上",
    "k_bearing": "方位角", "k_bearing_sub": "偏向机器人左侧",
    "k_tilt": "俯角", "k_tilt_sub": "低于水平",
    "k_fov": "视场角", "k_fov_sub": f"垂直，水平 {FOVX:.0f}&deg;",
    "k_clear": "手臂最近距离", "k_clear_sub": "mm，最坏情况",
    "h_plan": "平面图",
    "cap_plan": ("<span class=\"tag zone\">物体区</span> 是把待收纳物品放上去的范围；"
                 f"<span class=\"tag cam\">相机锥</span> 是它 {FOVX:.0f}&deg; 的水平视场"
                 f"在桌面上的投影。箱子在 <em>右侧</em> {abs(BIN_A):.0f}&deg;，"
                 "落在物体区之外，所以不会有东西被放到箱子上面。"),
    "h_elev": "剖面图",
    "cap_elev": (f"沿相机<strong>视线方向</strong>切开，不是过底座切 —— "
                 "箱子偏离视线 36&deg;，过底座的剖面会把它画到机器人身上。"
                 f"相机瞄准的点在 {RANGE * MM:.0f}&#8202;mm 外、桌面上方 "
                 f"{AIM[2] * MM:.0f}&#8202;mm 处。"),
    "h_view": "它应该看到什么",
    "cap_view": ("仿真相机在这个位姿下的六帧：近处亮、远处暗，目标物体被染色。"
                 "用它们核对真机安装 —— 手臂应该在画面上部偏中，"
                 "箱子应该完整落在左边缘以内并留有余量，近处桌面占下面三分之一。"
                 "如果箱子被切掉，说明相机瞄得太偏左。"),
    "h_geom": "完整尺寸",
    "th": ("部件", "量", "数值"),
    "h_hw": "对相机的要求",
    "th_hw": ("要求", "数值", "为什么"),
    "h_build": "搭建步骤",
    "h_notes": "动手之前值得知道的两件事",
    "note1_t": "精度不是目标。",
    "note1": (f"策略是在相机位姿绕这个标称值随机化的条件下训练的 —— "
              f"&plusmn;{camera.POS_JITTER_M * MM:.0f}&#8202;mm 和 "
              f"&plusmn;{math.degrees(camera.ROT_JITTER_RAD):.0f}&deg; —— "
              "所以标定只需要落进这个包络，不需要正好命中这个数。"
              f"视场取 {FOVY:.0f}&deg; 而不是刚好装得下的 45&deg;，也是同一个道理："
              "多出来的余量就是留给装偏的。但一定要装牢 —— "
              "标定之后还会移动的相机就是一台错的相机，而系统里没有任何东西会察觉。"),
    "note2_t": "唯一不能放的位置是箱子那一侧。",
    "note2": ("八个候选视角里，只有靠近箱子的那个失败了：目标物体的可见率只有 58%，"
              "其余位置都是 100%。原因是箱壁遮挡了物体区，而且视线几乎贴着桌面掠过。"
              "它恰恰是直觉上最想放的位置 —— 离放置动作最近 —— 也是唯一不能用的。"),
    "footer": ("有意留空的部分：深度噪声与空洞（必须在真实传感器上实测后放回仿真，"
               "不能靠猜）、桌面相对底座的平面，以及掩码的来源。这些要等工作台搭起来之后再做。"),
    "svg": {
      "plan": "平面图 &#183; 单位 mm，自机器人底座中心量起 &#183; +x 为底座正面方向",
      "elev": "剖面图 &#183; 沿相机视线方向切开，非过底座 &#183; 单位 mm",
      "base": "底座", "bin": "箱子", "cam": "相机", "zone": "物体区",
      "above": "桌面以上", "wide": f"桌面上张开 {FOVX:.0f}&#176;",
      "vfov": f"垂直视场 {FOVY:.0f}&#176;", "inner": "内径",
      "aim": "瞄准点，距相机 %d" % round(RANGE * MM),
      "nearest": "能看到的最近桌面：%d 外" % round(NEAR_GROUND * MM),
    },
  },
}

BUILD = {
  "en": [
    "Set the robot on the table and treat the centre of its base, at the table "
    "surface, as the origin. <span class=\"mono\">+x</span> is the direction the "
    "base faces, <span class=\"mono\">+y</span> is to its left.",
    "Mark the object area and the bin position on the table first &mdash; they "
    "are what the camera has to frame, and they are far easier to measure than a "
    "camera on a stand.",
    f"Mount the camera so its lens centre lands on the plan position. A frame or "
    f"a clamp arm holds steadier than a tripod at this height, and the arm never "
    f"comes closer than {ARM_CLEARANCE * MM:.0f}&#8202;mm to it even flailing.",
    f"Aim it at a marker placed at the aim point "
    f"({AIM[0] * MM:.0f}, {AIM[1] * MM:.0f}) on the table. Do not aim by eye at "
    "the object area &mdash; the aim point is deliberately biased towards the bin.",
    "Compare a live depth frame against the six above. This catches a mis-aimed "
    "camera in seconds, long before calibration would.",
    "Calibrate properly: mount a board on the gripper, drive 20+ poses spanning "
    "the object area, solve eye-to-hand for camera&nbsp;&rarr;&nbsp;base.",
  ],
  "zh": [
    "把机器人放在桌上，以底座中心在桌面上的投影为原点。"
    "<span class=\"mono\">+x</span> 是底座正面朝向，<span class=\"mono\">+y</span> 指向它的左侧。",
    "先在桌面上标出物体区和箱子的位置 —— 它们才是相机要框住的东西，"
    "而且在桌面上量比量一台架在半空的相机容易得多。",
    f"安装相机，使镜头中心落在平面图给出的位置。这个高度上，"
    f"支架或夹臂比三脚架稳；机械臂即使乱挥，离相机也不会近于 {ARM_CLEARANCE * MM:.0f}&#8202;mm。",
    f"把相机瞄向桌面上放在瞄准点 ({AIM[0] * MM:.0f}, {AIM[1] * MM:.0f}) 的标记物。"
    "不要凭眼睛瞄物体区 —— 瞄准点是有意偏向箱子一侧的。",
    "拿实时深度画面和上面那六帧比对。这一步能在几秒内发现瞄偏，远早于标定。",
    "正式标定：把标定板装在夹爪上，走 20 个以上覆盖物体区的位姿，"
    "解 eye-to-hand，得到 相机&nbsp;&rarr;&nbsp;底座 的变换。",
  ],
}

GEOM_ROWS = {
  "en": [
    ("Camera", "position in base frame (x, y, z)",
     f"{CAM[0]:+.3f}, {CAM[1]:+.3f}, {CAM[2]:+.3f} m"),
    ("", "aim point", f"{AIM[0]:+.3f}, {AIM[1]:+.3f}, {AIM[2]:+.3f} m"),
    ("", "range to aim point", f"{RANGE:.3f} m"),
    ("", "pan from +x axis", f"{math.degrees(math.atan2(*VIEW[::-1])):.1f}&deg;"),
    ("", "tilt below horizontal", f"{TILT:.1f}&deg;"),
    ("", "resolution", f"{W} &times; {H}"),
    ("Bin", "centre", f"{objects.BIN_CENTER[0]:+.3f}, {objects.BIN_CENTER[1]:+.3f} m"),
    ("", "from base", f"{BIN_R:.3f} m at {BIN_A:.1f}&deg;"),
    ("", "inner opening",
     f"{2000 * objects.BIN_INNER[0]:.0f} &times; {2000 * objects.BIN_INNER[1]:.0f} mm"),
    ("", "wall height &times; thickness",
     f"{1000 * objects.BIN_WALL_HEIGHT:.0f} &times; {1000 * objects.BIN_WALL_THICKNESS:.0f} mm"),
    ("Object area", "radius from base",
     f"{pc.SPAWN_RADIUS[0]:.3f} &ndash; {pc.SPAWN_RADIUS[1]:.3f} m"),
    ("", "bearing",
     f"{math.degrees(pc.SPAWN_ANGLE[0]):.0f}&deg; to {math.degrees(pc.SPAWN_ANGLE[1]):+.0f}&deg;"),
    ("Objects", "width across the jaws",
     f"{1000 * objects.OBJECT_WIDTH_RANGE[0]:.0f} &ndash; {2000 * objects.OBJECT_MAX_HALF_WIDTH:.0f} mm"),
    ("", "height",
     f"{1000 * objects.OBJECT_HEIGHT_FLOOR:.0f} &ndash; {2000 * objects.OBJECT_MAX_HALF_HEIGHT:.0f} mm"),
    ("", "mass",
     f"{1000 * objects.OBJECT_MASS_RANGE[0]:.0f} &ndash; {1000 * objects.OBJECT_MASS_RANGE[1]:.0f} g"),
  ],
}
GEOM_ROWS["zh"] = [
  ("相机", "底座系位置 (x, y, z)", GEOM_ROWS["en"][0][2]),
  ("", "瞄准点", GEOM_ROWS["en"][1][2]),
  ("", "到瞄准点距离", GEOM_ROWS["en"][2][2]),
  ("", "相对 +x 的水平方位", GEOM_ROWS["en"][3][2]),
  ("", "低于水平的俯角", GEOM_ROWS["en"][4][2]),
  ("", "分辨率", GEOM_ROWS["en"][5][2]),
  ("箱子", "中心", GEOM_ROWS["en"][6][2]),
  ("", "距底座", GEOM_ROWS["en"][7][2]),
  ("", "内开口", GEOM_ROWS["en"][8][2]),
  ("", "壁高 &times; 壁厚", GEOM_ROWS["en"][9][2]),
  ("物体区", "距底座半径", GEOM_ROWS["en"][10][2]),
  ("", "方位角范围", GEOM_ROWS["en"][11][2]),
  ("物体", "夹持方向宽度", GEOM_ROWS["en"][12][2]),
  ("", "高度", GEOM_ROWS["en"][13][2]),
  ("", "质量", GEOM_ROWS["en"][14][2]),
]

HW_ROWS = {
  "en": [
    ("Output", "depth", "Colour is not used. A stereo or structured-light depth "
     "camera is fine; the policy never sees RGB."),
    ("Horizontal field of view", f"&ge; {FOVX:.0f}&deg;",
     "Anything narrower clips the bin from this mounting distance."),
    ("Vertical field of view", f"&ge; {FOVY:.0f}&deg;", ""),
    ("Working range", f"{NEAR_RANGE:.1f} &ndash; {BASE_RANGE + 0.1:.1f} m",
     f"Nearest table it sees is {NEAR_RANGE:.2f}&#8202;m away, the robot base "
     f"{BASE_RANGE:.2f}&#8202;m. A camera with a 0.6&#8202;m minimum will be blind "
     "along the near edge."),
    ("Resolution", f"&ge; {W} &times; {H}",
     "What the policy consumes. A higher-resolution sensor is fine and gets "
     "downsampled; the number that matters is the field of view."),
    ("Mount", "rigid",
     "The pose is calibrated once. A camera that shifts afterwards is a camera "
     "that is wrong, and nothing in the system will notice."),
    ("Frame rate", "&ge; 30 Hz",
     "The policy runs at 50&#8202;Hz on proprioception; the image can lag it, but "
     "not by much."),
  ],
  "zh": [
    ("输出", "深度", "不用彩色。双目或结构光深度相机都可以；策略从不看 RGB。"),
    ("水平视场", f"&ge; {FOVX:.0f}&deg;", "在这个安装距离下，更窄的视场会把箱子切掉。"),
    ("垂直视场", f"&ge; {FOVY:.0f}&deg;", ""),
    ("工作距离", f"{NEAR_RANGE:.1f} &ndash; {BASE_RANGE + 0.1:.1f} m",
     f"它能看到的最近桌面在 {NEAR_RANGE:.2f}&#8202;m 处，机器人底座在 "
     f"{BASE_RANGE:.2f}&#8202;m。最小工作距离 0.6&#8202;m 的相机会在近边缘瞎掉。"),
    ("分辨率", f"&ge; {W} &times; {H}",
     "这是策略实际吃进去的尺寸。传感器分辨率更高没问题，会被降采样；真正要卡的是视场。"),
    ("安装", "刚性",
     "位姿只标定一次。之后还会移动的相机就是错的，而系统里没有任何东西会察觉。"),
    ("帧率", "&ge; 30 Hz",
     "策略以 50&#8202;Hz 跑本体感知；图像可以滞后，但不能滞后太多。"),
  ],
}


def _sector_pts(n: int = 25):
  a0, a1 = pc.SPAWN_ANGLE
  return [
    (r * math.cos(a0 + (a1 - a0) * i / (n - 1)), r * math.sin(a0 + (a1 - a0) * i / (n - 1)))
    for r in pc.SPAWN_RADIUS
    for i in range(n)
  ]


def plan_svg(L: dict) -> str:
  ox, oy = 330, 850
  o, a = [], None
  o_append = o.append

  def P(x, y):
    return ox + x * MM, oy - y * MM

  o_append('<svg viewBox="0 0 1360 1330" role="img" aria-label="plan">')
  o_append('<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
           'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
           '<path d="M0,0 L10,5 L0,10 z" fill="var(--rule-strong)"/></marker>')
  tx, ty = P(TABLE[0], TABLE[3])
  tw, th = (TABLE[1] - TABLE[0]) * MM, (TABLE[3] - TABLE[2]) * MM
  o_append(f'<clipPath id="tbl"><rect x="{tx:.0f}" y="{ty:.0f}" width="{tw:.0f}" '
           f'height="{th:.0f}" rx="14"/></clipPath></defs>')
  o_append(f'<rect class="table" x="{tx:.0f}" y="{ty:.0f}" width="{tw:.0f}" '
           f'height="{th:.0f}" rx="14"/>')

  bearing = math.atan2(*VIEW[::-1])
  half = math.radians(FOVX / 2)
  c = P(CAM[0], CAM[1])
  f1 = P(CAM[0] + 2.2 * math.cos(bearing - half), CAM[1] + 2.2 * math.sin(bearing - half))
  f2 = P(CAM[0] + 2.2 * math.cos(bearing + half), CAM[1] + 2.2 * math.sin(bearing + half))
  o_append(f'<path class="fov" clip-path="url(#tbl)" d="M{c[0]:.1f},{c[1]:.1f} '
           f'L{f1[0]:.1f},{f1[1]:.1f} L{f2[0]:.1f},{f2[1]:.1f} Z"/>')

  r0, r1 = pc.SPAWN_RADIUS
  a0, a1 = pc.SPAWN_ANGLE
  p0, p1 = P(r0 * math.cos(a0), r0 * math.sin(a0)), P(r1 * math.cos(a0), r1 * math.sin(a0))
  p2, p3 = P(r1 * math.cos(a1), r1 * math.sin(a1)), P(r0 * math.cos(a1), r0 * math.sin(a1))
  o_append(f'<path class="zone" d="M{p0[0]:.1f},{p0[1]:.1f} L{p1[0]:.1f},{p1[1]:.1f} '
           f'A{r1 * MM:.0f},{r1 * MM:.0f} 0 0 0 {p2[0]:.1f},{p2[1]:.1f} '
           f'L{p3[0]:.1f},{p3[1]:.1f} '
           f'A{r0 * MM:.0f},{r0 * MM:.0f} 0 0 1 {p0[0]:.1f},{p0[1]:.1f} Z"/>')
  lab = P(0.355 * math.cos(0.30), 0.355 * math.sin(0.30))
  o_append(f'<text class="zone-label" x="{lab[0]:.0f}" y="{lab[1]:.0f}" '
           f'text-anchor="middle">{L["zone"]}</text>')
  o_append(f'<text class="zone-sub" x="{lab[0]:.0f}" y="{lab[1] + 32:.0f}" '
           f'text-anchor="middle">r {pc.SPAWN_RADIUS[0] * MM:.0f}&#8211;'
           f'{pc.SPAWN_RADIUS[1] * MM:.0f}</text>')

  bw = objects.BIN_INNER[0] + objects.BIN_WALL_THICKNESS
  bh = objects.BIN_INNER[1] + objects.BIN_WALL_THICKNESS
  tl = P(objects.BIN_CENTER[0] - bw, objects.BIN_CENTER[1] + bh)
  o_append(f'<rect class="bin" x="{tl[0]:.1f}" y="{tl[1]:.1f}" '
           f'width="{2 * bw * MM:.0f}" height="{2 * bh * MM:.0f}" rx="4"/>')
  tli = P(objects.BIN_CENTER[0] - objects.BIN_INNER[0],
          objects.BIN_CENTER[1] + objects.BIN_INNER[1])
  o_append(f'<rect class="bin-in" x="{tli[0]:.1f}" y="{tli[1]:.1f}" '
           f'width="{2 * objects.BIN_INNER[0] * MM:.0f}" '
           f'height="{2 * objects.BIN_INNER[1] * MM:.0f}"/>')
  bc = P(*objects.BIN_CENTER)
  o_append(f'<text class="part" x="{bc[0]:.0f}" y="{bc[1] + 7:.0f}" '
           f'text-anchor="middle">{L["bin"]}</text>')
  o_append(f'<text class="note" x="{bc[0]:.0f}" y="{bc[1] + 118:.0f}" '
           f'text-anchor="middle">{2000 * objects.BIN_INNER[0]:.0f} &#215; '
           f'{2000 * objects.BIN_INNER[1]:.0f} {L["inner"]}</text>')

  b = P(0, 0)
  o_append(f'<circle class="base" cx="{b[0]:.0f}" cy="{b[1]:.0f}" r="58"/>')
  o_append(f'<text class="part" x="{b[0]:.0f}" y="{b[1] - 78:.0f}" '
           f'text-anchor="middle">{L["base"]}</text>')
  fx = P(0.185, 0)
  o_append(f'<line class="axis" x1="{b[0]:.0f}" y1="{b[1]:.0f}" x2="{fx[0]:.0f}" '
           f'y2="{fx[1]:.0f}" marker-end="url(#ar)"/>')
  o_append(f'<text class="axis-label" x="{fx[0] + 14:.0f}" y="{fx[1] + 6:.0f}">+x</text>')

  o_append(f'<line class="dim" x1="{b[0]:.0f}" y1="{b[1]:.0f}" x2="{c[0]:.0f}" '
           f'y2="{c[1]:.0f}" marker-start="url(#ar)" marker-end="url(#ar)"/>')
  o_append(f'<text class="dim-label" x="{(b[0] + c[0]) / 2 - 20:.0f}" '
           f'y="{(b[1] + c[1]) / 2 - 22:.0f}" text-anchor="middle">'
           f'{GROUND * MM:.0f}</text>')
  o_append(f'<line class="dim" x1="{b[0]:.0f}" y1="{b[1]:.0f}" x2="{bc[0]:.0f}" '
           f'y2="{bc[1]:.0f}" marker-start="url(#ar)" marker-end="url(#ar)"/>')
  o_append(f'<text class="dim-label" x="{(b[0] + bc[0]) / 2 + 44:.0f}" '
           f'y="{(b[1] + bc[1]) / 2 + 22:.0f}">{BIN_R * MM:.0f}</text>')

  rr, ang = 235, math.radians(BEARING)
  s, e = P(rr / MM, 0), P(rr / MM * math.cos(ang), rr / MM * math.sin(ang))
  o_append(f'<path class="dim" d="M{s[0]:.1f},{s[1]:.1f} A{rr},{rr} 0 0 0 '
           f'{e[0]:.1f},{e[1]:.1f}"/>')
  mid = P(rr / MM * 0.62 * math.cos(ang / 2), rr / MM * 0.62 * math.sin(ang / 2))
  o_append(f'<text class="dim-label" x="{mid[0]:.0f}" y="{mid[1] - 8:.0f}" '
           f'text-anchor="middle">+{BEARING:.1f}&#176;</text>')

  rb, bang = 168, math.radians(BIN_A)
  s2, e2 = P(rb / MM, 0), P(rb / MM * math.cos(bang), rb / MM * math.sin(bang))
  o_append(f'<path class="dim" d="M{s2[0]:.1f},{s2[1]:.1f} A{rb},{rb} 0 0 1 '
           f'{e2[0]:.1f},{e2[1]:.1f}"/>')
  mid2 = P(rb / MM * 1.62 * math.cos(bang / 2), rb / MM * 1.62 * math.sin(bang / 2))
  o_append(f'<text class="dim-label" x="{mid2[0]:.0f}" y="{mid2[1] + 8:.0f}">'
           f'&#8722;{abs(BIN_A):.1f}&#176;</text>')

  o_append(f'<circle class="cam" cx="{c[0]:.0f}" cy="{c[1]:.0f}" r="16"/>')
  o_append(f'<text class="cam-label" x="{c[0] + 26:.0f}" y="{c[1] - 14:.0f}">'
           f'{L["cam"]}</text>')
  o_append(f'<text class="cam-sub" x="{c[0] + 26:.0f}" y="{c[1] + 12:.0f}">'
           f'{CAM[2] * MM:.0f} {L["above"]}</text>')
  o_append(f'<text class="cam-sub" x="{c[0] + 26:.0f}" y="{c[1] + 38:.0f}">'
           f'{L["wide"]}</text>')
  o_append(f'<text class="note" x="56" y="1296">{L["plan"]}</text>')
  o_append('</svg>')
  return "\n".join(o)


def elev_svg(L: dict) -> str:
  ox, oy = 120, 620
  n = (VIEW[0] / VIEW_LEN, VIEW[1] / VIEW_LEN)
  tilt = math.radians(TILT)
  scale = 900.0 / 1.15
  o = []
  ap = o.append

  def P(u, z):
    return ox + u * scale, oy - z * scale

  def u_of(pt):
    return (pt[0] - CAM[0]) * n[0] + (pt[1] - CAM[1]) * n[1]

  half = math.radians(FOVY / 2)
  u_near = CAM[2] / math.tan(tilt + half)
  u_far = CAM[2] / math.tan(tilt - half)

  ap('<svg viewBox="0 0 1120 780" role="img" aria-label="elevation">')
  ap('<defs><marker id="ar2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
     'markerHeight="7" orient="auto-start-reverse">'
     '<path d="M0,0 L10,5 L0,10 z" fill="var(--rule-strong)"/></marker></defs>')

  t0, t1 = P(-0.06, 0), P(1.13, 0)
  s0, s1 = P(u_near, 0), P(min(u_far, 1.13), 0)
  ap(f'<rect class="fov" x="{s0[0]:.0f}" y="{s0[1] - 7:.0f}" '
     f'width="{s1[0] - s0[0]:.0f}" height="14"/>')
  ap(f'<line class="tabletop" x1="{t0[0]:.0f}" y1="{t0[1]:.0f}" x2="{t1[0]:.0f}" '
     f'y2="{t1[1]:.0f}"/>')

  c = P(0, CAM[2])
  for ang, uh in ((tilt + half, u_near), (tilt - half, min(u_far, 1.13))):
    p = P(uh, 0)
    ap(f'<line class="fov-line" x1="{c[0]:.0f}" y1="{c[1]:.0f}" x2="{p[0]:.0f}" '
       f'y2="{p[1]:.0f}"/>')

  us = [u_of(p) for p in _sector_pts()]
  z0, z1 = P(min(us), 0), P(max(us), 0)
  ap(f'<rect class="zone" x="{z0[0]:.0f}" y="{z0[1] - 46:.0f}" '
     f'width="{z1[0] - z0[0]:.0f}" height="46"/>')
  ap(f'<text class="zone-sub" x="{(z0[0] + z1[0]) / 2:.0f}" y="{z0[1] - 58:.0f}" '
     f'text-anchor="middle">{L["zone"]}</text>')

  ub = u_of((0.0, 0.0))
  b = P(ub, 0)
  ap(f'<rect class="base" x="{b[0] - 46:.0f}" y="{b[1] - 52:.0f}" width="92" '
     'height="52" rx="7"/>')
  j, e = P(ub - 0.02, 0.30), P(ub - 0.30, 0.14)
  ap(f'<path class="arm" d="M{b[0]:.0f},{b[1] - 52:.0f} L{j[0]:.0f},{j[1]:.0f} '
     f'L{e[0]:.0f},{e[1]:.0f}"/>')
  ap(f'<text class="part" x="{b[0]:.0f}" y="{b[1] + 32:.0f}" '
     f'text-anchor="middle">{L["base"]}</text>')

  a_pt = P(VIEW_LEN, AIM[2])
  ap(f'<line class="axis" x1="{c[0]:.0f}" y1="{c[1]:.0f}" x2="{a_pt[0]:.0f}" '
     f'y2="{a_pt[1]:.0f}" marker-end="url(#ar2)"/>')
  ap(f'<circle class="aim" cx="{a_pt[0]:.0f}" cy="{a_pt[1]:.0f}" r="7"/>')
  ap(f'<line class="dim" x1="{a_pt[0]:.0f}" y1="{a_pt[1] + 10:.0f}" '
     f'x2="{a_pt[0]:.0f}" y2="{P(0, 0)[1] + 78:.0f}"/>')
  ap(f'<text class="note" x="{a_pt[0]:.0f}" y="{P(0, 0)[1] + 100:.0f}" '
     f'text-anchor="middle">{L["aim"]}</text>')

  hx = c[0] - 66
  ap(f'<line class="dim" x1="{hx:.0f}" y1="{P(0, 0)[1]:.0f}" x2="{hx:.0f}" '
     f'y2="{c[1]:.0f}" marker-start="url(#ar2)" marker-end="url(#ar2)"/>')
  ap(f'<text class="dim-label" x="{hx - 12:.0f}" y="{(c[1] + P(0, 0)[1]) / 2:.0f}" '
     f'text-anchor="end">{CAM[2] * MM:.0f}</text>')
  arc = 120
  a1p = P(arc / scale, CAM[2])
  a2p = P(arc / scale * math.cos(tilt), CAM[2] - arc / scale * math.sin(tilt))
  ap(f'<path class="dim" d="M{a1p[0]:.1f},{a1p[1]:.1f} A{arc},{arc} 0 0 1 '
     f'{a2p[0]:.1f},{a2p[1]:.1f}"/>')
  ap(f'<text class="dim-label" x="{a1p[0] + 18:.0f}" y="{a1p[1] + 46:.0f}">'
     f'{TILT:.1f}&#176;</text>')

  ap(f'<circle class="cam" cx="{c[0]:.0f}" cy="{c[1]:.0f}" r="16"/>')
  ap(f'<text class="cam-label" x="{c[0] + 24:.0f}" y="{c[1] - 16:.0f}">{L["cam"]}</text>')
  ap(f'<text class="cam-sub" x="{c[0] + 24:.0f}" y="{c[1] + 30:.0f}">{L["vfov"]}</text>')
  nc = P(u_near, 0)
  ap(f'<text class="note" x="{nc[0] + 8:.0f}" y="{nc[1] + 34:.0f}">{L["nearest"]}</text>')
  ap(f'<text class="note" x="40" y="756">{L["elev"]}</text>')
  ap('</svg>')
  return "\n".join(o)


SANS = {
  "en": 'ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif',
  # A Latin face first so the digits and units keep their shapes, then the CJK
  # faces: a Chinese system font asked to set "480 mm" sets it in full-width
  # glyphs and the number stops lining up with the drawings.
  "zh": 'ui-sans-serif,system-ui,-apple-system,"Segoe UI",'
        '"PingFang SC","Hiragino Sans GB","Microsoft YaHei",'
        '"Source Han Sans SC","Noto Sans CJK SC",sans-serif',
}
MONO = ('ui-monospace,SFMono-Regular,Menlo,Consolas,'
        '"PingFang SC","Microsoft YaHei",monospace')


def css(lang: str) -> str:
  sans = SANS[lang]
  # Chinese runs need more leading and no letter-spacing on the small caps
  # labels, which is a Latin device and only smears CJK apart.
  lh = "1.75" if lang == "zh" else "1.6"
  track = "0" if lang == "zh" else ".14em"
  upper = "none" if lang == "zh" else "uppercase"
  return f"""
:root {{
  --bg:#F5F6F8; --surface:#FFFFFF; --ink:#12161C; --muted:#5A6472;
  --rule:#D8DDE4; --rule-strong:#8B95A3; --cam:#C2410C; --zone:#0F766E;
  --table-fill:#ECEFF3;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#0D1014; --surface:#161A20; --ink:#E4E8EE; --muted:#98A2B0;
    --rule:#272E37; --rule-strong:#6B7684; --cam:#F97316; --zone:#2DD4BF;
    --table-fill:#1B2027;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#0D1014; --surface:#161A20; --ink:#E4E8EE; --muted:#98A2B0;
  --rule:#272E37; --rule-strong:#6B7684; --cam:#F97316; --zone:#2DD4BF;
  --table-fill:#1B2027;
}}
*,*::before,*::after {{ box-sizing:border-box; }}
html {{ -webkit-text-size-adjust:100%; }}
body {{ background:var(--bg); color:var(--ink); margin:0;
  font-family:{sans}; line-height:{lh}; -webkit-font-smoothing:antialiased; }}
img,svg {{ max-width:100%; }}
.wrap {{ max-width:960px; margin:0 auto; padding:56px 24px 96px;
  display:flex; flex-direction:column; gap:44px; }}
.mono {{ font-family:{MONO}; font-variant-numeric:tabular-nums; }}
.eyebrow {{ font-size:11px; letter-spacing:{track}; text-transform:{upper};
  color:var(--muted); font-weight:600; font-family:{MONO}; }}
h1 {{ font-size:clamp(28px,4.4vw,40px); line-height:1.2; letter-spacing:-.02em;
  margin:10px 0 0; text-wrap:balance; font-weight:700; }}
h2 {{ font-size:20px; margin:0 0 4px; font-weight:650; }}
p {{ margin:0; max-width:{'42em' if lang == 'zh' else '68ch'}; }}
.lede {{ color:var(--muted); font-size:17px; margin-top:14px; }}
section {{ display:flex; flex-direction:column; gap:14px; }}
.card {{ background:var(--surface); border:1px solid var(--rule); border-radius:10px;
  padding:20px; overflow-x:auto; }}
.card svg {{ display:block; width:100%; height:auto; min-width:520px; }}
.keys {{ display:grid; gap:1px; background:var(--rule); border:1px solid var(--rule);
  border-radius:10px; overflow:hidden;
  grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); }}
.key {{ background:var(--surface); padding:16px 18px; }}
.key dt {{ font-size:11px; letter-spacing:{track}; text-transform:{upper};
  color:var(--muted); font-weight:600; margin:0 0 6px; font-family:{MONO}; }}
.key dd {{ margin:0; font-size:22px; font-weight:600; font-family:{MONO};
  font-variant-numeric:tabular-nums; }}
.key dd span {{ font-size:13px; color:var(--muted); font-weight:500;
  font-family:{sans}; }}
table {{ border-collapse:collapse; width:100%; font-size:14px; }}
th,td {{ text-align:left; padding:9px 12px; border-bottom:1px solid var(--rule);
  vertical-align:top; }}
th {{ font-size:11px; letter-spacing:{track}; text-transform:{upper};
  color:var(--muted); font-weight:600; font-family:{MONO}; }}
td.n {{ font-family:{MONO}; font-variant-numeric:tabular-nums; white-space:nowrap; }}
tbody tr:last-child td {{ border-bottom:none; }}
ol {{ margin:0; padding-left:24px; display:flex; flex-direction:column; gap:11px;
  max-width:{'42em' if lang == 'zh' else '68ch'}; }}
li::marker {{ color:var(--cam); font-weight:700; font-family:{MONO}; }}
img.view {{ display:block; width:100%; height:auto; border-radius:6px;
  image-rendering:pixelated; }}
.cap {{ font-size:13px; color:var(--muted); }}
hr {{ border:none; border-top:1px solid var(--rule); margin:0; }}
.tag {{ display:inline-block; font-size:11px; letter-spacing:{track};
  text-transform:{upper}; font-weight:650; padding:3px 8px; border-radius:4px;
  font-family:{MONO}; }}
.tag.cam {{ color:var(--cam); background:color-mix(in srgb,var(--cam) 10%,transparent);
  border:1px solid var(--cam); }}
.tag.zone {{ color:var(--zone); background:color-mix(in srgb,var(--zone) 12%,transparent);
  border:1px solid var(--zone); }}
.table {{ fill:var(--table-fill); stroke:var(--rule); stroke-width:2; }}
.zone {{ fill:var(--zone); fill-opacity:.13; stroke:var(--zone); stroke-width:2.5;
  stroke-dasharray:9 7; }}
.zone-label {{ fill:var(--zone); font-size:27px; font-weight:700; font-family:{sans}; }}
.zone-sub {{ fill:var(--zone); font-size:20px; font-family:{MONO}; }}
.fov {{ fill:var(--cam); fill-opacity:.10; stroke:var(--cam); stroke-width:2;
  stroke-dasharray:6 8; }}
.fov-line {{ stroke:var(--cam); stroke-width:2; stroke-dasharray:6 8; }}
.bin {{ fill:none; stroke:var(--ink); stroke-width:3.5; }}
.bin-in {{ fill:var(--bg); stroke:var(--rule-strong); stroke-width:1.5; }}
.base {{ fill:var(--surface); stroke:var(--ink); stroke-width:3.5; }}
.arm {{ fill:none; stroke:var(--rule-strong); stroke-width:11;
  stroke-linecap:round; stroke-linejoin:round; }}
.tabletop {{ stroke:var(--ink); stroke-width:3; }}
.cam {{ fill:var(--cam); stroke:var(--surface); stroke-width:3; }}
.aim {{ fill:var(--cam); stroke:none; }}
.cam-label {{ fill:var(--cam); font-size:25px; font-weight:700; font-family:{sans}; }}
.cam-sub {{ fill:var(--muted); font-size:19px; font-family:{MONO}; }}
.part {{ fill:var(--ink); font-size:19px; font-weight:700; font-family:{MONO}; }}
.axis {{ stroke:var(--rule-strong); stroke-width:2; }}
.axis-label {{ fill:var(--rule-strong); font-size:20px; font-family:{MONO}; }}
.dim {{ stroke:var(--rule-strong); stroke-width:1.6; fill:none; }}
.dim-label {{ fill:var(--ink); font-size:21px; font-weight:600; font-family:{MONO};
  font-variant-numeric:tabular-nums; }}
.note {{ fill:var(--muted); font-size:18px; font-family:{MONO}; }}
@media print {{
  body {{ background:#fff; color:#000; }}
  .wrap {{ max-width:none; padding:0; gap:24px; }}
  .card,.keys,.key {{ background:#fff; break-inside:avoid; }}
  section {{ break-inside:avoid; }}
}}
@media (prefers-reduced-motion:reduce) {{ * {{ animation:none!important;
  transition:none!important; }} }}
"""


def page(lang: str, frames_b64: str | None) -> str:
  t = T[lang]
  L = t["svg"]
  keys = [
    (t["k_run"], f"{GROUND * MM:.0f}", t["k_run_sub"]),
    (t["k_height"], f"{CAM[2] * MM:.0f}", t["k_height_sub"]),
    (t["k_bearing"], f"+{BEARING:.1f}&deg;", t["k_bearing_sub"]),
    (t["k_tilt"], f"{TILT:.1f}&deg;", t["k_tilt_sub"]),
    (t["k_fov"], f"{FOVY:.0f}&deg;", t["k_fov_sub"]),
    (t["k_clear"], f"{ARM_CLEARANCE * MM:.0f}", t["k_clear_sub"]),
  ]
  key_html = "\n".join(
    f'  <div class="key"><dt>{k}</dt><dd>{v} <span>{s}</span></dd></div>'
    for k, v, s in keys
  )
  geom_html = "\n".join(
    f'      <tr><td>{i}</td><td>{q}</td><td class="n">{v}</td></tr>'
    for i, q, v in GEOM_ROWS[lang]
  )
  hw_html = "\n".join(
    f'      <tr><td>{r}</td><td class="n">{v}</td><td>{w}</td></tr>'
    for r, v, w in HW_ROWS[lang]
  )
  steps = "\n".join(f"    <li>{s}</li>" for s in BUILD[lang])
  view = ""
  if frames_b64:
    view = f"""
<section>
  <h2>{t["h_view"]}</h2>
  <div class="card"><img class="view" alt="camera frames"
    src="data:image/png;base64,{frames_b64}"></div>
  <p class="cap">{t["cap_view"]}</p>
</section>
"""
  return f"""<!DOCTYPE html>
<html lang="{t["lang"]}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{t["title"]}</title>
<style>{css(lang)}</style>
</head>
<body>
<div class="wrap">

<header>
  <div class="eyebrow">{t["eyebrow"]}</div>
  <h1>{t["h1"]}</h1>
  <p class="lede">{t["lede"]}</p>
</header>

<dl class="keys">
{key_html}
</dl>

<section>
  <h2>{t["h_plan"]}</h2>
  <div class="card">{plan_svg(L)}</div>
  <p class="cap">{t["cap_plan"]}</p>
</section>

<section>
  <h2>{t["h_elev"]}</h2>
  <div class="card">{elev_svg(L)}</div>
  <p class="cap">{t["cap_elev"]}</p>
</section>
{view}
<section>
  <h2>{t["h_geom"]}</h2>
  <table>
    <thead><tr><th>{t["th"][0]}</th><th>{t["th"][1]}</th><th>{t["th"][2]}</th></tr></thead>
    <tbody>
{geom_html}
    </tbody>
  </table>
</section>

<section>
  <h2>{t["h_hw"]}</h2>
  <table>
    <thead><tr><th>{t["th_hw"][0]}</th><th>{t["th_hw"][1]}</th><th>{t["th_hw"][2]}</th></tr></thead>
    <tbody>
{hw_html}
    </tbody>
  </table>
</section>

<section>
  <h2>{t["h_build"]}</h2>
  <ol>
{steps}
  </ol>
</section>

<section>
  <h2>{t["h_notes"]}</h2>
  <p><strong>{t["note1_t"]}</strong> {t["note1"]}</p>
  <p><strong>{t["note2_t"]}</strong> {t["note2"]}</p>
</section>

<hr>

<section><p class="cap">{t["footer"]}</p></section>

</div>
</body>
</html>
"""


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--lang", choices=sorted(T), default="en")
  p.add_argument("--out", required=True)
  p.add_argument("--frames", default=None,
                 help="PNG of camera frames to embed, from scripts/render_rig_view")
  a = p.parse_args()
  b64 = None
  if a.frames:
    b64 = base64.b64encode(pathlib.Path(a.frames).read_bytes()).decode()
  out = pathlib.Path(a.out)
  out.parent.mkdir(parents=True, exist_ok=True)
  out.write_text(page(a.lang, b64))
  print(f"wrote {out}  {out.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
  main()
