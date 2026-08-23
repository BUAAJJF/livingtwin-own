# Manifold Tech Odin 1 — SDK research & install notes

Written 2026-08-24. Host: Ubuntu 24.04.3 (noble), kernel 7.0.0-30-generic, x86_64.

---

## 1. What the device is

**Odin 1**, by **Manifold Tech Limited** (Hong Kong; manifoldtech.com.co). Marketed as a
"spatial memory module": a fused solid-state LiDAR + colour camera + IMU with an onboard
Rockchip SoC that runs the vendor's MindSLAM fusion SLAM stack and emits point clouds,
odometry and relocalisation results over USB.

No ambiguity found — the product name, vendor, GitHub org and the USB descriptor strings on
the attached unit all agree (`iManufacturer = "manifold"`, `iProduct = "hawk"`).

| | |
|---|---|
| Vendor | Manifold Tech Limited |
| Product | Odin 1 (a.k.a. Odin1), internal codename "hawk" |
| Sensors | SPAD dTOF depth + RGB camera + IMU |
| Depth range | 70 m @ 90 % reflectivity, 30 m @ 10 % |
| FoV | 120° × 90° |
| Point rate | up to 700 kpts/s |
| Pose | ±5 cm + 1 %, odometry up to 400 Hz |
| Physical | 100 × 62 × 43 mm, ~280 g, 12–24 V, IP66 |
| Host link | **USB 3.0, vendor-specific class** — `2207:0019` (0x2207 is Rockchip's VID) |

Sources:

- Product wiki: <https://manifoldtechltd.github.io/wiki/odin_series/odin1/>
- Ubuntu quick start: <https://manifoldtechltd.github.io/wiki/odin_series/odin1/7.%20Odin1%20Quick%20start%20for%20ubuntu.html>
- Data output / topics: <https://manifoldtechltd.github.io/wiki/odin_series/odin1/5.%20Data%20output_.html>
- SDK + ROS driver: <https://github.com/manifoldsdk/odin_ros_driver>
- Navigation stack: <https://github.com/ManifoldTechLtd/Odin-Nav-Stack>
- Reference integration (depth + odom from one Odin1): <https://github.com/ManifoldTechLtd/SRU-Odin>
- Press: <https://kr-asia.com/memory-for-machines-manifold-techs-odin-1-brings-spatial-recall-to-robotics>
- Resellers: [RobotShop](https://www.robotshop.com/products/manifold-manifold-odin-1-spatial-memory-module), [Foxtech](https://store.foxtech.com/odin1-spatial-memory-module-for-robotics-uavs/)

### Confirmed present on this machine

```
Bus 004 Device 004: ID 2207:0019 Fuzhou Rockchip Electronics Company hawk
/sys/bus/usb/devices/4-1.1  manufacturer=manifold  product=hawk  serial=884531645a142e70
```

Single vendor-specific interface (class 255). It exposes **no UVC video node and no USB
network interface** — all streaming necessarily goes through the vendor's libusb host
library. There is no way to get frames off this device without that library.

---

## 2. Install route

### What the vendor actually ships

There is **no official Python SDK, no .deb, no wheel, and no registration wall**. Everything
is a public GitHub clone, Apache-2.0 licensed. What exists is:

1. `manifoldsdk/odin_ros_driver` — a ROS 1 / ROS 2 C++ driver, and
2. inside that repo, `lib/liblydHostApi_amd.a` (and `_arm.a`) — the **actual host SDK**, a
   prebuilt static archive exposing a clean C API declared in `include/lidar_api.h` and
   `include/lidar_api_type.h`.

Officially supported combos are Ubuntu 20.04 (Noetic / Foxy) and 22.04 (Humble). The README
says verbatim: *"Ubuntu 24.04 is not officially supported but may work with some
modifications."*

### Route chosen: relink the host SDK into a .so and drive it from Python via ctypes

The ROS 2 route was rejected as the primary path because (a) it drags in ~1 GB of
`ros-jazzy-*` packages that are not currently installed, (b) `rclpy` lives in the system
Python and cannot be used from a `uv` venv cleanly, and (c) it is a strictly longer path to
the same bytes — the ROS nodes are themselves just consumers of `liblydHostApi`.

Instead: the static archive is relinked into a shared object, and a ctypes binding package
`odin1` calls the vendor C API directly. **No ROS in the loop.** Depth, colour, IMU,
SLAM cloud and odometry are all reachable this way, because they are all just
`lidar_data_type_e` values on one callback.

### Layout

```
/home/yunfan/opt/odin1/
├── odin_ros_driver/          vendor SDK clone, v0.14.0 (6f993cc, 2026-08-09)
│   ├── include/lidar_api.h   the C API
│   └── lib/liblydHostApi_amd.a
├── build/libodin1_host.so    relinked shared object  <- ctypes loads this
├── python/                   the `odin1` binding package (editable install)
└── 99-odin-usb.rules         staged udev rule, NOT yet installed (needs root)

/home/yunfan/Project/PiperPush/LivingTwin/hardware/depth_bench/.venv-odin1/
    Python 3.12.3 venv: numpy 2.5.2, opencv-python-headless 5.0.0, odin1 0.1.0 (-e)
```

Nothing was installed into conda `base`. No apt package was installed, removed or
downgraded — every build dependency was already present. `librealsense2` was not touched.

---

## 3. Exact commands run

```bash
# 1. Fetch the vendor SDK (public, no login)
mkdir -p /home/yunfan/opt/odin1 && cd /home/yunfan/opt/odin1
git clone https://github.com/manifoldsdk/odin_ros_driver.git      # -> v0.14.0

# 2. Relink the static host SDK into a shared object.
#    --whole-archive is required: without it the linker drops every object
#    that nothing in the (empty) link set references.
mkdir -p /home/yunfan/opt/odin1/build && cd /home/yunfan/opt/odin1/build
g++-13 -shared -fPIC -o libodin1_host.so \
  -Wl,--whole-archive /home/yunfan/opt/odin1/odin_ros_driver/lib/liblydHostApi_amd.a \
  -Wl,--no-whole-archive \
  -lusb-1.0 -lssl -lcrypto -lpthread

# 3. Dedicated venv (conda base untouched)
cd /home/yunfan/Project/PiperPush/LivingTwin/hardware/depth_bench
uv venv --python 3.12 .venv-odin1

# 4. Install the ctypes bindings
VIRTUAL_ENV=$PWD/.venv-odin1 uv pip install -e /home/yunfan/opt/odin1/python
VIRTUAL_ENV=$PWD/.venv-odin1 uv pip install opencv-python-headless

# 5. Smoke test
.venv-odin1/bin/python -m odin1.probe
```

### System dependencies

**None were installed.** All of the vendor's listed build deps were already present:

| package | version already installed |
|---|---|
| `libusb-1.0-0-dev` | 2:1.0.27-1 |
| `libyaml-cpp-dev` | 0.8.0+dfsg-6build1 |
| `libopencv-dev` | 4.6.0+dfsg-13.1ubuntu1 |
| `libeigen3-dev` | 3.4.0-4build0.1 |
| `libssl-dev` | 3.0.13-0ubuntu3.12 |
| `libpcl-dev` | 1.14.0+dfsg-1 |
| `build-essential`, `cmake` | 12.10ubuntu1, 3.28.3 |

(`libssl`/`libcrypto` turned out not to be needed at all — the archive's MD5 is
self-contained. The final `.so` links only `libusb-1.0`, `libstdc++`, `libm`, `libgcc_s`,
`libc`.)

---

## 4. What succeeded

- Shared object builds cleanly from the vendor archive with `g++-13` on noble; the
  archive's objects are already position-independent.
- `ctypes.CDLL` loads it; **26/26** declared `lidar_*` symbols resolve
  (36 `lidar_*` symbols are exported in total).
- **ABI verified against the vendor headers.** A C++ program compiled against
  `lidar_api_type.h` was used to check every struct size and field offset the bindings
  depend on; all match the ctypes layout exactly:
  `lidar_device_info_t`=136, `buffer_List_t`=48, `capture_Image_List_t`=200,
  `lidar_data_t`=208, `imu_convert_data_t`=40, `slam_cloud_point_t`=28,
  `lidar_calibration_t`=100, `ros_odom_convert_complete_t`=688.
- `lidar_system_init()` succeeds, spawns the SDK's publisher/IMU/odom threads and the
  libusb hotplug monitor.
- Frame decoders exercised end-to-end against synthetic SDK buffers (real
  `capture_Image_List_t` structs pointing at numpy memory): DTOF → depth/xyz/confidence/
  intensity round-trips exactly, `.z` and `.range` agree with the XYZ channel, JPEG and
  NV12 colour both decode to BGR, and `project_to_color` maps the optical centre to
  `(cx, cy)` and returns NaN for points behind the camera.
- **The SDK enumerates this exact device.** It hard-codes `2207:0019` and logs
  `registerd hotplug_callback for device 2207:0019` then
  `find usb device 2207:0019 attach.` — so the unit on this machine is recognised, is
  powered, and its SoC has booted far enough to enumerate. It is *not* in maskrom mode.

### One bug worked around

`lidar_system_deinit()` never returns when the SDK detected a device but failed to open it
— the libusb event-monitor thread stays parked and the process has to be `SIGKILL`ed.
`Odin1.shutdown()` therefore runs `lidar_system_deinit` on a daemon thread with a 5 s
timeout and sets `dev.deinit_hung = True` instead of wedging the interpreter. This is a
vendor-side defect, not a binding issue.

---

## 5. BLOCKED — one thing, needs root

```
[ERROR][lib_usb.cpp:open_device:749]: cannot open device 2207:0019: LIBUSB_ERROR_ACCESS
[ERROR][lib_usb.cpp:event_monitor_routine:196]: detected device 2207:0019 attach, but open fail.
```

The device node is `crw-rw-r-- root root /dev/bus/usb/004/004`. `yunfan` has read but not
write access, and libusb needs write to claim the interface. The vendor's udev rule is
**not** shipped in the repo (I checked — there is no `.rules` file); it only appears as
copy-paste text in README §3.1 and §5.12.

`sudo` on this machine requires a password and there is no NOPASSWD entry, so I could not
install it.

### What the user must run (one time, ~5 seconds)

The rule file is already written and staged at
`/home/yunfan/opt/odin1/99-odin-usb.rules`. Install it with:

```bash
sudo install -m 0644 /home/yunfan/opt/odin1/99-odin-usb.rules \
     /etc/udev/rules.d/99-odin-usb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
```

Contents (matches the vendor README exactly, and matches **only** `2207:0019` — no other
device on this bus is affected, RealSense included):

```
SUBSYSTEM=="usb", ATTR{idVendor}=="2207", ATTR{idProduct}=="0019", MODE="0666", GROUP="plugdev"
```

If `udevadm trigger` does not re-apply to the already-attached device, unplug and replug it.
`yunfan` is already in `plugdev`, so no `usermod` or re-login is needed.

Then re-run, and this should reach the device:

```bash
cd /home/yunfan/Project/PiperPush/LivingTwin/hardware/depth_bench
.venv-odin1/bin/python -m odin1.probe --stream
```

**Nothing else is blocked.** No login, no licence acceptance, no registration, no firmware
flash. (No firmware was flashed and the device was never put into maskrom/loader mode.)

---

## 6. Opening a depth stream in Python

### Vendor call sequence (`lidar_api.h`)

```
lidar_system_init(device_cb)          register the hotplug callback
  -> device_cb fires with attach=true, carrying lidar_device_info_t
lidar_create_device(&info, &handle)
lidar_register_stream_callback(handle, {data_cb, user_data})
lidar_open_device(handle)
lidar_set_mode(handle, LIDAR_MODE_RAW | LIDAR_MODE_SLAM)
lidar_activate_stream_type(handle, LIDAR_DT_*)   configure
lidar_start_stream(handle, LIDAR_DT_*, &subframe_odr)   transmit
  -> data_cb fires per frame with lidar_data_t
lidar_stop_stream / lidar_close_device / lidar_destory_device / lidar_system_deinit
```

Stream types: `LIDAR_DT_RAW_RGB`(1) `LIDAR_DT_RAW_IMU`(2) `LIDAR_DT_RAW_DTOF`(3) in RAW
mode; SLAM mode adds `LIDAR_DT_SLAM_CLOUD`(4), `LIDAR_DT_SLAM_ODOMETRY`(5),
`LIDAR_DT_SLAM_ODOMETRY_HIGHFREQ`(7), `LIDAR_DT_SLAM_ODOMETRY_TF`(8).

### Continuous streaming, depth + colour

```python
from odin1 import Odin1, project_to_color, LIDAR_DT_RAW_DTOF, LIDAR_DT_RAW_RGB

with Odin1() as dev:                       # lidar_system_init
    info = dev.wait_for_device(timeout=10) # blocks for the attach callback
    print(info.serial, info.model, info.online)

    dev.open()                             # register cb + lidar_open_device
    dev.set_mode("raw")                    # or "slam"
    dev.set_depth_rate(0)                  # 0=10Hz, 1=14.5Hz, 2=29Hz
    dev.start_stream(LIDAR_DT_RAW_DTOF)
    dev.start_stream(LIDAR_DT_RAW_RGB)

    calib = dev.calibration()              # lidar_get_calibration
    print(calib.fx, calib.fy, calib.cx, calib.cy)
    print(calib.extrinsics)                # 4x4 Tcl, camera <- lidar

    for depth, color in dev.frames():      # blocks, paced by the depth stream
        z    = depth.z                     # float32 (192,256) metres, Z along optical axis
        pts  = depth.xyz                   # float32 (192,256,3) metres, lidar frame
        rng  = depth.range                 # float32 (192,256) metres, radial
        conf = depth.confidence            # uint8  (192,256)
        inten= depth.intensity             # uint16 (192,256)
        if color is not None:
            bgr = color.image              # uint8 (H,W,3) BGR, decoded lazily
        ...
# leaving the with-block stops streams, closes, destroys, deinits
```

One-shot equivalent: `depth, color = dev.read_frame(timeout=5.0)`.
Depth-only: `depth = dev.read_depth()`. Latest colour without blocking:
`dev.latest_color()`.

### Depth frame facts

| question | answer |
|---|---|
| resolution | **256 × 192** (`DTOF_WIDTH` × `DTOF_HEIGHT`), fixed |
| dtype | `float32` |
| units | **metres**, already scaled — no scale factor to apply |
| geometry | The authoritative channel is **XYZ**, `float32 (192,256,3)`, metres, right-handed lidar frame, **+Z forward along the optical axis**. `frame.z` is `xyz[...,2]` (Z-depth); `frame.range` is `‖xyz‖` (radial). |
| invalid pixels | gate on `confidence`; the vendor recommends a threshold of **30–35** (typical scene range 0…~1300, higher = more reliable). Note the raw DTOF `confidence` channel is `uint8` while the ROS `cloud_raw` message widens it to `uint16`. |
| rate | 10 / 14.5 / 29 Hz, selected by `set_depth_rate()` |

⚠️ **One caveat to resolve on first hardware run.** The SDK also hands back a separate
`float32` depth channel (`imageList[0]`, exposed as `frame.depth`) which the header
documents only as "depth in meters". The vendor's own ROS driver **never reads it** — it
builds everything from the XYZ channel — so whether it is Z-depth or radial range is not
established by the code. Settle it in one line once the udev rule is in:

```python
import numpy as np
d = dev.read_depth()
print("vs Z    :", np.nanmax(np.abs(d.depth - d.z)))
print("vs range:", np.nanmax(np.abs(d.depth - d.range)))   # whichever is ~0 wins
```

Until then, **use `frame.z` / `frame.xyz`**, which are unambiguous.

### Intrinsics

- `dev.calibration()` → `lidar_get_calibration`, giving a 3×3 K and a 4×4 camera←lidar
  extrinsic `Tcl`, as float32 numpy arrays. `calib.fx/.fy/.cx/.cy` are convenience
  accessors.
- **These are the colour camera's intrinsics.** The dTOF has *no* pinhole intrinsics — it
  is a lidar and returns XYZ directly, so there is no (fx, fy, cx, cy) for the depth grid
  and none is needed.
- For full accuracy the colour camera is **not** a plain pinhole. It uses the vendor's
  `PolynomialCamera` model (`include/polynomial_camera.hpp`): an affine intrinsic
  `A11, A12 (skew), A22, u0, v0` plus a 6-term equidistant/fisheye polynomial
  `k2 … k7` (and `p1, p2`). Distortion is treated as active when `|k2| > 1e-7`.
  `lidar_get_calibration` returns only K, so to get `k2…k7` dump the full calibration file:

  ```python
  dev.save_calibration("/home/yunfan/opt/odin1")   # writes calib.yaml there
  ```

  `calib.yaml` keys: `cam_num`, `img_topic_0`, `Tcl_0` (16 doubles, row-major),
  and under `cam_0`: `cam_model`, `image_width`, `image_height`,
  `A11 A12 A22 u0 v0`, `k2 k3 k4 k5 k6 k7 p1 p2`, `isFast`, `numDiff`, `maxIncidentAngle`.

### Colour ↔ depth registration: **you must do it yourself**

There is no hardware-registered pair. The two streams differ in resolution
(1536×1280 vs 256×192), rate and optical centre, and the device does not align them.
The ROS driver's `odin1/depth_img_competetion` topic — the one that *is* one-to-one with
`odin1/image_undistort` — is produced **on the host** by
`PointCloudToDepthConverter` (`src/pointcloud_depth_converter.cpp`), which transforms the
cloud by `Tcl` and rasterises `camera_point.z` into a float32 image. The README flags it as
a demo requiring "high computing power". So it is a host-side software step either way.

`odin1.project_to_color()` reproduces the pinhole part of that:

```python
u, v, z = project_to_color(depth.xyz, calib)   # NaN where the point is behind the camera
```

For edge-accurate projection on a 120°×90° FoV you will want the `k2…k7` terms from
`calib.yaml` as well — the helper deliberately uses K only and says so in its docstring.

Colour frames arrive as **JPEG** on firmware ≥ 0.13 and as **NV12 1536×1280** on older
firmware. The binding detects which by the vendor's own test
(`length == width*height*3/2` ⇒ NV12) and decodes either into BGR on access to
`color.image`.

### Memory-safety note

The SDK hands out pointers into a ring buffer that is recycled the instant the callback
returns. Every array the bindings expose is `.copy()`d inside the callback. If you extend
`_capi.py` / `__init__.py`, preserve that — and keep a Python reference alive to any
`CFUNCTYPE` trampoline you hand to the SDK, since it is invoked from the SDK's own USB
thread long after the registering call returns.

---

## 7. If you want the ROS 2 route instead

Viable but not free. `/opt/ros/jazzy` here is a **minimal** install (130 packages) missing
what the driver needs. It would take:

```bash
sudo apt-get install ros-jazzy-sensor-msgs ros-jazzy-cv-bridge \
                     ros-jazzy-pcl-conversions ros-jazzy-image-transport ros-jazzy-rviz2
mkdir -p ~/odin_ws/src && cp -r /home/yunfan/opt/odin1/odin_ros_driver ~/odin_ws/src/
cd ~/odin_ws/src/odin_ros_driver/script && ./build_ros2.sh
source ~/odin_ws/install/setup.bash
ros2 launch odin_ros_driver odin1_ros2.launch.py
```

Caveats: jazzy is not on the vendor's supported list (Humble is); the CMake distro regex in
`CMakeLists.txt` does not even name `jazzy`, though it falls through to ROS2 by default and
`build_ros2.sh` does probe for jazzy. `build_ros2.sh` also has a latent bug — it runs
`cd $WS_DIR` with `WS_DIR` unset. Topics you would then get include `/odin1/cloud_raw`,
`/odin1/image`, `/odin1/image_undistort`, `/odin1/imu`, `/odin1/odometry`,
`/odin1/depth_img_competetion`.

The same udev rule is required for this route too — it is the identical libusb open.

---

## 8. Quick reference

```bash
# activate
source /home/yunfan/Project/PiperPush/LivingTwin/hardware/depth_bench/.venv-odin1/bin/activate

# probe (works with no hardware; --stream pulls a real frame once udev is fixed)
python -m odin1.probe --wait 5
python -m odin1.probe --stream --verbose-sdk

# silence the SDK's stderr chatter
python -m odin1.probe 2>/dev/null
```

Override the library path with `$ODIN1_HOST_LIB` if you relocate it.

To rebuild the `.so` after a vendor SDK update:

```bash
cd /home/yunfan/opt/odin1/odin_ros_driver && git pull
cd /home/yunfan/opt/odin1/build && g++-13 -shared -fPIC -o libodin1_host.so \
  -Wl,--whole-archive ../odin_ros_driver/lib/liblydHostApi_amd.a \
  -Wl,--no-whole-archive -lusb-1.0 -lpthread
```

`odin1.probe` reports `symbols bound N/M`; if a vendor update renames anything, that
count drops and names the missing symbols.
