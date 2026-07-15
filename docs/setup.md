# Setup

## 1. Configure paths

```bash
cd /path/to/PhysClaw-0
cp configs/paths.env.example configs/paths.env
```

Fill `configs/paths.env` for **your** robot and cameras. The authors' reference
setup used Agilex Piper + Intel RealSense; nothing requires that hardware.

Also copy and edit extrinsics if needed:

```bash
cp configs/camera_extrinsics.example.json configs/camera_extrinsics.json
# then set CAMERA_EXTRINSICS_JSON in paths.env
```

## 2. Conda environments

### Project env (required)

Create one env for PhysClaw-0 tooling (ROS Python clients, OpenCV, h5py,
Pinocchio/CasADi for IK, etc.):

```bash
conda create -n physclaw-0 python=3.8   # version of your choice
conda activate physclaw-0
# install your stack deps: rospy bindings, numpy, opencv, h5py, pinocchio, ...
```

Set `CONDA_ENV=physclaw-0` in `paths.env`.

### SAM3 and AnyGrasp (official installs)

Do **not** fold these into `physclaw-0` unless you know the deps are compatible.
Follow each project's official environment instructions, then:

```bash
CONDA_ENV_SAM3=<name-from-sam3-docs>
CONDA_ENV_ANYGRASP=<name-from-anygrasp-docs>
SAM3_CHECKPOINT=/path/to/sam3.pt
GSNET_DIR=/path/to/anygrasp   # or place .so under grasp-tools/
ANYGRASP_CHECKPOINT=/path/to/checkpoint_detection.tar
```

## 3. Arm and camera (your hardware)

Set in `paths.env`:

| Variable | Purpose |
|----------|---------|
| `ARM_ROS_SETUP` | Optional `devel/setup.bash` for your arm workspace |
| `CAMERA_ROS_SETUP` | Optional camera workspace overlay |
| `CAMERA_LAUNCH` | `roslaunch ...` args for your cameras |
| `ARM_LAUNCH` | `roslaunch ...` args for your arms (infra helper) |
| `ARM_IK_DIR` / `ARM_URDF` / `ARM_ROS_SRC` | Pinocchio IK adapter paths |
| `TOPIC_*` | Color/depth/joint topic names |
| `CAM_FX` `CAM_FY` `CAM_CX` `CAM_CY` | Front-camera intrinsics (pixels) |
| `DEPTH_SCALE` | Depth units → mm scale (often `1000`) |
| `CAMERA_EXTRINSICS_JSON` | cam-to-base + TCP→flange |

Reference topic names in the example file match a Piper/RealSense-style naming
(`/camera_f/...`, `/puppet/joint_*`); change them to match your drivers.

## 4. HDF5 recorder (optional collect)

```bash
export COLLECT_RECORDER_PY=/path/to/your_recorder.py
```

## 5. VLM API (three-view / judges)

```bash
export ARK_API_KEY=...
```

## 6. Smoke check

```bash
source configs/paths.env
bash grasp-tools/auto_capture_rgb_depth.sh
bash grasp-tools/run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down \
  --task "banana" left 0.3 -0.27 0
```
