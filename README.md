# PhysGnosis

Agentic physical grasp and self-collection for dual-arm robots:
**SAM3 text prompt → AnyGrasp → joint grasp/place**, orchestrated by OpenClaw-style
skills with an analyze → collect → reset loop, plus a post-collect stats platform.

This repository packages first-party pipelines, motion helpers, dataset
stats/quality tools, and four agent skills. It does **not** redistribute the
proprietary AnyGrasp runtime (`.so`, license, checkpoint).

> **Hardware note:** The reference lab used Agilex Piper arms and Intel RealSense
> cameras. PhysGnosis does **not** require that stack — configure your own arm
> driver, camera launch, ROS topics, intrinsics, and extrinsics in
> `configs/paths.env`.

## Layout

```text
PhysGnosis/
├── skills/
├── grasp-tools/            # Recorder pipeline + collect/ stats
├── configs/                # paths.env.example, camera_extrinsics.example.json
└── docs/
```

## Quick start

1. Copy and edit config:

```bash
cp configs/paths.env.example configs/paths.env
# Set CONDA_ENV=physgnosis, CONDA_ENV_SAM3 / CONDA_ENV_ANYGRASP (official envs),
# ARM_* / CAMERA_* / CAM_FX.. / topics / extrinsics for YOUR robot.
```

2. Environments (see [docs/setup.md](docs/setup.md)):
   - **One** project conda env: `physgnosis` (capture, IK/motion, collect tooling)
   - Install **SAM3** and **AnyGrasp** with their official docs; set `CONDA_ENV_SAM3` / `CONDA_ENV_ANYGRASP` to those env names

3. Grasp + place:

```bash
source configs/paths.env
bash grasp-tools/run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down \
  --task "orange" right 0.3 0.27 0
```

4. Skills: point your agent at `skills/` (`self-learning` → `1-analyze-task`).

## Skills

| Skill | Role |
|-------|------|
| `self-learning` | Thin orchestrator; holds `analyze_result.yaml` |
| `1-analyze-task` | Plan collect/reset, then run the loop with VLM judges |
| `grasp-tool` | Calls `run_multi_pipeline_recorder.sh` |
| `understand-three-view-images` | Capture RGB and VLM-describe / judge success |

## Dependencies (summary)

| Component | Notes |
|-----------|--------|
| Conda `physgnosis` | Project code: capture, motion, collect stats |
| Official SAM3 env | Set `CONDA_ENV_SAM3` after following SAM3 install docs |
| Official AnyGrasp env | Set `CONDA_ENV_ANYGRASP` after following AnyGrasp install docs |
| Your arm + camera ROS stack | Configure launches, topics, calibration |
| `COLLECT_RECORDER_PY` | External HDF5 recorder |

## License

Apache License 2.0 — see [LICENSE](LICENSE). Third-party runtimes are **not**
shipped; see [NOTICE](NOTICE) and [docs/setup.md](docs/setup.md).
