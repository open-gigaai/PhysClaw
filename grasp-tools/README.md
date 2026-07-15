# grasp-tools

Capture → SAM3 → AnyGrasp → joint grasp/place, with optional HDF5 recording and
the post-collect stats platform under `collect/`.

Configure **your** arm/camera in `../configs/paths.env`. The reference lab used
Piper + RealSense; that is not a hard requirement.

**AnyGrasp proprietary runtime is not included.** See [ANYGRASP_RUNTIME.md](ANYGRASP_RUNTIME.md).

## Environments

| Env | Role |
|-----|------|
| `physclaw-0` (`CONDA_ENV`) | Capture, IK/motion, collect stats/video |
| Official SAM3 env (`CONDA_ENV_SAM3`) | `run_seg.py` / seg server |
| Official AnyGrasp env (`CONDA_ENV_ANYGRASP`) | `run_grasp*.py` |

## Main entry

```bash
source ../configs/paths.env
bash run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down \
  --task "banana" left 0.3 -0.27 0

bash run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down --collect \
  --task "banana" left 0.3 -0.27 0
```

## Dataset tooling

```bash
python collect/episode_stats.py --dataset-dir /path/to/data --all
python collect/data_quality_check.py --dataset-dir /path/to/data
python collect/serve_report.py --dataset-dir /path/to/data
```
