---
name: grasp-tool
description: Use when the user mentions "grasp", "pick up", or "pick and place". Uses SAM3 text segmentation + AnyGrasp planning to drive a robot arm for grasping, optionally place at a given (x,y); supports --collect to record HDF5.
---

# grasp-tool

This skill runs grasp-and-place (optional data collection) via `grasp-tools/run_multi_pipeline_recorder.sh` in the PhysGnosis repo.

Set `$PHYSGNOSIS_ROOT` to this repository root (or `source configs/paths.env` first).

## Workflow

Each `--task` is one "grasp + place" sub-task, executed in order. Use one for a single object, multiple for several objects.

```bash
bash $PHYSGNOSIS_ROOT/grasp-tools/run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down \
  --task "<text_prompt>" <arm> <place_x> <place_y> <approach_depth_offset> \
  [--task "<text_prompt>" <arm> <place_x> <place_y> <approach_depth_offset> ...]
```

- `<text_prompt>`: SAM3 text prompt (e.g. `"banana"`, `"orange"`)
- `<arm>`: `left` or `right`
- `<place_x> <place_y>`: place coordinates in the **current arm base frame**, meters (use the table below; do not invent values)
- `<approach_depth_offset>`: extra deepen along the grasp approach direction for this object (meters), **required per task**; usually start with `0`

Examples:

```bash
# Single object
bash $PHYSGNOSIS_ROOT/grasp-tools/run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down \
  --task "orange" right 0.3 0.27 0

# Multiple objects grasped in sequence
bash $PHYSGNOSIS_ROOT/grasp-tools/run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down \
  --task "orange" right 0.3 0.27 0 \
  --task "banana" left 0.3 -0.27 0

# Collect HDF5 (record arm-motion segments only)
bash $PHYSGNOSIS_ROOT/grasp-tools/run_multi_pipeline_recorder.sh --mode diligent --no-gui --top-down --collect \
  --task "banana" left 0.3 -0.27 0
```

### Fixed place coordinates (look up; do not invent)

| Scenario | arm | place-x | place-y |
|----------|-----|---------|---------|
| Place into center basket | left | 0.3 | -0.27 |
| Take out of basket | left | 0.34 | 0 |
| Place into center basket | right | 0.3 | 0.27 |
| Take out of basket | right | 0.34 | 0 |

### Parameters

| Parameter | Description |
|-----------|-------------|
| `--mode` | `diligent`: re-capture RGB-D for each task; `lazy`: capture once and reuse the same frame for multiple tasks (required) |
| `--task` | Quintuple: text prompt, arm, place x, place y, approach depth offset; may repeat |
| `--collect` | Record cameras + joints to HDF5 during arm motion |
| `--no-gui` | Required for agent calls; disables the AnyGrasp visualization window |
| `--top-down` | Top-down grasp; recommended for tabletop objects |

### Arm selection

- Object on the left side of the table → `left`
- Object on the right side of the table → `right`

## Common issues

- On any error, do not fix it yourself; ask the user and share the error message.
- Do not create or modify files arbitrarily.
- SAM3 / AnyGrasp failures: check the conda env and `docs/setup.md`, `grasp-tools/ANYGRASP_RUNTIME.md`.
