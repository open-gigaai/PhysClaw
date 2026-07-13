# Architecture

```text
self-learning
    └─► 1-analyze-task
            ├─► understand-three-view-images
            ├─► grasp-tool
            └─► collect/reset loop

grasp-tool
    └─► grasp-tools/run_multi_pipeline_recorder.sh
            ├─ home arms
            ├─ RGB-D capture (your camera topics)
            ├─ SAM3 text → mask  (official SAM3 conda env)
            ├─ AnyGrasp + mask → pose  (official AnyGrasp conda env)
            ├─ joint grasp/place  (physgnosis env + your IK/URDF)
            └─ optional --collect → HDF5 + collect/ stats
```

Calibration (intrinsics / extrinsics / topics) is user-supplied via
`configs/paths.env` and optional `CAMERA_EXTRINSICS_JSON`. The reference lab
used Piper + RealSense; the code paths are meant to be retargeted.

## Dataset tooling

`grasp-tools/collect/`: episode stats, quality checks, HTML report, video,
offline VLM judge, training-set prep.
