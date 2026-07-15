# AnyGrasp runtime (not included)

Install the official AnyGrasp SDK and place (or symlink) into this directory, or set:

- `GSNET_DIR` — directory containing `gsnet.so` / Python bindings
- `ANYGRASP_CHECKPOINT` — path to detection checkpoint

Typical local layout (legacy):

```text
grasp-tools/
  gsnet.so
  lib_cxx.so
  license/*.lic
  log/checkpoint_detection.tar
```

These files are gitignored and must not be redistributed with PhysClaw-0.
