---
name: understand-three-view-images
description: Skill for understanding RGB images from three cameras. Trigger when the user mentions "understand images" or "understand the environment".
---

# understand-three-view-images

This skill understands images from the robot head, left-arm, and right-arm cameras. It auto-captures view images, runs a multimodal model, and returns an environment understanding result.

Before running, set `ARK_API_KEY` (or `VOLCENGINE_API_KEY`) and `source $PHYSCLAW_0_ROOT/configs/paths.env`.

## Capabilities
- Understand three-view / front-view image content and return current environment understanding
- BBox prompt variant: `scripts/auto_understand_images_bbox.sh`

## Workflow
Use `scripts/auto_understand_images.sh --prompt "Please describe the current desktop scene"` to get environment understanding. The script auto-captures images and calls the understanding script.

## File layout
```
understand-three-view-images/
├── SKILL.md
└── scripts/
     ├── auto_understand_images.sh
     ├── auto_understand_images_bbox.sh
     ├── capture_and_understand_one_view.py
     └── ...
```

## Resources
- `scripts/auto_understand_images.sh`: capture and understand; requires `--prompt`
