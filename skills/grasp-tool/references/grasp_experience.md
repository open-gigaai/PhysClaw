# Grasp experience (read before agent calls)

Only verified rules; read in order.

1. **Always pass `--no-gui --top-down`** in the command.

2. **Arm selection**: object on the left of the table/frame → `left`; on the right → `right`. If unsure, call `understand-three-view-images` first; do not guess.

4. **Place coordinates**: use only the fixed values in the SKILL.md table. Every `--task` must include `<place_x> <place_y>`; do not invent coordinates.

5. **Distant objects often grasp too shallow**: set the 5th `--task` parameter `approach_depth_offset` to `0.01`–`0.03` for that object (per-object; do not use a global flag).

6. **SAM3 segmentation fails (empty mask)**: retry with a `text_prompt` that better matches the frame; do not keep retrying the same prompt.

7. **On script error**: send the full error to the user; do not change code, calibration, or place coordinates yourself.
