---
name: self-learning
description: Robot-arm self-learning skill that orchestrates analysis, collection, reset, and related skills in order.
---

# Robot-arm self-learning skill

This skill schedules other skills for self-learning. Repository root is `$PHYSGNOSIS_ROOT`.

## Workflow
1. Read `skills/1-analyze-task/SKILL.md` in this repo, produce collection and reset plans, and execute them.
2. Write analysis results to `analyze_result.yaml` in this directory (see `analyze_result.example.yaml`).

## Attention
1. Be concise.
2. Follow the steps strictly.
3. On any error message, do not fix it yourself; ask the user for help immediately.
4. Do not modify the system ROS or conda environments.
