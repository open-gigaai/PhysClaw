---
name: 1-analyze-task
description: First analyze the task and generate a collect/reset plan, then loop collect + reset until done. Trigger when the user mentions "analyze task", "step breakdown", or "learn a task".
---

# 1-analyze-task

This skill turns a natural-language request into a two-phase workflow: **Phase A (task analysis and plan generation)** and **Phase B (automated collect and environment reset loop)**. Execute strictly in order.

Set `$PHYSCLAW_0_ROOT` to this repository root. Three-view scripts and analysis results live under `skills/` in this repo.

## Workflow

### Phase A: Task Analysis and Plan Generation
1. **Scene perception**: Call `$PHYSCLAW_0_ROOT/skills/understand-three-view-images/scripts/auto_understand_images.sh --prompt "Briefly describe the current desktop scene, including which objects are present, their positions, and their states"` to extract current scene and object information.
2. **Confirm goals**: Combine the task description with available skills (e.g. `grasp-tool`) to define the action goals and skill invocation parameters needed to complete the task.
3. **Plan design**:
   - **Collection plan**: Design the full action chain for one collection round and the Shell commands (`collection_command`).
   - **Collection success criteria**: Define clear success conditions and a judging prompt.
   - **Reset plan**: Design the action chain that restores the environment to the pre-collection state and the Shell commands (`reset_command`).
   - **Reset success criteria**: Define clear success conditions and a judging prompt.
4. **Save and confirm (gate)**: Write the analysis strictly in the [Analysis Result Format] to the analysis result path. When done, **must** report the plan to the user so they can confirm and fine-tune it, and ask whether to continue. **Do not enter Phase B until the user explicitly agrees.**

### Phase B: Automated Collect and Reset Loop
> *Prerequisite: Phase A is complete and the user has explicitly confirmed.*

1. **Read the plan**: Read `analyze_result.yaml` for all commands, `prompt_for_judging_collect`, `prompt_for_judging_reset`, and the total number of episodes to collect (`need_collect_count`).
2. **Loop**: While `current_collect_count < need_collect_count`, run one round:
   - Collection actions: First observe to get fresh scene info, then execute Shell commands in `collection_command` one by one.
   - Judge collection: Call `$PHYSCLAW_0_ROOT/skills/understand-three-view-images/scripts/auto_understand_images.sh --prompt <prompt_for_judging_collect>`:
     - On success, proceed to reset.
     - On failure, retry from collection judging; if `collect_attempts > 1`, pause and request human help.
   - Reset actions: First observe to get fresh scene info, then execute Shell commands in `reset_command` one by one.
   - Judge reset: Call `$PHYSCLAW_0_ROOT/skills/understand-three-view-images/scripts/auto_understand_images.sh --prompt <prompt_for_judging_reset>`:
     - On success, proceed to the next round.
     - On failure, re-run reset actions; if `reset_attempts > 1`, pause and request human help.
3. In Phase B, keep user messages brief: omit non-essential info; only report the current round, collection result, reset result, and progress.

## Example
**User task**: "Clean the desktop (target: collect 50 episodes)"

**Phase A**:
1. The system finds an apple, a storage box, etc. on the table.
2. Plan: collection goal is "call grasp-tool to put the apple into the storage box"; reset goal is "call grasp-tool to take the apple out of the box and place it randomly on the table".
3. Success criteria: collection succeeds when the apple is in the box; reset succeeds when the apple is back on the table.
4. Save the plan to `analyze_result.yaml` and ask the user: "Analysis complete. Will run grasp-tool collection for ten rounds. Confirm to start?"
**Phase B** (starts after the user replies "confirm"):
1. Read the plan and start round 1 collection.
2. Re-perceive scene -> put into container -> wait -> visual judge (success).
3. Re-perceive scene -> take out and reset -> wait -> visual judge (success).
4. Auto-loop until 50 rounds finish or an exception occurs.

## Output Format

### Analysis result storage
- **Path**: `$PHYSCLAW_0_ROOT/skills/self-learning/analyze_result.yaml`
- **Example**: see `analyze_result.example.yaml` in the same directory
- **Analysis result format**:
```text
task_name: <name of the task being analyzed>
dataset_dir: <root directory for dataset storage>
need_collect_count: <total number of episodes to collect>
task_description: <understanding of the task and extra notes>
collection_plan: <natural-language collection plan, including skill call sequence and parameter design>
- step_1: <which skill to call first, with what parameters, and expected result>
collection_command: <shell commands for the collection plan>
- step_1: <first shell command, including skill invocation and parameters>
reset_plan: <natural-language reset plan>
- step_1: <which skill to call first, with what parameters, and expected result>
reset_command: <shell commands for the reset plan>
- step_1: <first shell command, including skill invocation and parameters>
collect_success_criteria: <how to judge collection success / task completion>
prompt_for_judging_collect: <prompt asking whether collection succeeded>
reset_success_criteria: <how to judge reset success / environment restored>
prompt_for_judging_reset: <prompt asking whether reset succeeded>
```

Note: Do not stop until all data has been collected.

## Handling Out-of-Distribution (OOD) cases
Any situation not covered in this document is OOD. See `skills/self-learning/reference/OOD_handling_flow.md`: pause and ask the user for help; do not change the plan or system environment on your own.

## File layout
```
$PHYSCLAW_0_ROOT/skills/self-learning/
├── analyze_result.yaml           # Phase A output / Phase B input (local, not committed)
├── analyze_result.example.yaml
└── reference/
    └── OOD_handling_flow.md
```
