# OOD handling flow

Any situation not covered by `1-analyze-task` or sibling skills is OOD.

1. **Stop** the collect/reset loop immediately.
2. **Do not** invent new shell commands, edit system ROS/conda, or change the plan YAML without user approval.
3. Report the error text / unexpected observation to the user in one short message.
4. Wait for explicit human instructions before continuing.

Common OOD examples: hardware faults, empty masks, AnyGrasp license errors, VLM API failures, unexpected objects on the table.
