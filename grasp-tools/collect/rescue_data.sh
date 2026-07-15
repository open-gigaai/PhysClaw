#!/usr/bin/env bash
# Re-judge every session whose result.txt still has "vision: unknown", then
# refresh the dataset report once. Run from anywhere.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_PHYSCLAW_0_CALLER="${BASH_SOURCE[0]}"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/../../configs/load_paths.sh"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/.." && pwd)"
DATASET_DIR="${1:-${ROOT}/collected_data}"

shopt -s nullglob
mapfile -t RESULTS < <(grep -l "vision: unknown" "${DATASET_DIR}"/*/*/result.txt 2>/dev/null || true)
if [[ ${#RESULTS[@]} -eq 0 ]]; then
    echo "no sessions with vision: unknown under ${DATASET_DIR}"
else
    for r in "${RESULTS[@]}"; do
        python "${HERE}/judge_episode_offline.py" \
            --session-dir "$(dirname "$r")" --write --no-report
    done
fi
python "${HERE}/episode_stats.py" --dataset-dir "${DATASET_DIR}" --all
