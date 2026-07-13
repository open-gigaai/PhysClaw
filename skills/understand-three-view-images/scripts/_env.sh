#!/usr/bin/env bash
# Shared bootstrap for understand-three-view-images scripts.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
_PHYSGNOSIS_CALLER="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
# shellcheck disable=SC1091
source "${SKILL_DIR}/../../configs/load_paths.sh"
TMP_DIR="${THREE_VIEW_TMP_DIR:-${SKILL_DIR}/tmp}"
mkdir -p "${TMP_DIR}"
export THREE_VIEW_TMP_IMAGE="${THREE_VIEW_TMP_IMAGE:-${TMP_DIR}/front_preview.jpg}"
