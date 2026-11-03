#!/usr/bin/env bash
# Train the flat-ground Go2 task using the shared mjlab virtual environment.
# Extra arguments are forwarded to the Python launcher to override its defaults.
# Set SUPERMIMIC_PYTHON to use a different environment's Python interpreter.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "${script_dir}/.." && pwd)"
train_python="${SUPERMIMIC_PYTHON:-${project_dir}/../.venvs/mjlab-1.6/bin/python}"

if [[ ! -x "${train_python}" ]]; then
  printf 'Python interpreter not found or not executable: %s\n' "${train_python}" >&2
  printf 'Set SUPERMIMIC_PYTHON to the Python path of your training environment.\n' >&2
  exit 1
fi

cd -- "${project_dir}"
export PYTHONUNBUFFERED=1

exec "${train_python}" "${script_dir}/train_flat.py" "$@"
