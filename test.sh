#!/usr/bin/env bash

set -euo pipefail

# Change directory to repo root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

uv run pytest .
