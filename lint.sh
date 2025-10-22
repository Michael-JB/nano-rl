#!/usr/bin/env bash

set -euo pipefail

usage() {
  echo "Usage: $0 [--fix]"
  echo "  --fix    Apply auto-fixes to linting and formatting issues"
  exit 1
}

# Change directory to repo root
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

# Parse arguments
FIX=false
while [ $# -gt 0 ]; do
  case "$1" in
    --fix)
      FIX=true
      shift
      ;;
    *)
      usage
      ;;
  esac
done

CHECK_FLAG=$( [ "$FIX" = true ] && echo "--fix" || echo "" )
FORMAT_FLAG=$( [ "$FIX" = true ] && echo "" || echo "--diff" )

echo "Linting..."
uv run ruff check . $CHECK_FLAG

echo "Formatting..."
uv run ruff format $FORMAT_FLAG .

echo "Checking types..."
uv run mypy .

echo "All checks passed!"
