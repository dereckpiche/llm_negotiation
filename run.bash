#!/usr/bin/env bash

# Minimal retry wrapper for running run.py with usual arguments.
# - Infinite immediate retries on non-zero exit (no sleep, no backoff, no envs)

set -u

handle_term() { exit 143; }
trap handle_term SIGINT SIGTERM

while true; do
  python -u run.py "$@"
  status=$?

  if [ ${status} -eq 0 ]; then
    exit 0
  fi
done
