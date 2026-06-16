#!/bin/bash
for arg in "$@"; do
  if [[ "$arg" == "-h" || "$arg" == "--help" ]]; then
    echo "Description: Runs the compiled test binary to verify the correctness of your kernel"
    echo "             against a reference implementation, and then runs performance benchmarks."
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    "$(dirname "$0")/run_test" -h | grep -v '^Usage:' | grep -v '^Options:'
    exit 0
  fi
done

exec "$(dirname "$0")/run_test" "$@"
