#!/usr/bin/env bash
# Run both verification spikes; exit non-zero if either fails a hard claim.
set -u
cd "$(dirname "$0")"

rc=0
echo "=== layer smoke check ==="
python variational_dense.py || rc=1
echo
echo "=== spike 1: KL / add_loss propagation ==="
python spike_kl_propagation.py || rc=1
echo
echo "=== spike 2: serialization round-trip ==="
python spike_serialization.py || rc=1
echo
if [ "$rc" -eq 0 ]; then
  echo "ALL SPIKES PASSED"
else
  echo "ONE OR MORE SPIKES FAILED"
fi
exit "$rc"
