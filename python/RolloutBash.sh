#!/usr/bin/env bash
set -euo pipefail

pids=()

cleanup() {
  echo "Stopping... pids: ${pids[*]:-}"
  kill -TERM "${pids[@]:-}" 2>/dev/null || true
  sleep 0.5
  kill -KILL "${pids[@]:-}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# ---- 1) ROS relay ----
bash -lc '
  source /choreonoid_ws/install/setup.bash
  exec python3 -u /userdir/python/relay_nodes_on_ros.py
' & pids+=($!)

# ---- 2) Rollout ACT (外部引数をそのまま渡す) ----
bash -lc "
  source /irsl_venv/bin/activate
  exec python -u /userdir/python/interfaces_on_rolloutAct.py $*
" & pids+=($!)

wait
trap - INT TERM EXIT