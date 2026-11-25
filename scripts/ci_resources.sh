#!/usr/bin/env bash
# Record what a CI runner has and how close a job comes to running out of it.
#
# A runner that runs out of memory or disk dies with no log, and GitHub reports only that it
# lost communication with the runner. So the job prints the runner's resources before the tests
# (`describe`), samples memory, swap and disk once a minute in the background (`start`), and
# prints the lowest headroom it saw when the job ends (`report`, run with `if: always()`). A job
# that survives shows how close it came; a shard whose margin shrinks shows it before it dies.
set -u
log="${RUNNER_TEMP:-/tmp}/cleverly-resources.log"

case "${1:-}" in
  describe)
    echo "image: ${ImageOS:-unknown} ${ImageVersion:-unknown}"
    echo "cores: $(nproc)"
    free -m
    swapon --show || true
    findmnt --target /tmp || true
    df -h / /tmp "${RUNNER_TEMP:-/tmp}" || true
    ;;
  start)
    : > "$log"
    nohup bash -c '
      while true; do
        avail=$(awk "/^MemAvailable:/ {print int(\$2 / 1024)}" /proc/meminfo)
        swap=$(awk "/^SwapTotal:/ {t=\$2} /^SwapFree:/ {f=\$2} END {print int((t - f) / 1024)}" /proc/meminfo)
        disk=$(df -m --output=avail / | tail -1 | tr -d " ")
        echo "$(date -u +%H:%M:%S) $avail $swap $disk"
        sleep 60
      done' >> "$log" 2>/dev/null &
    ;;
  report)
    if [ ! -s "$log" ]; then
      echo "no samples were recorded"
      exit 0
    fi
    echo "samples: $(wc -l < "$log"), one a minute"
    sort -n -k2 "$log" | head -1 | awk '{print "least memory available: " $2 " MiB at " $1}'
    sort -n -k3 "$log" | tail -1 | awk '{print "most swap in use: " $3 " MiB at " $1}'
    sort -n -k4 "$log" | head -1 | awk '{print "least disk free on /: " $4 " MiB at " $1}'
    ;;
  *)
    echo "usage: $0 describe|start|report" >&2
    exit 2
    ;;
esac
