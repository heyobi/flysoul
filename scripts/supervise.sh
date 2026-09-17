#!/usr/bin/env bash
# Keep the live agent running unattended.
#
# run.py retries a failed env.reset and rebuilds the environment, but an unhandled
# error anywhere else ends the process, and nothing restarts it. Overnight that is the
# difference between a few hundred episodes and however many fitted before the first
# hiccup. Learned synapses are written to checkpoints/ after every episode, so a restart
# resumes rather than starting the fly over.
#
#   ./scripts/supervise.sh            # run until stopped
#   MAX_RESTARTS=0 ./scripts/supervise.sh   # unlimited (default)
#
# Stop it with:  touch ~/flysoul/STOP    (or kill the supervisor, then the agent)

set -u

ROOT="${ROOT:-$HOME/flysoul}"
PY="${PY:-$ROOT/venv/bin/python}"
LOG="${LOG:-$ROOT/flysoul_run.log}"
SUP_LOG="${SUP_LOG:-$ROOT/supervisor.log}"
STOP_FILE="$ROOT/STOP"
MAX_RESTARTS="${MAX_RESTARTS:-0}"
ARGS="${ARGS:---game --continuous --explore --port 8080 --no-dashboard}"

export DISPLAY="${DISPLAY:-:0}"
export XAUTHORITY="${XAUTHORITY:-/run/user/1000/gdm/Xauthority}"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$SUP_LOG"; }

cd "$ROOT" || exit 1
rm -f "$STOP_FILE"
log "supervisor starting: $PY run.py $ARGS"

restarts=0
while true; do
    if [ -f "$STOP_FILE" ]; then
        log "STOP file present, exiting."
        break
    fi

    start=$(date +%s)
    # shellcheck disable=SC2086
    "$PY" run.py $ARGS >> "$LOG" 2>&1
    rc=$?
    ran=$(( $(date +%s) - start ))

    # Release any key the agent was holding when it died, or the player keeps walking.
    [ -f "$ROOT/release_keys.py" ] && "$PY" "$ROOT/release_keys.py" >/dev/null 2>&1

    if [ -f "$STOP_FILE" ]; then
        log "agent exited rc=$rc after ${ran}s; STOP file present, not restarting."
        break
    fi
    if [ $rc -eq 0 ]; then
        log "agent exited cleanly after ${ran}s; not restarting."
        break
    fi
    if [ $rc -eq 2 ]; then
        # A pre-flight refused: wrong key bindings, or the live game unreachable.
        # Restarting cannot fix either, and hammering the game would only make the
        # real cause harder to find in the log.
        log "agent refused to start (rc=2): see $LOG for the reason. Not restarting."
        break
    fi

    restarts=$((restarts + 1))
    if [ "$MAX_RESTARTS" -gt 0 ] && [ "$restarts" -ge "$MAX_RESTARTS" ]; then
        log "agent died rc=$rc after ${ran}s; restart limit $MAX_RESTARTS reached."
        break
    fi

    # A process that dies immediately is usually misconfigured rather than unlucky, so
    # back off instead of hammering the game with restarts.
    if [ "$ran" -lt 60 ]; then
        delay=$(( restarts < 5 ? 30 * restarts : 150 ))
    else
        delay=10
    fi
    log "agent died rc=$rc after ${ran}s (restart #$restarts); retrying in ${delay}s"
    sleep "$delay"
done
