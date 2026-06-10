#!/bin/sh
# Stop Bonsai-Image-Demo WebUI (backend + frontend)
# Usage: ./scripts/stop.sh

set -e

echo "Stopping Bonsai Image Demo WebUI..."

# Determine project directory (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── 1. Kill by port (backend 8000, frontend 3000) ──
for PORT in 8000 3000; do
    PID=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "  Port $PORT: stopping PID $PID..."
        kill -TERM "$PID" 2>/dev/null || true
    fi
done
sleep 1

# ── 2. Kill by process name (orphaned processes not bound to ports) ──
for PATTERN in "uvicorn.*local_backend" "next dev" "next start"; do
    PID=$(pgrep -f "$PATTERN" 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "  Killing stray process: $PATTERN (PID $PID)..."
        kill -TERM "$PID" 2>/dev/null || true
    fi
done

# ── 3. Kill serve.sh wrapper shell ──
for PID in $(pgrep -f "serve.sh" 2>/dev/null || true); do
    # Only kill serve.sh processes from THIS project directory
    if grep -q "$SCRIPT_DIR" /proc/$PID/cmdline 2>/dev/null; then
        echo "  Killing serve.sh wrapper (PID $PID)..."
        kill -TERM "$PID" 2>/dev/null || true
    fi
done

sleep 1

# ── 4. Force kill anything still holding our ports ──
for PORT in 8000 3000; do
    PID=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null || true)
    if [ -n "$PID" ]; then
        echo "  Force killing PID $PID on port $PORT..."
        kill -KILL "$PID" 2>/dev/null || true
    fi
done

echo ""
echo "Done."
