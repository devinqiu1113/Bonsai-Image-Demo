#!/bin/sh
# Stop Bonsai-Image-Demo WebUI (backend + frontend)
# Usage: ./scripts/stop.sh

set -e

echo "Stopping Bonsai Image Demo WebUI..."

# Kill backend (port 8000)
BACKEND_PID=$(lsof -nP -iTCP:8000 -sTCP:LISTEN -t 2>/dev/null)
if [ -n "$BACKEND_PID" ]; then
    echo "  Stopping backend (PID $BACKEND_PID, port 8000)..."
    kill -TERM $BACKEND_PID 2>/dev/null || true
    sleep 1
    # Force kill if still alive
    if kill -0 $BACKEND_PID 2>/dev/null; then
        kill -KILL $BACKEND_PID 2>/dev/null || true
    fi
else
    echo "  Backend not running on port 8000."
fi

# Kill frontend (port 3000)
FRONTEND_PID=$(lsof -nP -iTCP:3000 -sTCP:LISTEN -t 2>/dev/null)
if [ -n "$FRONTEND_PID" ]; then
    echo "  Stopping frontend (PID $FRONTEND_PID, port 3000)..."
    # Kill the process tree (npm → next-server)
    KIDS=$(pgrep -P "$FRONTEND_PID" 2>/dev/null || true)
    for KID in $KIDS; do
        kill -TERM "$KID" 2>/dev/null || true
    done
    kill -TERM "$FRONTEND_PID" 2>/dev/null || true
    sleep 1
    # Force kill remaining
    for PORT in 3000 8000; do
        REMAINING=$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null)
        if [ -n "$REMAINING" ]; then
            kill -KILL $REMAINING 2>/dev/null || true
        fi
    done
else
    echo "  Frontend not running on port 3000."
fi

echo ""
echo "Done."
