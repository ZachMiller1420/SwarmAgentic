#!/usr/bin/env bash
set -euo pipefail

export DISPLAY=${DISPLAY:-:99}
export HF_HOME=${HF_HOME:-/cache/huggingface}
mkdir -p "$HF_HOME" /app/logs /app/results

echo "Starting Xvfb on $DISPLAY ..."
Xvfb $DISPLAY -screen 0 1920x1080x24 -ac +extension RANDR &

echo "Starting lightweight window manager (fluxbox) ..."
fluxbox >/dev/null 2>&1 &

echo "Starting x11vnc ..."
x11vnc -display $DISPLAY -forever -shared -nopw -rfbport 5900 -quiet >/dev/null 2>&1 &

echo "Starting websockify/noVNC on :8080 ..."
websockify --web=/usr/share/novnc 0.0.0.0:8080 localhost:5900 >/dev/null 2>&1 &

echo "Launching application GUI ..."
exec python3 /app/main.py

