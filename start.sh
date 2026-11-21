#!/usr/bin/env bash
# Pose Perfect - one-shot setup + start script (macOS / Linux)
# Usage: ./start.sh            start everything (sets up on first run)
#        ./start.sh --setup    only run setup, do not start the servers
#        ./start.sh --clean    delete .venv + node_modules, then set up and start

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

VENV_DIR="$PROJECT_ROOT/.venv"
FRONTEND_DIR="$PROJECT_ROOT/Frontend"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"

SETUP_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --setup) SETUP_ONLY=1 ;;
    --clean) echo "Removing .venv and Frontend/node_modules..."
             rm -rf "$VENV_DIR" "$FRONTEND_DIR/node_modules" ;;
    -h|--help) sed -n '2,5p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
  esac
done

info() { printf '\033[1;36m==>\033[0m %s\n' "$1"; }
fail() { printf '\033[1;31mERROR:\033[0m %s\n' "$1" >&2; exit 1; }

# ---------------------------------------------------------------- prerequisites
info "Checking prerequisites"

PYTHON_BIN=""
for candidate in python3.11 python3.12 python3.10 python3.13 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done
[ -n "$PYTHON_BIN" ] || fail "Python 3.9+ not found. Install it from https://www.python.org/downloads/"

command -v node >/dev/null 2>&1 || fail "Node.js not found. Install it from https://nodejs.org/ (v18 or newer)."
command -v npm  >/dev/null 2>&1 || fail "npm not found. It ships with Node.js - reinstall from https://nodejs.org/"

echo "    Python: $("$PYTHON_BIN" --version) ($PYTHON_BIN)"
echo "    Node:   $(node --version)"

# ---------------------------------------------------------------- backend setup
if [ ! -d "$VENV_DIR" ]; then
  info "Creating Python virtual environment (.venv)"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

VENV_PY="$VENV_DIR/bin/python"

# Install requirements only when they changed since the last successful install.
REQ_STAMP="$VENV_DIR/.requirements.sha"
REQ_HASH="$(shasum -a 256 requirements.txt 2>/dev/null | awk '{print $1}' || sha256sum requirements.txt | awk '{print $1}')"
if [ ! -f "$REQ_STAMP" ] || [ "$(cat "$REQ_STAMP")" != "$REQ_HASH" ]; then
  info "Installing Python dependencies (first run takes a few minutes)"
  "$VENV_PY" -m pip install --upgrade pip --quiet
  "$VENV_PY" -m pip install -r requirements.txt
  echo "$REQ_HASH" > "$REQ_STAMP"
else
  info "Python dependencies already installed"
fi

# ---------------------------------------------------------------- model weights
if [ ! -f "$PROJECT_ROOT/yolo11m-pose.pt" ]; then
  info "Downloading YOLO11 pose model"
  "$VENV_PY" -c "from ultralytics import YOLO; YOLO('yolo11m-pose.pt')"
fi

# ---------------------------------------------------------------- frontend setup
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
  info "Installing frontend dependencies"
  ( cd "$FRONTEND_DIR" && npm install --no-audit --no-fund )
else
  info "Frontend dependencies already installed"
fi

if [ "$SETUP_ONLY" -eq 1 ]; then
  info "Setup complete. Run ./start.sh to launch the app."
  exit 0
fi

# ---------------------------------------------------------------- start servers
API_PID=""
WEB_PID=""

# Kill a process and everything it spawned. `npm run dev` starts vite as a child,
# so signalling npm alone would leave the dev server holding the port.
kill_tree() {
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree "$child"
  done
  kill "$pid" 2>/dev/null || true
}

shutdown() {
  trap - INT TERM
  echo ""
  info "Shutting down"
  [ -n "$WEB_PID" ] && kill_tree "$WEB_PID"
  [ -n "$API_PID" ] && kill_tree "$API_PID"
  wait 2>/dev/null || true
  exit 0
}
trap shutdown INT TERM

port_busy() { lsof -nP -iTCP:"$1" -sTCP:LISTEN >/dev/null 2>&1; }

port_busy "$API_PORT" && fail "Port $API_PORT is already in use. Stop that process or run: API_PORT=8001 ./start.sh"
port_busy "$WEB_PORT" && fail "Port $WEB_PORT is already in use. Stop that process or run: WEB_PORT=5174 ./start.sh"

info "Starting API server on http://localhost:$API_PORT"
# Run from the project root: the API resolves trained_models/ and the model weights relative to the CWD.
"$VENV_PY" -m uvicorn api.api_server:app --host 0.0.0.0 --port "$API_PORT" &
API_PID=$!

# Wait for the API to answer before starting the UI, so the first page load finds it.
info "Waiting for the API to become ready (loading pose models)"
for _ in $(seq 1 120); do
  if curl -fsS "http://localhost:$API_PORT/" >/dev/null 2>&1; then
    echo "    API is ready."
    break
  fi
  if ! kill -0 "$API_PID" 2>/dev/null; then
    fail "API server exited during startup. Scroll up for the Python traceback."
  fi
  sleep 1
done

info "Starting frontend on http://localhost:$WEB_PORT"
( cd "$FRONTEND_DIR" && npm run dev -- --port "$WEB_PORT" --strictPort ) &
WEB_PID=$!

cat <<EOF

------------------------------------------------------------
  Pose Perfect is running
    App : http://localhost:$WEB_PORT
    API : http://localhost:$API_PORT  (docs at /docs)

  Press Ctrl+C to stop both servers.
------------------------------------------------------------

EOF

# Poll instead of `wait -n`, which needs bash 5.1 (macOS still ships bash 3.2).
# If either server dies, take the other one down with it.
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 1
done
shutdown
