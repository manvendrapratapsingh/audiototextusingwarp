#!/usr/bin/env bash
set -euo pipefail

# Advanced Audio Transcription - Run/Serve Script
# - Foreground by default (best for debugging)
# - Background mode with logs and health check
# - Optional force-kill of existing process on the chosen port
# - Configurable port, log-level, and reload

PORT=8000
LOG_LEVEL="debug"     # debug | info | warning | error
RELOAD=1               # 1=enable, 0=disable
BACKGROUND=0           # 0=foreground, 1=background
FORCE_KILL=0           # Kill any process listening on $PORT before start
DIAGNOSE=0             # Run diagnose.py and exit

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -p, --port <PORT>        Port to bind (default: 8000)
  -l, --log-level <LEVEL>  Uvicorn log level (debug|info|warning|error) (default: debug)
  -n, --no-reload          Disable autoreload (default: enabled)
  -b, --background         Run in background, write logs to logs/server.out
  -f, --force              Kill any process already listening on the chosen port
      --diagnose           Run diagnose.py to validate environment and imports
  -h, --help               Show this help

Examples:
  $0                         # foreground, port 8000, debug, reload on
  $0 -b -f                   # background, force-kill existing, debug, reload on
  $0 -p 9000 -l info -n      # foreground on port 9000, info logs, no reload
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--port)
      PORT=${2:-8000}; shift 2;;
    -l|--log-level)
      LOG_LEVEL=${2:-debug}; shift 2;;
    -n|--no-reload)
      RELOAD=0; shift;;
    -b|--background)
      BACKGROUND=1; shift;;
    -f|--force)
      FORCE_KILL=1; shift;;
    --diagnose)
      DIAGNOSE=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1"; usage; exit 1;;
  esac
done

# Ensure virtual environment exists
if [[ ! -d "venv" ]]; then
  echo "❌ Virtual environment not found. Please run ./install.sh first."
  exit 1
fi

# Prepare directories
mkdir -p uploads temp/cache temp/chunks logs

# Performance/env
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}
export PYTHONUNBUFFERED=1

# Optionally run diagnostics
if [[ "$DIAGNOSE" -eq 1 ]]; then
  echo "🧪 Running diagnostics..."
  venv/bin/python diagnose.py
  exit 0
fi

# Check port usage
if lsof -iTCP:${PORT} -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  echo "⚠️  Port ${PORT} is in use."
  if [[ "$FORCE_KILL" -eq 1 ]]; then
    echo "🔪 Killing existing process(es) on port ${PORT}..."
    PIDS=$(lsof -tiTCP:${PORT} -sTCP:LISTEN -n -P || true)
    if [[ -n "$PIDS" ]]; then
      for p in $PIDS; do
        kill "$p" 2>/dev/null || true
      done
      sleep 1
      # Hard kill if still present
      for p in $PIDS; do
        if kill -0 "$p" 2>/dev/null; then
          kill -9 "$p" 2>/dev/null || true
        fi
      done
    fi
    echo "✅ Port ${PORT} cleared."
  else
    echo "Run with -f/--force to kill the existing process, or choose another port with -p."
    exit 1
  fi
fi

# Activate environment
source venv/bin/activate

# Launch server
RELOAD_FLAG=(--reload)
if [[ "$RELOAD" -eq 0 ]]; then
  RELOAD_FLAG=()
fi

echo "🎤 Starting Advanced Audio Transcription server"
echo "   • URL:      http://localhost:${PORT}"
echo "   • LogLevel: ${LOG_LEVEL}"
echo "   • Reload:   $([[ "$RELOAD" -eq 1 ]] && echo enabled || echo disabled)"
echo "   • Mode:     $([[ "$BACKGROUND" -eq 1 ]] && echo background || echo foreground)"

UVICORN_CMD=(python3 -m uvicorn backend.app:app --host 0.0.0.0 --port "${PORT}" --log-level "${LOG_LEVEL}" "${RELOAD_FLAG[@]}")

if [[ "$BACKGROUND" -eq 1 ]]; then
  LOG_FILE="logs/server.out"
  echo "📝 Writing logs to ${LOG_FILE}"
  nohup bash -lc "${UVICORN_CMD[*]}" > "$LOG_FILE" 2>&1 & echo $! > logs/server.pid
  sleep 2
  PID=$(cat logs/server.pid 2>/dev/null || true)
  echo "✅ Started in background (PID: ${PID:-unknown})"
  echo "🔍 Health check: curl -s http://localhost:${PORT}/health | jq ."
  echo "📜 Tail logs:    tail -n 200 -f ${LOG_FILE}"
else
  echo "Press Ctrl+C to stop the server"
  exec "${UVICORN_CMD[@]}"
fi
