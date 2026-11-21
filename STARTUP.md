# Startup Guide

One script sets up and starts the whole project — Python API + React frontend.

## Prerequisites

Install these once:

- **Python 3.9+** — https://www.python.org/downloads/ (on Windows, tick *"Add python.exe to PATH"*)
- **Node.js 18+** — https://nodejs.org/

## Start the project

**macOS / Linux**

```bash
./start.sh
```

**Windows** — double-click `start.bat`, or from PowerShell:

```powershell
.\start.ps1
```

The first run installs everything (a few minutes — it downloads PyTorch). Later runs
start in seconds.

When it's ready, open **http://localhost:5173**.

Press **Ctrl+C** in the terminal to stop both servers.

## What the script does

1. Checks that Python and Node are installed
2. Creates `.venv/` and installs the Python dependencies from `requirements.txt`
3. Downloads the YOLO11 pose model if it's missing
4. Installs the frontend dependencies (`Frontend/node_modules`)
5. Starts the API on port **8000**, waits for it to load the pose models
6. Starts the frontend on port **5173**

Steps 2–4 are skipped once they're done, so only the first run is slow.

## Options

| Command | What it does |
|---|---|
| `./start.sh` / `.\start.ps1` | Set up (if needed) and start everything |
| `./start.sh --setup` / `.\start.ps1 -Setup` | Only install dependencies, don't start |
| `./start.sh --clean` / `.\start.ps1 -Clean` | Delete `.venv` + `node_modules`, reinstall, start |

Ports are configurable if 8000 or 5173 are taken:

```bash
API_PORT=8001 WEB_PORT=5174 ./start.sh          # macOS / Linux
```
```powershell
.\start.ps1 -ApiPort 8001 -WebPort 5174          # Windows
```

## Troubleshooting

**"Port 8000 is already in use"** — something else is on that port. Stop it, or use the
port options above.

**"API Disconnected" in the browser** — the API is still loading its pose models. Give it
a few seconds and click *Retry Connection*.

**"START GAME" is greyed out** — the page needs a webcam. Allow camera access when the
browser asks, then reload.

**Installation failed** — run `./start.sh --clean` (`.\start.ps1 -Clean`) to wipe the
environments and reinstall from scratch.
