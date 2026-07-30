#!/bin/bash
cd "$(dirname "$0")"

# We use AppleScript to cleanly open two new terminal windows.
# Window 1: Proxy Engine (omniroute or swiftlet.cli)
# Window 2: Chat UI (swiftlet.py)

DIR_PATH=$(pwd)

osascript <<EOF
tell application "Terminal"
    do script "cd '$DIR_PATH' && source .venv/bin/activate && python3 -m swiftlet.cli"
    do script "cd '$DIR_PATH' && source .venv/bin/activate && python3 -m swiftlet.app"
end tell
EOF

# Optional: Close the launcher window (uncomment if desired)
# osascript -e 'tell application "Terminal" to close (every window whose name contains "start.command")' &
