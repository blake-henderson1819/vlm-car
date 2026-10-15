#!/bin/bash
# Double-click launcher (macOS). Expects a virtualenv at software/.venv (see README).
cd "$(dirname "$0")" && .venv/bin/python pad.py
# close only THIS Terminal window (matched by its tty) once the program exits
T=$(tty)
osascript -e "tell application \"Terminal\" to repeat with w in windows" -e "if tty of selected tab of w is \"$T\" then close w" -e "end repeat" >/dev/null 2>&1
exit 0
