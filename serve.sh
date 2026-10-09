#!/bin/sh
# Serves the repo so the demo can load its local model files (file:// cannot).
cd "$(dirname "$0")" && echo "Deck: http://localhost:8000/deck/  Demo: http://localhost:8000/demo/" && python3 -m http.server 8000 --bind 127.0.0.1
