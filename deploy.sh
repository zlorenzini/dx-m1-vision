#!/usr/bin/env bash
# Copy the app to the DX-M1 Pi and (re)start its systemd service.
set -euo pipefail
HOST="${1:-192.168.0.101}"
DEST="dx-m1-phone-app"
cd "$(dirname "$0")"
ssh "$HOST" "mkdir -p ~/$DEST/static"
scp -q server.py engine.py dx-m1-phone-app.service "$HOST:$DEST/"
scp -q static/index.html "$HOST:$DEST/static/"
ssh "$HOST" "sudo install -m 644 ~/$DEST/dx-m1-phone-app.service /etc/systemd/system/ && \
  sudo systemctl daemon-reload && sudo systemctl enable --now dx-m1-phone-app && \
  sudo systemctl restart dx-m1-phone-app && sleep 2 && systemctl --no-pager -l status dx-m1-phone-app | head -12"
echo "Open http://$HOST:8090 on your phone (same Wi-Fi)."
