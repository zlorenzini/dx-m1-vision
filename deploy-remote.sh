#!/usr/bin/env bash
# Install or update DX-M1 Vision on another Linux machine over SSH.
#
#   ./deploy-remote.sh user@host [install.py options...]
#   ./deploy-remote.sh pi@192.168.1.50 --port 8090 --download-models
#
# Copies this folder to ~/dx-m1-vision on the remote machine and runs install.py there
# with --yes (auto-detects DX-RT, installs the systemd service). The remote user needs
# sudo for the service step; pass --no-service to skip it.
set -euo pipefail
if [ $# -lt 1 ]; then sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 1; fi
HOST="$1"; shift
DEST="${DEST:-dx-m1-vision}"
cd "$(dirname "$0")"

FILES=(server.py engine.py config.py install.py static LICENSE README.md)
echo "==> copying to $HOST:~/$DEST"
tar czf - "${FILES[@]}" | ssh "$HOST" "mkdir -p ~/$DEST && tar xzf - -C ~/$DEST"

echo "==> running install.py on $HOST"
# -t gives sudo a terminal for its password prompt if the remote user needs one
ssh -t "$HOST" "cd ~/$DEST && python3 install.py --yes $*"
