# DX-M1 Vision

A phone web app for the **DEEPX DX-M1** NPU. You take a photo, or a video of up to 10 s, on your phone, send it to the
machine with the DX-M1, and get it back with **bounding boxes** (object detection) or **stick figures** (pose estimation).

- Works with any DX-M1 host: **x86_64 Linux** PCs, **ARM64** boards such as the Raspberry Pi 5 or CM5, and **Windows**.
- The phone needs only a browser. The page uses the phone's own camera, so there's no app to install and no HTTPS to set up.
- The server uses only the Python standard library plus the `dx_engine`, OpenCV and numpy that DX-RT already provides.

| Mode | Model (DEEPX model zoo 2.2.0) | NPU time per frame* |
|---|---|---|
| Objects | `yolo26s-1.dxnn`: 80 COCO classes | ~45 ms |
| Objects (accurate) | `yolo26m-1.dxnn` | ~53 ms |
| Pose | `yolo26s-pose.dxnn`: 17-keypoint skeletons | ~36 ms |
| Pose (accurate) | `yolo26m-pose.dxnn` | ~44 ms |

\*Measured on a DX-M1 with a Raspberry Pi CM5 host. Other hosts will differ.

## Requirements

- A DX-M1 with the DEEPX **driver** and **DX-RT 3.0+** installed and working (`dxrt-cli -s` shows the device).
  The usual way to install them is DEEPX's `dx-all-suite`; its `install.sh` creates the `venv-dx-runtime` Python environment.
- **ffmpeg** for video. Photos work without it, and the installer offers to install it.
- Python 3.8+ to run the installer.

## Install

```bash
git clone https://github.com/zlorenzini/dx-m1-vision.git
cd dx-m1-vision
python3 install.py
```

The installer:

1. finds the Python that has `dx_engine` (it searches the usual DX-RT locations; to choose one yourself, use `--python /path/to/venv/bin/python3`)
2. checks OpenCV and numpy in that environment, and offers to `pip install opencv-python-headless` if either is missing
3. checks for the DX-M1 device and for ffmpeg with an H.264 encoder (libx264, OpenH264, NVENC or Media Foundation)
4. finds the YOLO26 models. If they're missing, it streams DEEPX's public model zoo archive and keeps only the four files it needs (or use `--model-dir`)
5. writes `config.json`, then runs a **self-test of every mode on the NPU**
6. optionally sets the server to start at boot: a **systemd** service on Linux, or a **logon task** on Windows

At the end it prints the address to open on your phone, for example `http://192.168.1.50:8090`. The phone must be on the
same network.

Useful flags: `--yes` (no prompts), `--port 8090`, `--no-service`, `--download-models`, `--model-dir DIR`, `--python PY`.

### Install on another machine over SSH

```bash
./deploy-remote.sh user@dx-host [install.py flags...]
```

This copies the app to `~/dx-m1-vision` on that machine and runs `install.py --yes` there.

### Run manually

```bash
./run-server.sh            # Linux (created by install.py)
run-server.bat             # Windows
<dx-rt-python> server.py --port 8090
<dx-rt-python> server.py --self-test
```

If the service is installed: `sudo systemctl status dx-m1-vision` and `journalctl -u dx-m1-vision -f`.

## Using it

1. Open the address on your phone. To make it feel like an app, use "Add to Home Screen".
2. Pick a mode, and adjust the confidence slider if you like.
3. Choose **Photo** or **Video**, then **Take photo / Record video**, or pick a file from your **Library**.
4. Tap **Send to DX-M1**. You get back the annotated photo or MP4, counts of what was found, and Save/Share buttons.

Videos are rotated using the phone's orientation metadata, capped at 30 fps and 1280 px on the long side, and
**trimmed to the first 10 s**. They come back as H.264 MP4 with the original audio. A 10 s clip takes about 20 s end to end.
The NPU handles one job at a time; requests from other phones wait in line.

## Configuration

`config.json` is written by the installer. You can override any setting with an environment variable:

| Setting | Env var | Default |
|---|---|---|
| port / bind address | `PORT` / `HOST` | `8090` / `0.0.0.0` |
| model folder | `DX_MODEL_DIR` | auto-detected |
| ffmpeg binary | `FFMPEG` | `ffmpeg` on PATH |
| H.264 encoder | `H264_ENCODER` | best available |
| scratch folder | `DXAPP_WORK` | `./work` |

To add a mode, add a line to `MODES` in `engine.py`. The current code handles YOLO26 end-to-end outputs
(`[N, 6]` boxes and `[N, 57]` pose rows).

## API

```
GET  /api/modes
POST /api/jobs?kind=image|video&mode=<id>&conf=0.35     body = raw file   -> {"id": ...}
GET  /api/jobs/<id>                                     -> {"state": queued|running|done|error, "progress": 0..1, "summary": {...}}
GET  /api/jobs/<id>/result                              -> annotated JPEG or MP4
```

## Security note

The server has no login. Run it only on a network you trust. Don't expose it to the internet.

## License

MIT. See [LICENSE](LICENSE). The DX-M1 runtime and the models come from DEEPX and are covered by DEEPX's own licenses.
