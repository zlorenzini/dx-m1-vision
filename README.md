# DX-M1 Vision (phone web app)

Take a photo or a video of up to 10 s on your phone, send it to a DEEPX DX-M1 NPU, and get it back with
bounding boxes (object detection) or stick figures (pose).

- **Server:** `server.py` + `engine.py`. Uses only the Python standard library, plus the `cv2`, `numpy` and `dx_engine`
  that the DX-RT install already provides.
- **Phone page:** `static/index.html`. A single file that uses the phone's native camera through `<input capture>`,
  so it works over plain HTTP on the LAN.
- **Models:** YOLO26 end-to-end exports from the DEEPX model zoo (`models-2_2_0`). They need no NMS.

| Mode | Model | NPU time (DX-M1 on a Pi CM5) |
|---|---|---|
| Objects | yolo26s-1.dxnn | ~45 ms |
| Objects (accurate) | yolo26m-1.dxnn | |
| Pose | yolo26s-pose.dxnn | ~35 ms |
| Pose (accurate) | yolo26m-pose.dxnn | ~43 ms |

To add a mode, add a line to `MODES` in `engine.py`. A mode only appears if its `.dxnn` file is present.

Video is automatically rotated using the phone's orientation metadata, capped at 30 fps and 1280 px on the long side,
and trimmed to 10 s. It comes back as H.264 MP4 with the original audio. A 10 s clip takes about 20 s end to end.

## Run

```bash
/path/to/dx-runtime/venv-dx-runtime/bin/python3 server.py --port 8090
# Optional environment variables: DX_MODEL_DIR=/path/to/models  DXAPP_WORK=/path/to/scratch
```

`./deploy.sh [host]` copies the app to `~/dx-m1-phone-app` on the host and installs the `dx-m1-phone-app` systemd service.

## API

`POST /api/jobs?kind=image|video&mode=<id>&conf=0.35` with the raw file as the body → `{id}`.
Then poll `GET /api/jobs/<id>` and fetch `GET /api/jobs/<id>/result` (JPEG or MP4).
