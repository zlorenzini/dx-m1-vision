"""engine.py — DX-M1 model registry, inference, and drawing.

Models are YOLO26 end-to-end (.dxnn) exports from the DEEPX model zoo:
  * detect: output (1, 300, 6)  -> x1, y1, x2, y2, score, class_id
  * pose:   output (1, 300, 57) -> x1, y1, x2, y2, score, class_id, 17 x (kx, ky, kconf)
No NMS is needed; rows are already final detections in letterboxed input space.
"""

import os
import threading
import time

import cv2
import numpy as np
from dx_engine import InferenceEngine

MODEL_DIR = os.environ.get(
    "DX_MODEL_DIR",
    "/mnt/jumpdata/accelerators/dx_m1/dx-runtime/dx_app/assets/models/models-2_2_0",
)

# id -> (label, description, kind, model file)
MODES = {
    "detect":    ("Objects",          "80 everyday object types (YOLO26s)",   "detect", "yolo26s-1.dxnn"),
    "detect-m":  ("Objects (accurate)", "Slower, catches more (YOLO26m)",     "detect", "yolo26m-1.dxnn"),
    "pose":      ("Pose",             "Stick figures for people (YOLO26s-pose)", "pose", "yolo26s-pose.dxnn"),
    "pose-m":    ("Pose (accurate)",  "Slower, steadier joints (YOLO26m-pose)",  "pose", "yolo26m-pose.dxnn"),
}

COCO_80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

# COCO-17 keypoints: 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows,
# 9-10 wrists, 11-12 hips, 13-14 knees, 15-16 ankles.  Colors are BGR.
HEAD, ARMS, LEGS, TORSO = (80, 220, 80), (0, 160, 255), (255, 150, 50), (230, 80, 230)
SKELETON = [
    ((15, 13), LEGS), ((13, 11), LEGS), ((16, 14), LEGS), ((14, 12), LEGS),
    ((11, 12), TORSO), ((5, 11), TORSO), ((6, 12), TORSO), ((5, 6), TORSO),
    ((5, 7), ARMS), ((6, 8), ARMS), ((7, 9), ARMS), ((8, 10), ARMS),
    ((1, 2), HEAD), ((0, 1), HEAD), ((0, 2), HEAD), ((1, 3), HEAD), ((2, 4), HEAD),
]
KPT_MIN_CONF = 0.5


def available_modes():
    out = []
    for mid, (label, desc, kind, fname) in MODES.items():
        if os.path.exists(os.path.join(MODEL_DIR, fname)):
            out.append({"id": mid, "label": label, "description": desc, "kind": kind})
    return out


def class_color(class_id):
    """Stable, saturated BGR color per class id."""
    hue = (class_id * 47) % 180
    hsv = np.uint8([[[hue, 220, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


class Runner:
    """Loads .dxnn engines lazily and serializes access to the NPU."""

    def __init__(self):
        self._engines = {}
        self._lock = threading.Lock()

    def _engine(self, mode):
        if mode not in self._engines:
            fname = MODES[mode][3]
            self._engines[mode] = InferenceEngine(os.path.join(MODEL_DIR, fname))
        return self._engines[mode]

    def infer(self, mode, img_bgr, conf):
        """Run one BGR frame. Returns (detections ndarray in image coords, ms)."""
        kind = MODES[mode][2]
        h, w = img_bgr.shape[:2]
        with self._lock:
            ie = self._engine(mode)
            ih, iw = ie.get_input_tensors_info()[0]["shape"][1:3]
            gain = min(ih / h, iw / w)
            nw, nh = int(round(w * gain)), int(round(h * gain))
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if (nw, nh) != (w, h):
                rgb = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
            top, left = (ih - nh) // 2, (iw - nw) // 2
            tensor = cv2.copyMakeBorder(rgb, top, ih - nh - top, left, iw - nw - left,
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
            t0 = time.perf_counter()
            out = ie.run([tensor])
            ms = (time.perf_counter() - t0) * 1000

        rows = np.squeeze(out[0], axis=0).astype(np.float32)
        rows = rows[rows[:, 4] >= conf].copy()
        if len(rows):
            rows[:, [0, 2]] = np.clip((rows[:, [0, 2]] - left) / gain, 0, w - 1)
            rows[:, [1, 3]] = np.clip((rows[:, [1, 3]] - top) / gain, 0, h - 1)
            if kind == "pose":
                rows[:, 6::3] = np.clip((rows[:, 6::3] - left) / gain, 0, w - 1)
                rows[:, 7::3] = np.clip((rows[:, 7::3] - top) / gain, 0, h - 1)
        return rows, ms


def summarize(mode, rows):
    """Count detections per class name."""
    kind = MODES[mode][2]
    counts = {}
    for r in rows:
        name = "person" if kind == "pose" else COCO_80[int(r[5])] if 0 <= int(r[5]) < 80 else str(int(r[5]))
        counts[name] = counts.get(name, 0) + 1
    return counts


def draw(mode, img, rows):
    """Draw boxes (detect) or stick figures (pose) onto img in place."""
    kind = MODES[mode][2]
    h, w = img.shape[:2]
    s = max(h, w) / 1000.0
    thick = max(2, int(round(3 * s)))
    font_scale = max(0.5, 0.7 * s)
    font_thick = max(1, int(round(2 * s)))

    for r in rows:
        x1, y1, x2, y2, score, cid = r[:6]
        cid = int(cid)
        if kind == "pose":
            color = (255, 255, 255)
            name = "person"
            box_thick = max(1, thick // 2)
        else:
            color = class_color(cid)
            name = COCO_80[cid] if 0 <= cid < 80 else str(cid)
            box_thick = thick
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, box_thick, cv2.LINE_AA)

        label = f"{name} {score:.0%}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        ly = p1[1] - 4 if p1[1] - th - base - 4 > 0 else p1[1] + th + base + 4
        cv2.rectangle(img, (p1[0], ly - th - base), (p1[0] + tw + 6, ly + base // 2), color, -1)
        cv2.putText(img, label, (p1[0] + 3, ly - base // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thick, cv2.LINE_AA)

        if kind == "pose":
            k = r[6:].reshape(17, 3)
            for (a, b), c in SKELETON:
                if k[a, 2] >= KPT_MIN_CONF and k[b, 2] >= KPT_MIN_CONF:
                    cv2.line(img, (int(k[a, 0]), int(k[a, 1])), (int(k[b, 0]), int(k[b, 1])),
                             c, thick + 1, cv2.LINE_AA)
            for x, y, c in k:
                if c >= KPT_MIN_CONF:
                    cv2.circle(img, (int(x), int(y)), thick + 2, (255, 255, 255), -1, cv2.LINE_AA)
                    cv2.circle(img, (int(x), int(y)), thick + 2, (40, 40, 40), 1, cv2.LINE_AA)
    return img
