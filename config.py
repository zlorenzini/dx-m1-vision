"""config.py — settings and environment discovery shared by server.py and install.py.

Pure standard library and platform agnostic (Linux x86_64 / aarch64, Windows).

Resolution order for every setting:
    environment variable  >  config.json (written by install.py)  >  auto-discovery
"""

import glob
import json
import os
import platform
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(HERE, "config.json")
IS_WINDOWS = platform.system() == "Windows"

# Model files the app knows how to drive (YOLO26 end-to-end exports, DEEPX model zoo 2.2.0+).
REQUIRED_MODELS = ["yolo26s-1.dxnn", "yolo26s-pose.dxnn"]
OPTIONAL_MODELS = ["yolo26m-1.dxnn", "yolo26m-pose.dxnn"]
MODEL_ZOO_URL = "https://sdk.deepx.ai/res/models/models-2_2_0.tar.gz"

# Preferred H.264 encoders, best first. All produce MP4s that phones can play.
# mpeg4 is a last resort: Android plays it; iOS may not.
H264_ENCODERS = ["libx264", "libopenh264", "h264_nvenc", "h264_mf", "mpeg4"]


def load_file_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_file_config(cfg):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")


def home():
    return os.path.expanduser("~")


# ── DX-RT install / model discovery ──────────────────────────────────────────

def dxrt_roots():
    """Likely DX-RT install roots (directories that contain dx_app / venv-dx-runtime)."""
    h = home()
    pats = [
        os.path.join(h, "dx-all-suite", "dx-runtime"),
        os.path.join(h, "dx_all_suite", "dx-runtime"),
        os.path.join(h, "dx-runtime"),
        os.path.join(h, "deepx", "dx-runtime"),
        os.path.join(h, "*", "dx-runtime"),
        os.path.join(h, "*", "*", "dx-runtime"),
        "/opt/dx-runtime", "/opt/deepx/dx-runtime", "/opt/dx-all-suite/dx-runtime",
        "/usr/local/dx-runtime",
        "/mnt/*/dx-runtime", "/mnt/*/*/dx-runtime", "/mnt/*/*/*/dx-runtime",
        "/mnt/*/accelerators/*/dx-runtime",
    ]
    if IS_WINDOWS:
        pats += [r"C:\dx-all-suite\dx-runtime", r"C:\deepx\dx-runtime", r"C:\dx-runtime",
                 os.path.join(os.environ.get("ProgramFiles", r"C:\Program Files"), "DEEPX", "*")]
    seen, out = set(), []
    for p in pats:
        for d in sorted(glob.glob(p)):
            d = os.path.realpath(d)
            if os.path.isdir(d) and d not in seen:
                seen.add(d)
                out.append(d)
    return out


def venv_python(venv_dir):
    exe = os.path.join(venv_dir, "Scripts", "python.exe") if IS_WINDOWS else os.path.join(venv_dir, "bin", "python3")
    return exe if os.path.isfile(exe) else None


def candidate_pythons():
    """Python interpreters that might have dx_engine installed, most likely first."""
    out = []
    if os.environ.get("DXRT_PYTHON"):
        out.append(os.environ["DXRT_PYTHON"])
    for root in dxrt_roots():
        for v in ("venv-dx-runtime", "venv", ".venv"):
            p = venv_python(os.path.join(root, v))
            if p:
                out.append(p)
        for v in glob.glob(os.path.join(os.path.dirname(root), "venv*")):
            p = venv_python(v)
            if p:
                out.append(p)
    for name in (["python", "py"] if IS_WINDOWS else ["python3", "python"]):
        p = shutil.which(name)
        if p:
            out.append(p)
    dedup = []
    for p in out:
        if p not in dedup:
            dedup.append(p)
    return dedup


def model_dir_has(d, names=REQUIRED_MODELS):
    return all(os.path.isfile(os.path.join(d, n)) for n in names)


def find_model_dir():
    """Directory containing the required .dxnn files, or None."""
    env = os.environ.get("DX_MODEL_DIR")
    if env:
        return env
    cfg = load_file_config().get("model_dir")
    if cfg and model_dir_has(cfg):
        return cfg
    cands = [os.path.join(HERE, "models")]
    for root in dxrt_roots():
        cands += sorted(glob.glob(os.path.join(root, "dx_app", "assets", "models", "models-*")), reverse=True)
        cands += sorted(glob.glob(os.path.join(root, "..", "workspace", "res", "models", "models-*")), reverse=True)
        cands += sorted(glob.glob(os.path.join(root, "dx_app", "assets", "models")), reverse=True)
    for d in cands:
        if model_dir_has(d):
            return os.path.realpath(d)
    return None


def npu_devices():
    """Device nodes for DX-M1 cards (Linux). Windows has no /dev; returns None there."""
    if IS_WINDOWS:
        return None
    return sorted(glob.glob("/dev/dxrt*"))


# ── ffmpeg ───────────────────────────────────────────────────────────────────

def ffmpeg_path():
    env = os.environ.get("FFMPEG")
    if env:
        return env
    cfg = load_file_config().get("ffmpeg")
    if cfg and os.path.isfile(cfg):
        return cfg
    return shutil.which("ffmpeg")


def h264_encoder(ffmpeg):
    """Best available H.264 encoder name for this ffmpeg, or None."""
    if not ffmpeg:
        return None
    env = os.environ.get("H264_ENCODER") or load_file_config().get("h264_encoder")
    try:
        txt = subprocess.run([ffmpeg, "-hide_banner", "-encoders"], capture_output=True,
                             text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    names = {line.split()[1] for line in txt.splitlines() if len(line.split()) > 1 and line.startswith(" V")}
    if env and env in names:
        return env
    for enc in H264_ENCODERS:
        if enc in names and _encoder_works(ffmpeg, enc):
            return enc
    return None


def _encoder_works(ffmpeg, enc):
    """Hardware encoders are listed even with no hardware; do a 1-frame trial."""
    if enc in ("libx264", "libopenh264", "mpeg4"):
        return True
    try:
        r = subprocess.run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=c=black:s=128x128:d=0.1",
                            "-frames:v", "1", "-pix_fmt", "yuv420p", "-c:v", enc, "-f", "null", "-"],
                           capture_output=True, timeout=20)
        return r.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def encoder_args(enc):
    """Quality/speed arguments for each encoder."""
    return {
        "libx264":     ["-preset", "veryfast", "-crf", "23"],
        "libopenh264": ["-b:v", "6M"],
        "h264_nvenc":  ["-preset", "p4", "-cq", "23"],
        "h264_mf":     ["-b:v", "6M"],
        "mpeg4":       ["-q:v", "4"],
    }.get(enc, [])


def lan_addresses():
    """Best-guess LAN IPv4 addresses for printing phone URLs (sends no packets)."""
    import socket
    addrs = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        addrs.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in addrs:
                addrs.append(ip)
    except OSError:
        pass
    return addrs or ["127.0.0.1"]
