#!/usr/bin/env python3
"""install.py — set up DX-M1 Vision on this machine.

Works on Linux (x86_64 and aarch64) and Windows. Run it with any Python 3.8+:

    python3 install.py                 # interactive
    python3 install.py --yes           # accept defaults (installs service, downloads models if missing)
    python3 install.py --help

What it does:
  1. finds the Python that has DEEPX's dx_engine (the DX-RT venv)
  2. checks OpenCV/numpy in it (offers to pip-install opencv-python-headless)
  3. checks for the DX-M1 device node and ffmpeg with an H.264 encoder
  4. finds the YOLO26 .dxnn models, or downloads just those files from the DEEPX model zoo
  5. writes config.json and runs a self-test on the NPU
  6. optionally installs a service (systemd on Linux, a logon task on Windows)
"""

import argparse
import getpass
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import urllib.request

import config

PROBE = r"""
import json, sys
r = {"exe": sys.executable, "version": sys.version.split()[0]}
try:
    import dx_engine
    r["dx_engine"] = True
    try:
        from dx_engine import Configuration
        r["dxrt"] = Configuration().get_version()
    except Exception as e:
        r["dxrt"] = "unknown"
except Exception as e:
    r["dx_engine"] = False
    r["dx_err"] = str(e)
for m in ("cv2", "numpy"):
    try:
        mod = __import__(m)
        r[m] = getattr(mod, "__version__", "yes")
    except Exception:
        r[m] = None
print(json.dumps(r))
"""

ARGS = None


def say(msg=""):
    print(msg, flush=True)


def ok(msg):
    say(f"  [ok]   {msg}")


def warn(msg):
    say(f"  [warn] {msg}")


def fail(msg):
    say(f"  [FAIL] {msg}")


def ask(question, default=True):
    if ARGS.yes or not sys.stdin.isatty():
        return default
    hint = "[Y/n]" if default else "[y/N]"
    ans = input(f"  ? {question} {hint} ").strip().lower()
    return default if not ans else ans.startswith("y")


def run(cmd, **kw):
    say(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")
    return subprocess.run(cmd, **kw)


def probe(py):
    try:
        out = subprocess.run([py, "-c", PROBE], capture_output=True, text=True, timeout=60)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:
        return None


# ── steps ────────────────────────────────────────────────────────────────────

def step_python():
    say("\n1. DX-RT Python (dx_engine)")
    cands = [ARGS.python] if ARGS.python else config.candidate_pythons()
    for py in cands:
        info = probe(py)
        if info and info.get("dx_engine"):
            ok(f"{info['exe']}  (Python {info['version']}, DX-RT {info.get('dxrt')})")
            return info
    fail("No Python with DEEPX dx_engine found.")
    say("     Install DX-RT from DEEPX's dx-all-suite (its install.sh creates venv-dx-runtime),\n"
        "     then rerun, or point at it:  python3 install.py --python /path/to/venv/bin/python3")
    if cands:
        say("     Checked: " + ", ".join(cands))
    return None


def step_cv(info):
    py = info["exe"]
    if info.get("cv2") and info.get("numpy"):
        ok(f"OpenCV {info['cv2']}, numpy {info['numpy']}")
        return
    missing = [p for p, k in (("opencv-python-headless", "cv2"), ("numpy", "numpy")) if not info.get(k)]
    warn(f"missing in the DX-RT venv: {', '.join(missing)}")
    if ask(f"pip install {' '.join(missing)} into {py}?"):
        if run([py, "-m", "pip", "install", *missing]).returncode != 0:
            fail("pip install failed")
            sys.exit(1)
        ok("installed")
    else:
        sys.exit(1)


def step_device():
    say("\n2. DX-M1 device")
    devs = config.npu_devices()
    if devs is None:
        ok("Windows: device check is done by the self-test")
    elif devs:
        ok(", ".join(devs))
        bad = [d for d in devs if not os.access(d, os.R_OK | os.W_OK)]
        if bad:
            warn(f"no read/write access to {', '.join(bad)} as {getpass.getuser()}; "
                 "the self-test will tell if that matters")
    else:
        warn("no /dev/dxrt* device. Is the DX-M1 seated and the dx_rt_npu_linux_driver loaded? "
             "(lsmod | grep dx ; dxrt-cli -s)")
    tool = shutil.which("dxrt-cli")
    if tool:
        r = subprocess.run([tool, "-s"], capture_output=True, text=True, timeout=30)
        line = next((l.strip() for l in r.stdout.splitlines() if "Device" in l or "device" in l), "")
        if line:
            ok(f"dxrt-cli: {line}")


def step_ffmpeg(offer_install=True):
    say("\n3. ffmpeg (needed for video, optional for photos)")
    ff = config.ffmpeg_path()
    if not ff:
        warn("ffmpeg not found")
        hint = None
        if config.IS_WINDOWS:
            hint = ["winget", "install", "--id", "Gyan.FFmpeg", "-e"]
        elif shutil.which("apt-get"):
            hint = ["sudo", "apt-get", "install", "-y", "ffmpeg"]
        elif shutil.which("dnf"):
            hint = ["sudo", "dnf", "install", "-y", "ffmpeg"]
        elif shutil.which("pacman"):
            hint = ["sudo", "pacman", "-S", "--noconfirm", "ffmpeg"]
        if not offer_install and hint:
            say(f"     Install with: {' '.join(hint)}")
        elif hint and ask(f"Install it now with: {' '.join(hint)} ?"):
            run(hint)
            ff = shutil.which("ffmpeg")
            if not ff and config.IS_WINDOWS:
                warn("winget installed ffmpeg; open a new terminal so PATH updates, then rerun install.py")
    if not ff:
        warn("continuing without video support")
        return None, None
    enc = config.h264_encoder(ff)
    if enc in ("libx264", "libopenh264", "h264_nvenc", "h264_mf"):
        ok(f"{ff}  (encoder {enc})")
    elif enc:
        warn(f"{ff} has no H.264 encoder; falling back to {enc} (iPhones may not play results). "
             "Install an ffmpeg build with libx264 for best results.")
    else:
        warn(f"{ff} has no usable video encoder; video disabled")
    return ff, enc


def download_models(dest):
    wanted = set(config.REQUIRED_MODELS + config.OPTIONAL_MODELS)
    os.makedirs(dest, exist_ok=True)
    say(f"  Streaming {config.MODEL_ZOO_URL}")
    say("  (the full zoo is ~1.8 GB; only these files are kept: " + ", ".join(sorted(wanted)) + ")")
    req = urllib.request.Request(config.MODEL_ZOO_URL, headers={"User-Agent": "dx-m1-vision-installer"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        total = int(resp.headers.get("Content-Length") or 0)

        class Counter:
            def __init__(self, f):
                self.f, self.n, self.last = f, 0, -1

            def read(self, k=-1):
                b = self.f.read(k)
                self.n += len(b)
                if total:
                    pct = self.n * 100 // total
                    if pct // 5 != self.last:
                        self.last = pct // 5
                        print(f"\r  downloading… {pct}% ", end="", flush=True)
                return b

        with tarfile.open(fileobj=Counter(resp), mode="r|gz") as tf:
            for m in tf:
                name = os.path.basename(m.name)
                if m.isfile() and name in wanted:
                    with tf.extractfile(m) as src, open(os.path.join(dest, name), "wb") as dst:
                        shutil.copyfileobj(src, dst)
                    wanted.discard(name)
                    print(f"\r  got {name}{' ' * 20}", flush=True)
                    if not wanted:
                        break
    print()
    return config.model_dir_has(dest)


def step_models():
    say("\n4. Models (YOLO26 from the DEEPX model zoo)")
    d = ARGS.model_dir or config.find_model_dir()
    if d and config.model_dir_has(d):
        have = [n for n in config.REQUIRED_MODELS + config.OPTIONAL_MODELS if os.path.isfile(os.path.join(d, n))]
        ok(f"{d}  ({', '.join(have)})")
        return d
    warn("required models not found: " + ", ".join(config.REQUIRED_MODELS))
    dest = os.path.join(config.HERE, "models")
    if ARGS.download_models or ask(f"Download them into {dest}? (streams ~1.8 GB, keeps ~100 MB)"):
        try:
            if download_models(dest):
                ok(f"models saved to {dest}")
                return dest
        except Exception as e:
            fail(f"download failed: {e}")
    say("     Get them with DEEPX's dx_app/setup_sample_models.sh, or pass --model-dir /path/to/models-2_2_0")
    sys.exit(1)


def step_selftest(py):
    say("\n5. Self-test on the NPU")
    r = subprocess.run([py, os.path.join(config.HERE, "server.py"), "--self-test"],
                       capture_output=True, text=True, timeout=300, cwd=config.HERE)
    try:
        rep = json.loads(r.stdout[r.stdout.index("{"):])
    except ValueError:
        fail("self-test crashed:\n" + (r.stdout + r.stderr)[-2000:])
        return False
    for mid, m in rep.get("modes", {}).items():
        if m.get("ok"):
            ok(f"{mid:9s} NPU {m['npu_ms']:6.1f} ms")
        else:
            fail(f"{mid:9s} {m.get('error')}")
    return rep.get("ok", False)


def write_launchers(py):
    if config.IS_WINDOWS:
        path = os.path.join(config.HERE, "run-server.bat")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f'@echo off\r\ncd /d "{config.HERE}"\r\n"{py}" server.py %*\r\n')
    else:
        path = os.path.join(config.HERE, "run-server.sh")
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"#!/bin/sh\ncd {shlex.quote(config.HERE)}\nexec {shlex.quote(py)} server.py \"$@\"\n")
        os.chmod(path, 0o755)
    return path


def step_service(py, launcher):
    say("\n6. Run at boot")
    if config.IS_WINDOWS:
        if ARGS.service or (ARGS.service is None and ask("Create a logon task (Task Scheduler) to start the server?", False)):
            run(["schtasks", "/Create", "/F", "/SC", "ONLOGON", "/TN", "DX-M1 Vision", "/TR", launcher])
            ok("task 'DX-M1 Vision' created; Windows Firewall may ask to allow Python on first run")
        else:
            say(f"  Start manually: {launcher}")
        return
    if not shutil.which("systemctl") or not os.path.isdir("/run/systemd/system"):
        say(f"  No systemd; start manually: {launcher}")
        return
    if not (ARGS.service or (ARGS.service is None and ask("Install a systemd service so it starts at boot?"))):
        say(f"  Start manually: {launcher}")
        return
    env = ""
    if os.environ.get("LD_LIBRARY_PATH"):
        env = f"Environment=LD_LIBRARY_PATH={os.environ['LD_LIBRARY_PATH']}\n"
    unit = f"""[Unit]
Description=DX-M1 Vision phone app (DEEPX DX-M1 detection and pose)
After=network-online.target
Wants=network-online.target

[Service]
User={getpass.getuser()}
WorkingDirectory={config.HERE}
Environment=PYTHONUNBUFFERED=1
{env}ExecStart={py} {os.path.join(config.HERE, 'server.py')}
Restart=on-failure
RestartSec=3

[Install]
WantedBy=multi-user.target
"""
    tmp = os.path.join(config.HERE, "dx-m1-vision.service")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(unit)
    sudo = [] if os.geteuid() == 0 else ["sudo"]
    steps = [
        sudo + ["install", "-m", "644", tmp, "/etc/systemd/system/dx-m1-vision.service"],
        sudo + ["systemctl", "daemon-reload"],
        sudo + ["systemctl", "enable", "dx-m1-vision"],
        sudo + ["systemctl", "restart", "dx-m1-vision"],
    ]
    for s in steps:
        if run(s).returncode != 0:
            fail("service install failed; start manually: " + launcher)
            return
    os.remove(tmp)
    ok("service dx-m1-vision enabled and started (logs: journalctl -u dx-m1-vision -f)")
    if shutil.which("ufw"):
        say(f"  If ufw is active:       sudo ufw allow {ARGS.port}/tcp")
    if shutil.which("firewall-cmd"):
        say(f"  If firewalld is active: sudo firewall-cmd --add-port={ARGS.port}/tcp --permanent && sudo firewall-cmd --reload")


def main():
    global ARGS
    ap = argparse.ArgumentParser(description="Install DX-M1 Vision on this machine.")
    ap.add_argument("--python", help="Python interpreter that has dx_engine (default: auto-detect)")
    ap.add_argument("--model-dir", help="folder with the YOLO26 .dxnn files (default: auto-detect)")
    ap.add_argument("--download-models", action="store_true", help="download models without asking if missing")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0", help="bind address (default: all interfaces)")
    svc = ap.add_mutually_exclusive_group()
    svc.add_argument("--service", dest="service", action="store_true", default=None, help="install the boot service")
    svc.add_argument("--no-service", dest="service", action="store_false", help="don't install a service")
    ap.add_argument("-y", "--yes", action="store_true", help="accept defaults without prompting")
    ARGS = ap.parse_args()

    say(f"DX-M1 Vision installer  ({platform.system()} {platform.machine()}, {config.HERE})")
    info = step_python()
    if info:
        step_cv(info)
    step_device()
    ff, enc = step_ffmpeg(offer_install=bool(info))
    if not info:
        say("\nStopped: DX-RT (dx_engine) is required. Fix the items above and rerun install.py.")
        sys.exit(1)
    py = info["exe"]
    model_dir = step_models()

    cfg = config.load_file_config()
    cfg.update({"python": py, "model_dir": model_dir, "ffmpeg": ff, "h264_encoder": enc,
                "port": ARGS.port, "host": ARGS.host})
    config.save_file_config(cfg)
    ok(f"wrote {config.CONFIG_PATH}")

    if not step_selftest(py):
        say("\nSelf-test failed; fix the errors above and rerun install.py.")
        sys.exit(1)
    launcher = write_launchers(py)
    step_service(py, launcher)

    say("\nDone. On your phone (same network) open:")
    for h in config.lan_addresses():
        say(f"  http://{h}:{ARGS.port}")


if __name__ == "__main__":
    main()
