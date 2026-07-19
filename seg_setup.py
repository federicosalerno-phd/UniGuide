"""
UniGuide segmentation environment bootstrap.

Creates the dedicated segmentation Python environment on the end user's machine and
installs everything the automatic segmentation needs (torch, MOOSE, and the imaging /
meshing stack). Model weights are NOT downloaded here; MOOSE fetches them on first use.

Run by the app (through a base Python it provides) as:

    python seg_setup.py <target_env_dir> [--base-python <path>] [--force-cpu] [--force-cuda]

Progress is streamed to stderr (STAGE / step lines the UI shows live); the final JSON
summary is printed on stdout:

    {"ok": true, "python": "<venv python>", "device": "cuda"|"cpu"}

On success a pointer file is written to %LOCALAPPDATA%/UniGuide/seg_python.txt so the app
finds the environment on the next launch.
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from pathlib import Path

# torch build pinned to the versions validated on the dev machine (T2000, CUDA 12.8).
CUDA_INDEX = "https://download.pytorch.org/whl/cu128"
CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CUDA_TORCH = ["torch==2.11.0", "torchvision==0.26.0"]
CPU_TORCH = ["torch", "torchvision"]

# the rest of the segmentation stack (moosez pulls nnunetv2 + numpy + networkx + pydicom)
STACK = ["moosez", "SimpleITK", "scikit-image", "scipy", "trimesh", "rtree"]


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def stage(msg):
    log("STAGE:", msg)


def _user_data_dir():
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    d = Path(base) / "UniGuide"
    d.mkdir(parents=True, exist_ok=True)
    return d


def detect_cuda():
    """True if an NVIDIA GPU looks usable (nvidia-smi present and lists a device)."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        out = subprocess.run([exe, "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def run(cmd, env=None):
    """Run a command, streaming its output to stderr; raise on non-zero exit."""
    log("$", " ".join(str(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, bufsize=1)
    for line in proc.stdout:
        line = line.rstrip()
        if line:
            log("  " + line[:200])
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("command failed (%d): %s" % (proc.returncode, " ".join(map(str, cmd))))


def venv_python(env_dir):
    p = Path(env_dir)
    win = p / "Scripts" / "python.exe"
    return str(win if os.name == "nt" else p / "bin" / "python")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--base-python", default=sys.executable)
    ap.add_argument("--force-cpu", action="store_true")
    ap.add_argument("--force-cuda", action="store_true")
    args = ap.parse_args(argv)

    target = os.path.abspath(args.target)
    cache = os.environ.get("PIP_CACHE_DIR") or str(_user_data_dir() / "pip_cache")
    env = dict(os.environ)
    env["PIP_CACHE_DIR"] = cache

    use_cuda = (not args.force_cpu) and (args.force_cuda or detect_cuda())
    device = "cuda" if use_cuda else "cpu"
    stage("GPU detected, installing CUDA build" if use_cuda else "no GPU, installing CPU build")

    py = venv_python(target)
    if not Path(py).exists():
        stage("creating environment at %s" % target)
        run([args.base_python, "-m", "venv", target])

    stage("upgrading pip")
    run([py, "-m", "pip", "install", "-q", "--upgrade", "pip", "setuptools<82", "wheel"], env=env)

    stage("installing PyTorch (%s), this is the big download" % device)
    if use_cuda:
        run([py, "-m", "pip", "install", "--index-url", CUDA_INDEX] + CUDA_TORCH, env=env)
    else:
        run([py, "-m", "pip", "install", "--index-url", CPU_INDEX] + CPU_TORCH, env=env)

    stage("installing MOOSE and the imaging stack")
    run([py, "-m", "pip", "install"] + STACK, env=env)

    stage("verifying")
    check = ("import torch, moosez, SimpleITK, skimage, scipy, trimesh, rtree;"
             "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())")
    run([py, "-c", check], env=env)

    # remember where the environment lives so the app finds it next launch
    ptr = _user_data_dir() / "seg_python.txt"
    ptr.write_text(py, encoding="utf-8")
    stage("done")
    print(json.dumps({"ok": True, "python": py, "device": device}))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1:]))
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        print(json.dumps({"ok": False, "error": "%s: %s" % (type(e).__name__, e)}))
        sys.exit(1)
