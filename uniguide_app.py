"""UniGuide -- modular cutting-guide composer.

A desktop tool to compose **standardized, partially reusable mandibular cutting
guides** and to auto-generate the patient-specific connector that locks the
reusable modules into a single rigid body. A Qt window hosts an embedded web UI
(``ui.html``) that talks to this Python backend over ``QWebChannel`` and walks
the user through:

* **Model** -- load the mandible STL (in the clinical workflow it already
  carries the resection cutting planes from the virtual surgical plan).
* **Planes** -- inspect/define the two cutting planes (mandibular sides).
* **Guides** -- place the two unified modules so each cutting face coincides
  with a plane; one module carries two support feet, the other one, giving a
  determinate three-point contact once they are bridged.
* **Connector** -- auto-generate the bridge from a guided spline and a chosen
  cross-section profile (the only patient-specific, printed part).
* **Check** -- kinematic checks (three feet seated, clearance, faces coincident).
* **Export** -- write a print-ready STL (Formlabs resin workflow), either the
  connector alone or the full assembly.

The reusable left/right modules come from a small size library and are meant to
be resterilized between cases; only the connector is fabricated per patient.

Run with ``python uniguide_app.py`` (or ``launch.bat``).
"""
import os, sys, json, struct, shutil

# ── Interpreter self-bootstrap ──────────────────────────────────────────────────
# UniGuide needs PyQt6 + PyQt6-WebEngine (installed in the arm-controller virtualenv,
# together with numpy/trimesh/manifold3d for the welded-guide export). If this script is
# started with a Python that lacks them — e.g. the IDE's ▶ Run button using the SYSTEM
# Python, giving "ModuleNotFoundError: No module named 'PyQt6'" — we silently RE-LAUNCH
# ourselves with a Python that HAS Qt, so the app "just works" no matter which interpreter
# is selected. Force a specific one with the UNIGUIDE_PYTHON env var.
def _qt_ok():
    try:
        import PyQt6.QtWebEngineWidgets  # noqa: F401
        return True
    except Exception:
        try:
            import PyQt5.QtWebEngineWidgets  # noqa: F401
            return True
        except Exception:
            return False

if (__name__ == "__main__" and not getattr(sys, "frozen", False)
        and not os.environ.get("UNIGUIDE_BOOTSTRAPPED") and not _qt_ok()):
    # (a PyInstaller .exe bundles Qt, so it must NEVER re-exec to another interpreter)
    import glob as _glob
    import subprocess as _sp
    _here = os.path.abspath(__file__)
    _me = os.path.abspath(sys.executable)
    _cands = []
    if os.environ.get("UNIGUIDE_PYTHON"):
        _cands.append(os.environ["UNIGUIDE_PYTHON"])
    _cands.append(os.path.expanduser(os.path.join("~", ".virtualenvs",
                  "arm-controller-I0EXsfi0", "Scripts", "python.exe")))
    _cands += sorted(_glob.glob(os.path.expanduser(os.path.join(
                  "~", ".virtualenvs", "*", "Scripts", "python.exe"))))
    for _py in _cands:
        if not _py or not os.path.isfile(_py) or os.path.abspath(_py) == _me:
            continue
        try:  # only hand off to a Python that actually has Qt (avoids relaunch loops)
            _chk = _sp.run([_py, "-c", "import PyQt6.QtWebEngineWidgets"],
                           capture_output=True, timeout=40)
        except Exception:
            continue
        if _chk.returncode != 0:
            continue
        sys.stderr.write("UniGuide: the selected Python has no PyQt6 -> relaunching with\n"
                         "          %s\n" % _py)
        _env = dict(os.environ, UNIGUIDE_BOOTSTRAPPED="1")
        sys.exit(_sp.call([_py, _here] + sys.argv[1:], env=_env))
    # No Qt-capable Python found → fall through; the import below raises a clear message.

# Rendering backend for the embedded web view.
#   True  -> GPU acceleration: smooth WebGL 3D. Recommended.
#   False -> software rendering: use ONLY if the window comes up black/empty.
USE_GPU = True
if not USE_GPU:
    os.environ.setdefault("QT_OPENGL", "software")

from pathlib import Path

# Qt binding: prefer PyQt6 (modern Chromium compositor, better WebGL), fall back
# to PyQt5 + PyQtWebEngine so the app still runs if PyQt6 is not installed yet.
#   To get the preferred path:  pip install PyQt6 PyQt6-WebEngine
try:
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    from PyQt6.QtWebEngineCore import QWebEngineSettings          # moved module in Qt6
    from PyQt6.QtWebChannel import QWebChannel
    from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal, QUrl, Qt, QProcess, QProcessEnvironment
    from PyQt6.QtGui import QColor
    USE_QT6 = True
except Exception:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
    from PyQt5.QtWebChannel import QWebChannel
    from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QUrl, Qt, QProcess, QProcessEnvironment
    from PyQt5.QtGui import QColor
    USE_QT6 = False

APP_NAME = "UniGuide"
APP_VERSION = "0.1.5"


def _res_dir():
    """Folder holding the bundled, read-only data files (ui.html, libs/, modules/,
    input/, guides.json). When packaged with PyInstaller the data is unpacked to
    ``sys._MEIPASS``; running from source it sits next to this file. Everything that
    reads a shipped asset MUST go through here so the frozen .exe finds it."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", "") or Path(sys.executable).parent)
    return Path(__file__).parent


def _user_data_dir():
    """A user-writable folder for logs/scratch (the bundle itself may be read-only)."""
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    d = Path(base) / "UniGuide"
    try:
        d.mkdir(parents=True, exist_ok=True)
    except Exception:
        d = Path(os.path.abspath("."))
    return d


# ── STL I/O ───────────────────────────────────────────────────────────────────
def _parse_stl(path):
    """Parse a binary or ASCII STL into a flat triangle-soup position list.

    Returns ``[ax, ay, az, bx, by, bz, cx, cy, cz, ...]`` (9 floats per facet),
    ready to drop straight into a Three.js ``Float32BufferAttribute``. Binary vs
    ASCII is detected from the file size against the 84 + 50*n binary layout.
    """
    raw = Path(path).read_bytes()
    is_binary = True
    if len(raw) >= 84:
        n = struct.unpack_from("<I", raw, 80)[0]
        is_binary = (len(raw) == 84 + n * 50)
    else:
        is_binary = False
    if not is_binary and raw[:5].lower().lstrip().startswith(b"solid"):
        return _parse_ascii_stl(raw.decode("utf-8", "replace"))
    if not is_binary:
        # Header lied but it is not valid ASCII either: try ASCII as a last resort.
        try:
            return _parse_ascii_stl(raw.decode("utf-8", "replace"))
        except Exception:
            return []
    n = struct.unpack_from("<I", raw, 80)[0]
    pos = []
    off = 84
    for _ in range(n):
        # 12 floats per facet (normal + 3 vertices), we keep the 9 vertex floats.
        vals = struct.unpack_from("<12f", raw, off)
        pos.extend(vals[3:12])
        off += 50
    return pos


def _parse_ascii_stl(text):
    pos = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("vertex"):
            _, x, y, z = line.split()[:4]
            pos.extend((float(x), float(y), float(z)))
    return pos


# ── backend ───────────────────────────────────────────────────────────────────
class Backend(QObject):
    """Python side of the QWebChannel bridge.

    Only the operations that genuinely need the host (native file dialogs,
    reading/writing files on disk) live here; all geometry and rendering stays
    in ``ui.html``. Add new ``@pyqtSlot`` methods below as the app grows.
    """

    modelLoaded = pyqtSignal(str)   # emitted with the loaded model JSON
    segEvent = pyqtSignal(str)      # segmentation progress/done/error events (JSON)

    @staticmethod
    def _input_dir():
        return _res_dir() / "input"

    def __init__(self):
        super().__init__()
        d = self._input_dir()
        self._last_dir = str(d if d.exists() else Path.home())
        self._seg_procs = []        # keep QProcess refs alive while running
        self._edit = None           # in-memory mask-editing session (bundle loaded)

    @pyqtSlot(result=str)
    def app_info(self):
        """Return basic app metadata (used by the UI title/status)."""
        return json.dumps({"name": APP_NAME, "version": APP_VERSION,
                           "qt": "PyQt6" if USE_QT6 else "PyQt5"})

    def _start_dir(self):
        """Dialogs open in the ``input/`` folder (next to the app) when present."""
        d = self._input_dir()
        return str(d) if d.exists() else self._last_dir

    @pyqtSlot(str, result=str)
    def read_input(self, fname):
        """Parse a demo STL from the ``input/`` folder (the demo samples)."""
        if not fname or ("/" in fname or "\\" in fname or ".." in fname):
            return json.dumps({"error": "bad name"})
        p = self._input_dir() / fname
        if not p.exists():
            return json.dumps({"error": "not found: " + fname})
        try:
            pos = _parse_stl(str(p))
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"name": p.name, "positions": pos, "count": len(pos) // 9})

    @pyqtSlot(result=str)
    def open_model(self):
        """Open a native dialog, parse the chosen STL and return it as JSON.

        Returns ``{"name": ..., "positions": [...], "count": n}`` or ``""`` if
        the dialog is cancelled. ``positions`` is a flat xyz triangle soup.
        """
        path, _ = QFileDialog.getOpenFileName(
            None, "Open mandible STL", self._start_dir(), "STL meshes (*.stl);;All files (*)")
        if not path:
            return ""
        self._last_dir = str(Path(path).parent)
        try:
            pos = _parse_stl(path)
        except Exception as e:
            return json.dumps({"error": str(e)})
        out = json.dumps({"name": Path(path).name, "positions": pos,
                         "count": len(pos) // 9})
        self.modelLoaded.emit(out)
        return out

    @pyqtSlot(result=str)
    def open_fibula(self):
        """Open a native dialog and parse a fibula STL (donor bone for reconstruction)."""
        path, _ = QFileDialog.getOpenFileName(
            None, "Open fibula STL", self._start_dir(), "STL meshes (*.stl);;All files (*)")
        if not path:
            return ""
        self._last_dir = str(Path(path).parent)
        try:
            pos = _parse_stl(path)
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"name": Path(path).name, "positions": pos, "count": len(pos) // 9})

    @pyqtSlot(result=str)
    def open_teeth(self):
        """Open a native dialog and parse a TEETH STL — a visual reference the user aligns
        manually to the mandible in the guide steps (not the splint; alignment only)."""
        path, _ = QFileDialog.getOpenFileName(
            None, "Open teeth STL", self._start_dir(), "STL meshes (*.stl);;All files (*)")
        if not path:
            return ""
        self._last_dir = str(Path(path).parent)
        try:
            pos = _parse_stl(path)
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"name": Path(path).name, "positions": pos, "count": len(pos) // 9})

    # -------------------------------------------------------------- segmentation
    # The heavy DICOM segmentation (MOOSE / nnU-Net / torch) lives in a dedicated
    # Python environment and is driven through ``segmentation.py`` as a subprocess,
    # so the app itself stays light. Long-running commands run async via QProcess and
    # report back through the ``segEvent`` signal; the UI shows one unified panel.

    def _seg_python(self):
        """Locate the segmentation environment's Python interpreter.

        Order: ``UNIGUIDE_SEG_PYTHON`` env var, a ``seg_python.txt`` pointer saved in
        the user-data dir, then a couple of common dev locations. Empty if none found.
        """
        cand = os.environ.get("UNIGUIDE_SEG_PYTHON", "").strip()
        if cand and Path(cand).exists():
            return cand
        try:
            ptr = _user_data_dir() / "seg_python.txt"
            if ptr.exists():
                t = ptr.read_text(encoding="utf-8").strip()
                if t and Path(t).exists():
                    return t
        except Exception:
            pass
        for c in (r"D:/uniguide_seg/Scripts/python.exe",
                  str(Path.home() / "uniguide_seg" / "Scripts" / "python.exe")):
            if Path(c).exists():
                return c
        return ""

    def _seg_script(self):
        return str(_res_dir() / "segmentation.py")

    @pyqtSlot(result=str)
    def seg_status(self):
        """Report whether the segmentation environment is ready."""
        py = self._seg_python()
        return json.dumps({"available": bool(py) and Path(self._seg_script()).exists(),
                           "python": py, "script": self._seg_script()})

    def _cases_dir(self):
        """Where the patient DICOM cases live. Prefer a fast LOCAL copy, then the env override,
        then a Tests folder next to the app. Reading DICOM off a network drive is slow, so the
        local copy (made once) is opened first."""
        for c in (os.environ.get("UNIGUIDE_CASES_DIR", "").strip(),
                  r"D:/UniGuide_Clean", r"D:/UniGuide_Patients",
                  str(Path.home() / "UniGuide_Patients")):
            if c and Path(c).is_dir():
                return c
        cand = _res_dir().parent / "Tests"
        return str(cand) if cand.is_dir() else ""

    @pyqtSlot(result=str)
    def seg_pick_dicom_dir(self):
        """Native dialog to choose the patient folder; opens in the cases/Tests folder."""
        start = getattr(self, "_seg_last_dir", "") or self._cases_dir() or self._start_dir()
        d = QFileDialog.getExistingDirectory(None, "Choose the patient folder (DICOM)", start)
        if d:
            self._seg_last_dir = d
        return d or ""

    def _spawn(self, python, argv, cmd_tag):
        """Run ``python argv...`` async, streaming stderr as progress and the final
        stdout line as the result, both over ``segEvent``."""
        if not python:
            self.segEvent.emit(json.dumps({"kind": "error", "cmd": cmd_tag,
                                           "error": "No Python interpreter available"}))
            return
        proc = QProcess()
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        proc.setProcessEnvironment(env)
        buf = {"out": b"", "err": b""}

        def on_out():
            buf["out"] += bytes(proc.readAllStandardOutput())

        def _emit_err_lines(final=False):
            # A single reveal SLICE line (sharp native-CT PNG, base64) is far larger than the OS pipe
            # buffer, so QProcess delivers it across several reads. Accumulate and emit only COMPLETE
            # (newline-terminated) lines, keeping the trailing partial for the next read; otherwise a
            # long SLICE gets chopped into a truncated line + orphan tail and the UI silently drops it
            # (patchy leg reveal). On finish, flush whatever remains.
            data = buf["err"].replace(b"\r", b"\n")
            parts = data.split(b"\n")
            rest = b"" if final else parts.pop()          # last piece has no newline yet: hold it back (unless final)
            buf["err"] = rest
            for lb in parts:
                ln = lb.decode("utf-8", "replace").strip()
                if ln:
                    self.segEvent.emit(json.dumps({"kind": "progress", "cmd": cmd_tag, "text": ln}))

        def on_err():
            buf["err"] += bytes(proc.readAllStandardError())
            _emit_err_lines()

        def on_fin(code, _status):
            buf["err"] += bytes(proc.readAllStandardError())   # drain any tail QProcess still holds
            _emit_err_lines(final=True)
            out = buf["out"].decode("utf-8", "replace")
            line = ""
            for ln in out.splitlines():
                if ln.startswith("UNIGUIDE_RESULT "):
                    line = ln[len("UNIGUIDE_RESULT "):]
            if not line:
                nonblank = [l for l in out.splitlines() if l.strip()]
                line = nonblank[-1] if nonblank else ""
            self.segEvent.emit(json.dumps({"kind": "done", "cmd": cmd_tag,
                                           "code": int(code), "result": line}))
            if proc in self._seg_procs:
                self._seg_procs.remove(proc)

        proc.readyReadStandardOutput.connect(on_out)
        proc.readyReadStandardError.connect(on_err)
        proc.finished.connect(on_fin)
        self._seg_procs.append(proc)
        proc.start(python, list(argv))

    def _run_seg_async(self, args, cmd_tag):
        """Start segmentation.py <args> in the seg env; stream events over segEvent."""
        py = self._seg_python()
        if not py:
            self.segEvent.emit(json.dumps({"kind": "error", "cmd": cmd_tag,
                                           "error": "Segmentation environment not configured"}))
            return
        self._spawn(py, [self._seg_script()] + list(args), cmd_tag)

    # ---- first-run bootstrap: install the segmentation environment on demand ----

    def _base_python(self):
        """A Python that can create the seg venv. In dev this is the app's own; frozen,
        a bundled standalone under the app, else a system Python on PATH."""
        cand = os.environ.get("UNIGUIDE_BASE_PYTHON", "").strip()
        if cand and Path(cand).exists():
            return cand
        if not getattr(sys, "frozen", False):
            return sys.executable
        bundled = _res_dir() / "pybase" / ("python.exe" if os.name == "nt" else "bin/python")
        if bundled.exists():
            return str(bundled)
        for name in ("py", "python", "python3"):
            w = shutil.which(name)
            if w:
                return w
        return ""

    def _seg_setup_script(self):
        return str(_res_dir() / "seg_setup.py")

    @pyqtSlot(str)
    def seg_setup(self, force):
        """Create/install the segmentation environment (async; events via segEvent,
        cmd 'setup'). ``force`` is '', 'cpu' or 'cuda'."""
        target = str(_user_data_dir() / "segenv")
        argv = [self._seg_setup_script(), target]
        if force == "cpu":
            argv.append("--force-cpu")
        elif force == "cuda":
            argv.append("--force-cuda")
        self._spawn(self._base_python(), argv, "setup")

    @pyqtSlot(str)
    def seg_list_series(self, folder):
        """List DICOM series in a folder (async; result via segEvent, cmd 'list-series')."""
        self._run_seg_async(["list-series", folder], "list-series")

    @pyqtSlot(str, str, str)
    def seg_run(self, folder, series_id, region):
        """Segment one series for a region (async; result via segEvent, cmd 'segment')."""
        work = str(_user_data_dir() / "seg_work")
        self._run_seg_async(["segment", folder, series_id, region, work], "segment")

    @pyqtSlot(str, str)
    def seg_ctbundle(self, folder, series_id):
        """Prepare a chosen series for manual drawing, empty mask (async, cmd 'ctbundle')."""
        work = str(_user_data_dir() / "seg_work_manual")
        self._run_seg_async(["ctbundle", folder, series_id, work], "ctbundle")

    @pyqtSlot(str, result=str)
    def seg_read_stl(self, path):
        """Parse an STL the segmentation produced and return its triangle-soup positions as base64
        float32. A raw JSON array of ~750k numbers is a ~10 MB string the browser must JSON.parse
        (hundreds of ms of frozen UI right before the model appeared, which read as an abrupt pop);
        the compact binary decodes in ~50 ms so the finished model can GROW in smoothly instead."""
        p = Path(path)
        if not p.exists():
            return json.dumps({"error": "not found: " + path})
        try:
            pos = _parse_stl(str(p))
        except Exception as e:
            return json.dumps({"error": str(e)})
        import numpy as _np, base64 as _b64
        b = _b64.b64encode(_np.asarray(pos, dtype=_np.float32).tobytes()).decode("ascii")
        return json.dumps({"name": p.name, "b64": b, "count": len(pos) // 9})

    # ------------------------------------------------------------ mask editing
    # Hand-correct or draw the segmentation on axial/coronal/sagittal slices. The CT
    # and the label mask live here as numpy arrays (loaded from the seg bundle); the
    # browser shows slices and paints, edits come back, then we re-mesh via the seg env.

    @staticmethod
    def _slice(arr, axis, idx):
        if axis == 0:
            return arr[idx]
        if axis == 1:
            return arr[:, idx, :]
        return arr[:, :, idx]

    @pyqtSlot(str, result=str)
    def edit_load(self, bundle_path):
        """Load a segmentation bundle (CT + label mask + geometry) for editing."""
        import numpy as np
        p = Path(bundle_path)
        if not p.exists():
            return json.dumps({"ok": False, "error": "bundle not found: " + bundle_path})
        try:
            d = np.load(str(p), allow_pickle=True)
            self._edit = {"path": str(p), "ct": d["ct"], "labels": np.ascontiguousarray(d["labels"]),
                          "spacing": d["spacing"], "names": json.loads(str(d["names"]))}
        except Exception as e:
            return json.dumps({"ok": False, "error": str(e)})
        nz, ny, nx = self._edit["ct"].shape
        return json.dumps({"ok": True, "nz": int(nz), "ny": int(ny), "nx": int(nx),
                           "names": self._edit["names"]})

    @pyqtSlot(int, int, float, float, result=str)
    def edit_ct_slice(self, axis, idx, level, width):
        """Return one CT slice, windowed to grayscale, as a PNG data URL."""
        import numpy as np, io, base64
        from PIL import Image
        if not self._edit:
            return ""
        sl = self._slice(self._edit["ct"], axis, int(idx)).astype(np.float32)
        lo, hi = level - width / 2.0, level + width / 2.0
        g = np.clip((sl - lo) / max(1e-3, hi - lo), 0, 1) * 255.0
        buf = io.BytesIO()
        Image.fromarray(g.astype(np.uint8), mode="L").save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    @pyqtSlot(int, int, result=str)
    def edit_mask_slice(self, axis, idx):
        """Return one label-mask slice as raw base64 uint8 (the UI colours it)."""
        import numpy as np, base64
        if not self._edit:
            return json.dumps({"ok": False})
        sl = np.ascontiguousarray(self._slice(self._edit["labels"], axis, int(idx)).astype(np.uint8))
        h, w = sl.shape
        return json.dumps({"ok": True, "h": int(h), "w": int(w),
                           "data": base64.b64encode(sl.tobytes()).decode()})

    @pyqtSlot(int, int, str, result=str)
    def edit_set_slice(self, axis, idx, b64):
        """Write an edited mask slice back into the volume."""
        import numpy as np, base64
        if not self._edit:
            return json.dumps({"ok": False})
        lab = self._edit["labels"]
        idx = int(idx)
        shape = self._slice(lab, axis, idx).shape
        arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8).reshape(shape)
        if axis == 0:
            lab[idx] = arr
        elif axis == 1:
            lab[:, idx, :] = arr
        else:
            lab[:, :, idx] = arr
        return json.dumps({"ok": True})

    @pyqtSlot(int, int, int, int, int, float, result=str)
    def edit_region_grow(self, axis, idx, x, y, label, tol):
        """Flood-fill from a seed on the CT slice: the connected region of voxels whose HU
        is within ``tol`` of the seed gets ``label``. Returns the updated mask slice."""
        import numpy as np, base64
        from scipy import ndimage
        if not self._edit:
            return json.dumps({"ok": False})
        axis, idx, x, y, label = int(axis), int(idx), int(x), int(y), int(label)
        ct = self._slice(self._edit["ct"], axis, idx).astype(np.float32)
        h, w = ct.shape
        if not (0 <= y < h and 0 <= x < w):
            return json.dumps({"ok": False})
        seed = ct[y, x]
        band = np.abs(ct - seed) <= float(tol)
        lab, n = ndimage.label(band)
        comp = lab[y, x]
        if comp == 0:
            return json.dumps({"ok": False})
        region = (lab == comp)
        msl = self._slice(self._edit["labels"], axis, idx)
        msl = msl.copy()
        msl[region] = label
        # write back
        laba = self._edit["labels"]
        if axis == 0:
            laba[idx] = msl
        elif axis == 1:
            laba[:, idx, :] = msl
        else:
            laba[:, :, idx] = msl
        sl = np.ascontiguousarray(msl.astype(np.uint8))
        return json.dumps({"ok": True, "h": int(h), "w": int(w),
                           "data": base64.b64encode(sl.tobytes()).decode()})

    @pyqtSlot(int, str)
    def edit_remesh(self, label, out_dir):
        """Persist the edited mask and re-mesh one label via the seg env (async, cmd 'remesh')."""
        import numpy as np
        if not self._edit:
            self.segEvent.emit(json.dumps({"kind": "error", "cmd": "remesh", "error": "no edit session"}))
            return
        bundle = self._edit["path"]
        np.save(str(Path(bundle).parent / "labels_edited.npy"), self._edit["labels"])
        out = str(Path(out_dir or str(Path(bundle).parent)) / ("edited_%d.stl" % int(label)))
        self._spawn(self._seg_python(), [self._seg_script(), "remesh", bundle, str(int(label)), out], "remesh")

    @pyqtSlot(str, str, result=str)
    def save_stl(self, suggested_name, stl_text):
        """Write an STL produced in the browser to a user-chosen path.

        ``stl_text`` is the full ASCII-STL string built in ``ui.html`` (the
        connector, or the full assembly). Returns the saved path or ``""``.
        """
        suggested = suggested_name or "uniguide_export.stl"
        start = str(Path(self._last_dir) / suggested)
        path, _ = QFileDialog.getSaveFileName(
            None, "Export STL", start, "STL mesh (*.stl);;All files (*)")
        if not path:
            return ""
        if not path.lower().endswith(".stl"):
            path += ".stl"
        try:
            Path(path).write_text(stl_text, encoding="utf-8")
        except Exception as e:
            return json.dumps({"error": str(e)})
        self._last_dir = str(Path(path).parent)
        return path

    @pyqtSlot(str, result=str)
    def weld_fibula_guides(self, payload_json):
        """Boolean-weld each fibula cutting guide (truncated body + its rim/sponda
        extrusions) into ONE watertight solid and save a SEPARATE print-ready STL
        per guide.

        The browser cannot do a robust manifold boolean, so the heavy lifting is
        done here with ``manifold3d`` (guaranteed-manifold CSG). ``payload_json``:
            {"base": "uniguide_fib_guide",
             "guides": [ {"gi": 0,
                          "guide": <base64 Float32 world triangle-soup of the FULL
                                    (untruncated) guide geometry>,
                          "planes": [[nx,ny,nz,constant], [nx,ny,nz,constant]],
                          "rims":  [<base64 Float32 world triangle-soup>, ...] },
                         ... ] }
        For each guide: build a manifold from the full guide, TRIM it by the two
        cut planes (keep n.p + constant >= 0, i.e. the segment side), then UNION
        every rim. One save dialog picks the base path; files are written as
        ``<base>_1.stl``, ``<base>_2.stl`` ... (one per guide). Returns JSON with
        the saved paths + per-guide watertight/volume, or an ``error``.
        """
        import base64
        try:
            import numpy as np
            import trimesh
            import manifold3d as m3d
        except Exception as e:
            return json.dumps({"error": "missing Python libs (need numpy, trimesh, manifold3d): " + str(e)})
        try:
            data = json.loads(payload_json)
        except Exception as e:
            return json.dumps({"error": "bad payload: " + str(e)})
        guides = data.get("guides", [])
        if not guides:
            return json.dumps({"error": "no guides in payload"})

        base = data.get("base") or "uniguide_fib_guide"
        start = str(Path(self._last_dir) / (base + ".stl"))
        path, _ = QFileDialog.getSaveFileName(
            None, "Export welded fibula guides (one STL per guide)", start,
            "STL mesh (*.stl);;All files (*)")
        if not path:
            return json.dumps({"cancelled": True})
        stem = path[:-4] if path.lower().endswith(".stl") else path
        self._last_dir = str(Path(path).parent)

        def decode(b):
            return np.frombuffer(base64.b64decode(b), dtype=np.float32).reshape(-1, 3).astype(np.float64)

        def to_mesh(pts):
            # weld a triangle-soup (3T,3) into a clean indexed mesh (0.1 micron grid)
            key = np.round(pts * 1e4).astype(np.int64)
            uniq, inv = np.unique(key, axis=0, return_inverse=True)
            V = np.zeros((len(uniq), 3)); np.add.at(V, inv, pts); V /= np.bincount(inv)[:, None]
            return trimesh.Trimesh(vertices=V, faces=inv.reshape(-1, 3), process=False)

        def mani(mesh):
            return m3d.Manifold(m3d.Mesh(
                vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))

        saved, errors = [], []
        for g in guides:
            gi = g.get("gi", 0)
            try:
                gm = to_mesh(decode(g["guide"]))
                if not gm.is_watertight:      # library STL should be closed; repair defensively
                    gm.fill_holes(); gm.fix_normals()
                M = mani(gm)
                for pl in g.get("planes", []):
                    M = M.trim_by_plane([pl[0], pl[1], pl[2]], -pl[3])   # keep n.p + constant >= 0
                for rb in g.get("rims", []):
                    r = to_mesh(decode(rb)); r.fix_normals()
                    M = M + mani(r)
                mesh = M.to_mesh()
                V = np.asarray(mesh.vert_properties)[:, :3].astype(np.float64)
                F = np.asarray(mesh.tri_verts).astype(np.int64)
                tm = trimesh.Trimesh(vertices=V, faces=F, process=False)   # manifold3d output is already clean-indexed
                outp = "%s_%d.stl" % (stem, int(gi) + 1)
                tm.export(outp)
                saved.append({"gi": gi, "path": outp, "watertight": bool(tm.is_watertight),
                              "volume": round(float(tm.volume), 1), "tris": int(len(tm.faces)),
                              "bodies": int(tm.body_count)})
            except Exception as e:
                errors.append({"gi": gi, "error": str(e)})
        return json.dumps({"saved": saved, "errors": errors, "dir": self._last_dir})

    @pyqtSlot(str, str, result=str)
    def save_state(self, suggested_name, json_text):
        """Write the UI session state (planes, tumour, orientation, axis, guides)
        to a user-chosen ``.json`` file. Returns the saved path or ``""``."""
        suggested = suggested_name or "uniguide_session.json"
        start = str(Path(self._last_dir) / suggested)
        path, _ = QFileDialog.getSaveFileName(
            None, "Save session", start, "UniGuide session (*.json);;All files (*)")
        if not path:
            return ""
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            Path(path).write_text(json_text, encoding="utf-8")
        except Exception as e:
            return json.dumps({"error": str(e)})
        self._last_dir = str(Path(path).parent)
        return path

    @pyqtSlot(result=str)
    def load_state(self):
        """Open a saved ``.json`` session and return its text (or ``""``)."""
        path, _ = QFileDialog.getOpenFileName(
            None, "Open session", self._last_dir,
            "UniGuide session (*.json);;All files (*)")
        if not path:
            return ""
        try:
            txt = Path(path).read_text(encoding="utf-8")
        except Exception as e:
            return json.dumps({"error": str(e)})
        self._last_dir = str(Path(path).parent)
        return txt

    @pyqtSlot(result=str)
    def library(self):
        """Return the unified-module size library.

        Loaded from ``guides.json`` next to this file if present, otherwise a
        built-in default. Replace/extend with your real catalogue.
        """
        f = _res_dir() / "guides.json"
        if f.exists():
            try:
                return f.read_text(encoding="utf-8")
            except Exception:
                pass
        return json.dumps({
            "sizes": [
                {"id": "S",  "Lg": 14, "Wg": 9,  "Hg": 8,  "nW": 5, "nD": 4},
                {"id": "M",  "Lg": 18, "Wg": 11, "Hg": 9,  "nW": 6, "nD": 4},
                {"id": "L",  "Lg": 22, "Wg": 13, "Hg": 10, "nW": 7, "nD": 5},
                {"id": "XL", "Lg": 26, "Wg": 15, "Hg": 11, "nW": 8, "nD": 5},
            ]
        })

    # ── unified-module STL library (folder next to the app) ──────────────────
    @staticmethod
    def _modules_dir():
        return _res_dir() / "modules"

    @pyqtSlot(result=str)
    def list_modules(self):
        """List the STL files in the ``modules/`` folder beside this app.

        These are the standardized, reusable guide elements. The folder is the
        single source of truth: drop an ``*.stl`` in it and it shows up in the
        Guides step. Returns ``{"dir": <path>, "modules": [{"file","name"}...]}``.

        Modelling convention (so a module locks onto a cutting plane): model the
        element with its **cutting face on the local XY plane (Z = 0), the face
        normal along +Z, and the part centred on X/Y**. The app puts that local
        origin on the plane, keeping the face coincident while you slide/rotate
        it in-plane.
        """
        d = self._modules_dir()
        mods = []
        if d.exists():
            for p in sorted(d.iterdir(), key=lambda q: q.name.lower()):
                if p.is_file() and p.suffix.lower() == ".stl":   # case-insensitive (.stl / .STL)
                    mods.append({"file": p.name, "name": p.stem})
        return json.dumps({"dir": str(d), "modules": mods})

    @pyqtSlot(str, result=str)
    def read_module(self, fname):
        """Parse one module STL from the ``modules/`` folder into a triangle soup."""
        if not fname or ("/" in fname or "\\" in fname or ".." in fname):
            return json.dumps({"error": "bad name"})
        p = self._modules_dir() / fname
        if not p.exists():
            return json.dumps({"error": "not found"})
        try:
            pos = _parse_stl(str(p))
        except Exception as e:
            return json.dumps({"error": str(e)})
        return json.dumps({"file": fname, "positions": pos, "count": len(pos) // 9})

    @pyqtSlot(str, result=str)
    def dbg(self, msg):
        """Append a diagnostic line to debug/guide_load.log (so silent failures are inspectable)."""
        try:
            p = _user_data_dir() / "guide_load.log"   # writable location (the bundle may be read-only when frozen)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(str(msg) + "\n")
        except Exception:
            pass
        return "ok"

    # ── add your backend slots here ──────────────────────────────────────────
    # Examples of the "macroscopic" logic to wire from VS Code later:
    #   * import the resection planes from the VSP (read them off the STL or a
    #     companion file) and hand them to the UI;
    #   * run a server-side generative pipeline for the connector;
    #   * manage a real on-disk module catalogue / patient cases.


# ── main window ───────────────────────────────────────────────────────────────
HTML = (_res_dir() / "ui.html").read_text(encoding="utf-8")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} -- cutting-guide composer")
        self.resize(1280, 800)
        self.setMinimumSize(1024, 640)
        central = QWidget(); self.setCentralWidget(central)
        lay = QVBoxLayout(central); lay.setContentsMargins(0, 0, 0, 0)
        self.view = QWebEngineView()
        s = self.view.settings()
        WA = QWebEngineSettings.WebAttribute if USE_QT6 else QWebEngineSettings   # scoped enum in Qt6
        s.setAttribute(WA.JavascriptEnabled, True)
        s.setAttribute(WA.LocalContentCanAccessRemoteUrls, True)
        try:
            s.setAttribute(WA.LocalContentCanAccessFileUrls, True)   # let the page load libs/*.js
        except Exception:
            pass
        try:
            s.setAttribute(WA.WebGLEnabled, True)
        except Exception:
            pass
        self.view.setZoomFactor(1.0)
        lay.addWidget(self.view)
        self.backend = Backend()
        self.channel = QWebChannel()
        self.channel.registerObject("backend", self.backend)
        self.view.page().setWebChannel(self.channel)
        self.view.page().setBackgroundColor(QColor("#111111"))
        # Base-URL = the local libs/ folder, so <script src="three.min.js"> tags
        # resolve to the bundled file and the page gets a file origin to load it.
        _libs = (_res_dir() / "libs").as_posix()
        if not _libs.endswith("/"):
            _libs += "/"
        self.view.setHtml(HTML, QUrl.fromLocalFile(_libs))


if __name__ == "__main__":
    # Force GPU compositing on (helps WebGL on driver blocklists). If the window
    # comes up black, set USE_GPU=False at the top of the file.
    _flags = "--ignore-gpu-blocklist " if USE_GPU else "--disable-gpu --disable-features=VizDisplayCompositor "
    os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = _flags + "--force-device-scale-factor=1"
    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "0"
    os.environ["QT_SCALE_FACTOR"] = "1"
    if USE_QT6:
        try:
            QApplication.setHighDpiScaleFactorRoundingPolicy(
                Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
        except Exception:
            pass
    else:
        QApplication.setAttribute(Qt.AA_DisableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, False)
    try:
        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        win = MainWindow()
        win.show()
        sys.exit(app.exec() if USE_QT6 else app.exec_())
    except SystemExit:
        raise
    except BaseException:
        # In a windowed (console-less) .exe a startup crash would vanish silently.
        # Persist the traceback so it can be inspected + surface a native message box.
        import traceback
        _tb = traceback.format_exc()
        try:
            (_user_data_dir() / "startup_error.log").write_text(_tb, encoding="utf-8")
        except Exception:
            pass
        try:
            from PyQt6.QtWidgets import QMessageBox  # type: ignore
            QMessageBox.critical(None, APP_NAME + " — startup error", _tb[-1500:])
        except Exception:
            sys.stderr.write(_tb)
        raise
