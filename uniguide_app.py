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
import os, sys, json, struct

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
    from PyQt6.QtCore import QObject, pyqtSlot, pyqtSignal, QUrl, Qt
    from PyQt6.QtGui import QColor
    USE_QT6 = True
except Exception:
    from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QFileDialog
    from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEngineSettings
    from PyQt5.QtWebChannel import QWebChannel
    from PyQt5.QtCore import QObject, pyqtSlot, pyqtSignal, QUrl, Qt
    from PyQt5.QtGui import QColor
    USE_QT6 = False

APP_NAME = "UniGuide"
APP_VERSION = "0.1.0"


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

    @staticmethod
    def _input_dir():
        return _res_dir() / "input"

    def __init__(self):
        super().__init__()
        d = self._input_dir()
        self._last_dir = str(d if d.exists() else Path.home())

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
