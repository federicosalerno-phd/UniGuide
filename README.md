# UniGuide: modular cutting-guide composer

A desktop tool to **compose standardized, partially reusable mandibular cutting
guides** and to **auto-generate the patient-specific connector** that locks the
reusable modules into a single rigid body. It is the software released alongside
the cutting-guide standardization work.

The UI is an embedded web view (`ui.html`) driven by a Python/Qt backend over
`QWebChannel`, the same architecture as the IMTOP annotator. The workflow is:

1. **Model**, load the mandible STL (in the clinical workflow it already carries
   the resection cutting planes from the virtual surgical plan).
2. **Planes**, inspect and define the two cutting planes (mandibular sides).
3. **Guides**, place the two unified modules so each cutting face coincides with
   a plane. One module carries two support feet, the other one, giving a
   determinate three-point contact once they are bridged. Module sizes are
   picked from a small library.
4. **Connector**, auto-generate the bridge from a guided spline and a chosen
   cross-section profile. This is the only patient-specific, printed part.
5. **Check**, kinematic checks (three feet seated, clearance, faces coincident).
6. **Export**, write a print-ready STL for the Formlabs resin workflow, either
   the connector alone or the full assembly.

The reusable left/right modules are meant to be resterilized between cases; only
the connector is fabricated per patient. This separation, standardized
resterilizable modules plus a minimal patient-specific bridge, is the point of
the tool.

## Download & install (Windows) — no Python needed

The easiest way, like any normal app:

1. Go to the [**Releases**](https://github.com/federicosalerno-phd/UniGuide/releases)
   page and download **`UniGuide-windows.zip`** from the latest release.
2. Unzip it anywhere (e.g. your Desktop).
3. Open the `UniGuide` folder and double-click **`UniGuide.exe`**.

That's it — Python and every library are bundled inside; nothing else to install.
To make a desktop icon: right-click `UniGuide.exe` → *Send to → Desktop (create shortcut)*.

> **Windows SmartScreen** may show "Windows protected your PC" because the app is
> not code-signed (normal for open-source apps). Click **More info → Run anyway**.

Prefer to run from source instead? See [Running from source](#running-from-source) below.

## Repository layout

```
uniguide_app.py     Main application (Qt window + backend)
ui.html             Embedded web UI (Three.js viewer + workflow)
libs/               Bundled JS for the 3D viewer (three.min.js)
modules/            Standardized guide-element STL library
input/              Demo sample STLs (mandible / fibula)
guides.json         Unified-module size library (editable)
launch.bat          Windows launcher (from source)
build.bat           Build the standalone .exe locally (PyInstaller)
UniGuide.spec       PyInstaller build recipe
requirements.txt    Pinned Python dependencies
.github/workflows/  CI: builds the Windows .exe on each Release
```

## Requirements

Clone the repository and enter it:

```bash
git clone https://github.com/federicosalerno-phd/UniGuide.git
cd UniGuide
```

- **Python 3.11**
- The packages in [`requirements.txt`](requirements.txt) (PyQt6 + PyQt6-WebEngine).

```bash
pip install -r requirements.txt
```

The app prefers **PyQt6** (smoother embedded-Chromium compositor) and falls back
to **PyQt5 + PyQtWebEngine** automatically if PyQt6 is not installed.

### Bundled 3D library

The viewer uses Three.js r128, vendored at `libs/three.min.js` so the app runs
offline. If you cloned without it, fetch it once:

```bash
curl -L -o libs/three.min.js https://raw.githubusercontent.com/mrdoob/three.js/r128/build/three.min.js
```

When `ui.html` is opened directly in a browser for development, it falls back to
the same release from a CDN, so a missing `libs/three.min.js` is not fatal there.

## Running from source

```bash
python uniguide_app.py
```

On Windows, double-click **`launch.bat`** — it launches with the project's
virtualenv. The app is also **self-bootstrapping**: if you start it with a Python
that lacks PyQt6 (e.g. the system Python via the IDE's ▶ Run button), it
automatically re-launches itself with a Python that has Qt, so it "just works".
Point it at a specific interpreter with the `UNIGUIDE_PYTHON` environment variable.

## Building the standalone .exe yourself

```bash
pip install "setuptools<81" pyinstaller
pyinstaller UniGuide.spec --noconfirm      # or double-click build.bat on Windows
```

Output: `dist/UniGuide/UniGuide.exe` (a self-contained one-folder app — zip the
folder to share it). GitHub Actions builds and attaches this zip to every
[Release](https://github.com/federicosalerno-phd/UniGuide/releases) automatically.

## Status

This is an early release (0.1.0). What works today:

- loading and viewing a real mandible STL (binary or ASCII), through the native
  Qt file dialog or by drag and drop;
- the full composition workflow on a built-in parametric demo mandible: plane
  inspection, module placement from the size library, connector generation, and
  print-ready STL export of the connector or the full assembly.

The next steps, the "macroscopic" parts to wire from an editor, are:

- mapping module placement onto an arbitrary imported mesh (the demo arch does
  not yet align to a generic STL);
- importing the resection planes directly from the virtual surgical plan;
- the fibula counterpart (same connector, unified modules reversed);
- a real on-disk module catalogue and patient cases.

## License

Released under the [MIT License](LICENSE).

## Citation

If you use this software, please cite it using [`CITATION.cff`](CITATION.cff).
