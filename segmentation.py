"""
UniGuide automatic segmentation backend.

Runs in the dedicated segmentation Python environment (torch + nnU-Net + MOOSE +
SimpleITK + scikit-image + scipy + trimesh). The main UniGuide app invokes this as a
subprocess and talks to it through a single JSON object printed on stdout, so the heavy
deep-learning stack never has to live inside the app's own interpreter.

The interface is intentionally tiny:

  python segmentation.py list-series <dicom_folder>
      -> {"ok": true, "series": [ {series_id, description, modality, slices, thickness,
                                    size, spacing}, ... ]}

  python segmentation.py segment <dicom_folder> <series_id> <region> <work_dir>
      region in {"head", "leg"}
      -> {"ok": true, "region": ..., "label_nii": <path>,
          "stls": { "<PartName>": "<stl_path>", ... }}

Design notes validated on real patient data (see memory uniguide-segmentation-pipeline):
  * One engine, MOOSE (moosez), covers both regions: clin_ct_dental for the head
    (== DentalSegmentator, Dataset112) and clin_ct_peripheral_bones for the leg.
  * The caller MUST copy the DICOM off any network drive into a local work_dir first;
    reading a series straight from Google Drive is far too slow.
  * Out-of-distribution cleanup is mandatory: the head model, trained on head CT/CBCT,
    mislabels neck/thorax bone as mandible on a "collo-torace" scan. Single bones keep
    only their largest connected component; the mandibular canal is constrained to the
    dilated mandible; bilateral / multi-part structures keep every sizeable component.
  * Every surface is emitted in the CT physical frame (origin/spacing/direction), so the
    STLs drop straight onto the DICOM-derived anatomy with no extra registration.

All progress / status text goes to stderr; stdout carries only the final JSON.
"""

import os
import sys
import json
import glob
import shutil
import subprocess

import numpy as np
import SimpleITK as sitk
from skimage import measure
from scipy import ndimage
import trimesh


# --------------------------------------------------------------------------- config

REGIONS = {
    "head": {
        "moose_model": "clin_ct_dental",
        # DentalSegmentator (Dataset112) native label indices 1-5, preserved by MOOSE.
        # Label 6 (Tumour) is not produced by the model: it is there so the editor offers it
        # for the surgeon to DRAW the tumour mass on the slices, then re-mesh it to an STL.
        "labels": {1: "Skull", 2: "Mandible", 3: "UpperTeeth", 4: "LowerTeeth",
                   5: "MandibularCanal", 6: "Tumour"},
        "largest_only": {1},             # the skull is a single connected bone
        # The mandible: keep EVERY large piece, not just the biggest. A reconstruction patient's
        # mandible is often in two (or more) segments because the anterior body has been resected
        # or eroded (a defect gap), and both remaining segments must be kept.
        "keep_large": {2},
        "constrain_to_host": {5: 2},     # canal (5) must live inside the mandible (2)
        # Only mesh what the reconstruction needs: the mandible and the nerve canal. The full
        # 5-label mask is still saved in the bundle, so skull/teeth stay editable, but we skip
        # the very large skull mesh so the result is fast. The tumour is drawn by hand later.
        "stl_labels": {2, 5},
    },
    "leg": {
        "moose_model": "clin_ct_peripheral_bones",
        # Dataset666_Peripheral-Bones indices (both sides kept; the app picks the donor side).
        "labels": {7: "Fibula_L", 8: "Fibula_R", 26: "Tibia_L", 27: "Tibia_R"},
        "largest_only": {7, 8, 26, 27},
        "constrain_to_host": {},
    },
}


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ------------------------------------------------------------------- DICOM handling
# Reading DICOM straight off a network drive (Google Drive) is far too slow, so listing
# reads only a few header tags per file, in parallel, and copies nothing. The actual
# segmentation copies just the chosen series to a local cache, then works locally.

_SCAN_TAGS = ["SeriesInstanceUID", "SeriesDescription", "Modality", "SliceThickness", "InstanceNumber"]


def _iter_files(root):
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            yield os.path.join(dirpath, f)


def _read_tags(fp):
    try:
        import pydicom
        ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True, specific_tags=_SCAN_TAGS)
        uid = str(ds.get("SeriesInstanceUID", "") or "")
        if not uid:
            return None
        return (uid, fp, str(ds.get("SeriesDescription", "") or ""),
                str(ds.get("Modality", "") or ""), str(ds.get("SliceThickness", "") or ""))
    except Exception:
        return None


def scan_series(root):
    """Group all DICOM under ``root`` by series, reading minimal tags in parallel."""
    from concurrent.futures import ThreadPoolExecutor
    files = list(_iter_files(root))
    total = len(files)
    if not total:
        return {}
    series = {}
    done = [0]
    step = max(1, total // 20)

    def work(fp):
        r = _read_tags(fp)
        done[0] += 1
        if done[0] % step == 0:
            log("scanning: %d/%d files" % (done[0], total))
        return r

    with ThreadPoolExecutor(max_workers=24) as ex:
        for r in ex.map(work, files):
            if not r:
                continue
            uid, fp, desc, mod, thk = r
            s = series.get(uid)
            if s is None:
                s = {"files": [], "description": desc, "modality": mod,
                     "thickness": thk, "folder": os.path.dirname(fp)}
                series[uid] = s
            s["files"].append(fp)
    return series


def list_series(root):
    """List DICOM series anywhere under ``root``, richest first. No copying."""
    series = scan_series(root)
    out = []
    for uid, s in series.items():
        try:
            rel = os.path.relpath(s["folder"], root)
        except Exception:
            rel = s["folder"]
        out.append({
            "series_id": uid, "folder": s["folder"], "rel": rel,
            "description": s["description"], "modality": s["modality"],
            "thickness": s["thickness"], "slices": len(s["files"]),
        })
    out.sort(key=lambda r: -r["slices"])
    log("found %d series under %s" % (len(out), root))
    return out


def _copy_series_local(files, dst):
    """Copy a series' files to a local folder in parallel, with progress."""
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(dst, exist_ok=True)
    total = len(files)
    done = [0]
    step = max(1, total // 20)

    def cp(i_fp):
        i, fp = i_fp
        out = os.path.join(dst, "%06d.dcm" % i)
        try:
            if not (os.path.exists(out) and os.path.getsize(out) > 0):
                shutil.copyfile(fp, out)
        except Exception as e:
            log("copy failed:", fp, e)
        done[0] += 1
        if done[0] % step == 0:
            log("copying: %d/%d files" % (done[0], total))

    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(cp, enumerate(files)))
    log("copied %d files -> %s" % (total, dst))
    return dst


def series_to_nifti(folder, series_id, out_path):
    """Read one series and write it as an int16 NIfTI, preserving geometry.

    ``folder`` is the local copy that holds ONLY the chosen series, so if the exact id
    lookup misses (SimpleITK can format the series id a bit differently from pydicom's UID)
    we simply read the single series present in the folder.
    """
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(folder, series_id) if series_id else []
    if not files:
        files = reader.GetGDCMSeriesFileNames(folder)  # the only series in the local copy
    if not files:
        raise ValueError("no readable DICOM series in %s" % folder)
    reader.SetFileNames(files)
    img = reader.Execute()
    img = sitk.Cast(img, sitk.sitkInt16)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sitk.WriteImage(img, out_path)
    log("nifti:", img.GetSize(), img.GetSpacing(), "->", out_path)
    return out_path


# --------------------------------------------------------------------------- MOOSE

def _moose_exe():
    d = os.path.dirname(sys.executable)
    for name in ("moosez.exe", "moosez"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return "moosez"  # rely on PATH


def run_moose(ct_nifti, model, work_dir):
    """Run a MOOSE model on a single CT and return the multilabel segmentation path.

    MOOSE wants <main>/<subject>/CT_*.nii.gz, so we stage the image accordingly.
    """
    main_dir = os.path.join(work_dir, "moose_in")
    subj = os.path.join(main_dir, "CASE")
    os.makedirs(subj, exist_ok=True)
    staged = os.path.join(subj, "CT_" + os.path.basename(ct_nifti).replace("CT_", ""))
    if os.path.abspath(staged) != os.path.abspath(ct_nifti):
        shutil.copyfile(ct_nifti, staged)

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    cmd = [_moose_exe(), "-d", main_dir, "-m", model]
    log("running MOOSE:", " ".join(cmd))
    proc = subprocess.run(cmd, env=env, stdout=sys.stderr, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError("MOOSE exited with code %d" % proc.returncode)

    hits = glob.glob(os.path.join(subj, "**", "segmentations", "*" + model.replace("clin_ct_", "").replace("clin_", "") + "*.nii.gz"), recursive=True)
    if not hits:
        hits = glob.glob(os.path.join(subj, "**", "segmentations", "*.nii.gz"), recursive=True)
    if not hits:
        raise RuntimeError("MOOSE produced no segmentation under %s" % subj)
    hits.sort(key=os.path.getmtime)
    log("MOOSE output:", hits[-1])
    return hits[-1]


def _find_dental_model():
    """Locate the DentalSegmentator (Dataset112) 3d_fullres trainer folder and the fold that
    actually has a checkpoint. Returns (trainer_folder, fold) or (None, None). MOOSE ships the
    weights under fold_all; a manual download may use fold_0."""
    bases = []
    try:
        import moosez
        bases.append(os.path.join(os.path.dirname(moosez.__file__), "models",
                                  "nnunet_trained_models", "Dataset112_DentalSegmentator_v100"))
    except Exception:
        pass
    bases.append(r"D:/uniguide_seg_models/Dataset112_DentalSegmentator_v100")
    for base in bases:
        if not os.path.isdir(base):
            continue
        for name in os.listdir(base):
            tf = os.path.join(base, name)
            if "3d_fullres" not in name or not os.path.isdir(tf):
                continue
            for fold in ("all", "0"):
                if os.path.exists(os.path.join(tf, "fold_" + fold, "checkpoint_final.pth")):
                    return tf, fold
    return None, None


def _crop_to_bone(nifti_path, out_path, thr=150, margin_mm=10.0):
    """Crop a CT to the bounding box of the bone (+margin), dropping the air/table around it.
    DentalSegmentator resamples to a very fine spacing, so on a large head CT the on-CPU
    segmentation map can need >10 GB of RAM; trimming the empty margin cuts that a lot. The
    physical frame is preserved, so the resulting STL still lands in the right place."""
    img = sitk.ReadImage(nifti_path)
    arr = sitk.GetArrayFromImage(img)  # (z,y,x)
    mask = arr > thr
    if int(mask.sum()) < 5000:
        return nifti_path
    sx, sy, sz = img.GetSpacing()
    # Drop the burned-in annotation text (patient name, "BONE+/W630", kV) that CT scanners
    # paint into the image corners. Those pixels are bright, so a plain threshold treats them
    # as bone and the bbox stretches out to the corners. The text sits in tiny blobs that are
    # disconnected from the body, so keep only connected components big enough to be real bone
    # (>= 0.5 cm3), which removes the text while keeping the skull, mandible and spine.
    try:
        from scipy import ndimage
        lab, n = ndimage.label(mask)
        if n > 1:
            counts = np.bincount(lab.ravel())
            counts[0] = 0
            keep = np.where(counts * (sx * sy * sz) >= 500.0)[0]
            if len(keep):
                mask = np.isin(lab, keep)
    except Exception as e:
        log("bone-component cleanup skipped:", e)
    zz, yy, xx = np.where(mask)
    mx, my, mz = int(margin_mm / sx), int(margin_mm / sy), int(margin_mm / sz)
    x0, x1 = max(0, int(xx.min()) - mx), min(arr.shape[2], int(xx.max()) + mx + 1)
    y0, y1 = max(0, int(yy.min()) - my), min(arr.shape[1], int(yy.max()) + my + 1)
    z0, z1 = max(0, int(zz.min()) - mz), min(arr.shape[0], int(zz.max()) + mz + 1)
    crop = img[x0:x1, y0:y1, z0:z1]
    sitk.WriteImage(crop, out_path)
    log("cropped to bone bbox:", crop.GetSize(), "from", img.GetSize())
    return out_path


def _reveal_predictor_class():
    """A nnUNetPredictor subclass that reveals the finalized Z-bands DURING the sliding window,
    so the mandible can grow bottom to top in the UI while a SINGLE in-process inference runs:
    no extra GPU work, no seams, same memory as the plain run. Returns None if the nnU-Net
    internals it overrides are missing, so the caller falls back to the stock predictor.
    Verified against nnU-Net 2.8.1 (predict_from_raw_data._internal_predict_sliding_window_*)."""
    try:
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        from nnunetv2.inference.sliding_window_prediction import compute_gaussian
        from nnunetv2.utilities.helpers import empty_cache
        from queue import Queue
        from threading import Thread
        from tqdm import tqdm
        import torch
    except Exception as e:
        log("reveal predictor unavailable, using the plain one:", e)
        return None

    class RevealPredictor(nnUNetPredictor):
        reveal_cb = None       # reveal_cb(logits, n_predictions, frontier_working_z, ascending)
        z_ascending = True

        def _internal_get_sliding_window_slicers(self, image_size):
            s = super()._internal_get_sliding_window_slicers(image_size)
            # work the tiles bottom to top (working Z is slicer index 2) so the mandible body
            # finalizes first; flip when the scan's superior direction runs the other way.
            try:
                s.sort(key=lambda sl: sl[2].start, reverse=not self.z_ascending)
            except Exception as e:
                log("slicer sort skipped:", e)
            return s

        @torch.inference_mode()
        def _internal_predict_sliding_window_return_logits(self, data, slicers, do_on_device=True):
            # Verbatim copy of nnUNetPredictor's method (nnU-Net 2.8.1) with ONE added reveal
            # hook. If anything in the hook fails it disables itself and the inference finishes
            # exactly as the stock method would, so behaviour degrades safely.
            predicted_logits = n_predictions = prediction = gaussian = workon = None
            results_device = self.device if do_on_device else torch.device("cpu")

            def producer(d, slh, q):
                for s in slh:
                    q.put((torch.clone(d[s][None], memory_format=torch.contiguous_format).to(self.device), s))
                q.put("end")

            try:
                empty_cache(self.device)
                data = data.to(results_device)
                queue = Queue(maxsize=2)
                t = Thread(target=producer, args=(data, slicers, queue), daemon=True)
                t.start()
                predicted_logits = torch.zeros((self.label_manager.num_segmentation_heads, *data.shape[1:]),
                                               dtype=torch.half, device=results_device)
                n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)
                if self.use_gaussian:
                    gaussian = compute_gaussian(tuple(self.configuration_manager.patch_size), sigma_scale=1. / 8,
                                                value_scaling_factor=10, device=results_device)
                else:
                    gaussian = 1
                every = max(1, len(slicers) // 30)      # ~30 slice previews across the run, so it flows
                with tqdm(desc=None, total=len(slicers), disable=not self.allow_tqdm) as pbar:
                    while True:
                        item = queue.get()
                        if item == "end":
                            queue.task_done()
                            break
                        workon, sl = item
                        prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)
                        if self.use_gaussian:
                            prediction *= gaussian
                        predicted_logits[sl] += prediction
                        n_predictions[sl[1:]] += gaussian
                        queue.task_done()
                        pbar.update()
                        # Send the CURRENT mandible as a 3D point cloud often (~1/30 of the tiles)
                        # so it forms live in 3D. No meshing here: the solid is built once at the
                        # very end, which is far cheaper than re-meshing every step.
                        if self.reveal_cb is not None and (pbar.n % every == 0):
                            try:
                                self.reveal_cb(predicted_logits, n_predictions)
                            except Exception as ex:
                                log("reveal hook disabled:", ex)
                                self.reveal_cb = None
                queue.join()
                torch.div(predicted_logits, n_predictions, out=predicted_logits)
                if torch.any(torch.isinf(predicted_logits)):
                    raise RuntimeError("Encountered inf in predicted array.")
                return predicted_logits
            except Exception as e:
                del predicted_logits, n_predictions, prediction, gaussian, workon
                empty_cache(self.device)
                empty_cache(results_device)
                raise e

    return RevealPredictor


def _save_cloud_png(world, debug_dir, step):
    """Debug: a 3D scatter of the forming mandible point cloud, so the build is visible on disk too."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(debug_dir, exist_ok=True)
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(world[:, 0], world[:, 1], world[:, 2], s=1, c="#dd5555")
        ax.set_title("mandible cloud, step %d" % step)
        try:
            ax.set_box_aspect((np.ptp(world[:, 0]) or 1, np.ptp(world[:, 1]) or 1, np.ptp(world[:, 2]) or 1))
        except Exception:
            pass
        ax.view_init(elev=15, azim=-75)
        ax.axis("off")
        fig.savefig(os.path.join(debug_dir, "cloud_%02d.png" % step), dpi=70)
        plt.close(fig)
    except Exception as e:
        log("cloud png skipped:", e)


def _make_reveal(crop_img, grow_dir, cm, transpose_back, debug_dir=None):
    """Build the reveal callback: send the CURRENT mandible as a 3D POINT CLOUD (its voxel centres
    in CT world coordinates) on a POINTS line, so the UI shows it forming in 3D with NO meshing at
    all (cheap). The solid STL is built once at the very end. Called often, so it skips when nothing
    new was added and downsamples so each update stays small."""
    sp_w = np.array([float(cm.spacing[2]), float(cm.spacing[0]), float(cm.spacing[1])])  # sitk (x,y,z)
    origin = np.array(crop_img.GetOrigin(), float)
    D = np.array(crop_img.GetDirection(), float).reshape(3, 3)
    state = {"step": 0, "last_vox": 0}

    def cb(logits, npred):
        # logits: (C, Y', Z', X') working grid, still un-normalized; npred: (Y', Z', X') weights.
        touched = npred > 0.5
        seg = (logits / npred.clamp(min=1e-4)).argmax(0)          # (Y', Z', X')
        mand = ((seg == 2) & touched).to("cpu").numpy()           # (Y', Z', X') bool
        vox = int(mand.sum())
        if vox < 200 or vox < state["last_vox"] * 1.03:           # nothing meaningfully new -> skip
            return
        state["last_vox"] = vox
        arr = np.ascontiguousarray(mand.transpose(transpose_back))            # -> (Z', Y', X') sitk order
        arr = clean_components(arr, largest_only=False, rel=0.40)             # both mandible segments, no stray blobs
        kz, ky, kx = np.where(arr)
        n = len(kz)
        if n < 100:
            return
        if n > 6000:                                              # downsample so each update stays small
            sel = np.linspace(0, n - 1, 6000).astype(np.int64)
            kz, ky, kx = kz[sel], ky[sel], kx[sel]
        idx = np.column_stack([kx, ky, kz]).astype(float)         # (i, j, k) sitk voxel indices
        world = (origin + (idx * sp_w) @ D.T).astype(np.float32)  # (N, 3) world mm, same frame as the final STL
        state["step"] += 1
        import base64 as _b64
        log("POINTS %d %s" % (len(world), _b64.b64encode(world.tobytes()).decode()))
        if debug_dir:
            _save_cloud_png(world, debug_dir, state["step"])

    return cb


def run_dental_nnunet(ct_nifti, work_dir):
    """Head path: run DentalSegmentator directly (nnU-Net, no test-time mirroring) so it is
    faster than MOOSE AND streams a real percentage. Returns the label map, or None to fall
    back to MOOSE if the model is not present / it runs out of memory."""
    model, fold = _find_dental_model()
    if not model:
        log("dental model not found, falling back to MOOSE")
        return None
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
    in_dir = os.path.join(work_dir, "nn_in")
    out_dir = os.path.join(work_dir, "nn_out")
    grow_dir = os.path.join(work_dir, "grow")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    cropped = _crop_to_bone(ct_nifti, os.path.join(work_dir, "ct_bone.nii.gz"))
    crop_img = sitk.ReadImage(cropped)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("dental nnU-Net on", dev.type, "fold", fold, "(no mirroring)")
    _args = dict(tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
                 perform_everything_on_device=(dev.type == "cuda" and False),
                 device=dev, verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    use_fold = int(fold) if str(fold).isdigit() else fold
    RP = _reveal_predictor_class()          # progressive predictor, or None to use the plain one
    try:
        predictor = (RP or nnUNetPredictor)(**_args)
        predictor.initialize_from_trained_model_folder(model, use_folds=(use_fold,), checkpoint_name="checkpoint_final.pth")
    except Exception as e:
        log("reveal predictor init failed, using the plain one:", e)
        RP = None
        predictor = nnUNetPredictor(**_args)
        predictor.initialize_from_trained_model_folder(model, use_folds=(use_fold,), checkpoint_name="checkpoint_final.pth")

    # Run the fine pass on the whole bone crop. The mandible is recovered as the largest label-2
    # component (out-of-distribution neck/thorax bone the model mislabels stays as smaller pieces
    # and is dropped), so no jaw pre-crop is needed; an earlier teeth-slab cropped some mandibles
    # in half and is gone.
    fine_nifti, fine_img, paste = cropped, crop_img, None
    shutil.copyfile(fine_nifti, os.path.join(in_dir, "CASE_0000.nii.gz"))

    # RAM-aware resolution: the on-CPU segmentation map is (classes x resampled-volume). On a
    # large scan at the model's fine spacing that can be >10 GB. If free RAM is too small,
    # coarsen the working spacing just enough to fit (the mandible stays well resolved). The
    # voxel-count estimate is a product over axes, so axis order does not matter.
    try:
        import psutil
        cm = predictor.configuration_manager
        sh = np.array(fine_img.GetSize(), dtype=float)          # sitk (x,y,z)
        isp = np.array(fine_img.GetSpacing(), dtype=float)      # sitk (x,y,z)
        tgt = np.array(cm.spacing, dtype=float)                 # working order (y,z,x)
        n_cls = len(predictor.dataset_json.get("labels", {})) or 6
        # SPEED: the mandible does not need the model's finest spacing (~0.31 mm); cap the
        # working spacing so the head segments several times faster with negligible loss on the
        # bone surface. Floor it PER AXIS at the data's own spacing so no axis is ever upsampled
        # (a coarse through-plane axis pulled finer would blow up RAM for no added detail).
        # cm.spacing is in working order (y,z,x) = sitk (sy,sz,sx), so reorder the sitk spacing.
        TARGET_MAND = 0.6
        isp_w = isp[[1, 2, 0]]                                  # sitk (x,y,z) -> working (y,z,x)
        work = np.maximum(tgt, np.maximum(TARGET_MAND, isp_w))
        # RAM: coarsen further if the on-CPU map would not fit free memory (physical extent over
        # the working voxel volume, times the class count, fp32).
        soft_gb = float(np.prod(sh * isp) / np.prod(work)) * n_cls * 4 / 1e9
        free_gb = psutil.virtual_memory().available / 1e9
        budget = max(1.5, free_gb * 0.35)   # leave RAM for the model, the copies and the mesh
        if soft_gb > budget:
            # coarsen to fit, but never past 0.9 mm (keep the mandible usable) and never finer than
            # the data's own spacing on any axis (the cap must not upsample a coarse through-plane).
            work = np.maximum(np.minimum(work * (soft_gb / budget) ** (1.0 / 3.0), 0.9), isp_w)
            soft_gb = float(np.prod(sh * isp) / np.prod(work)) * n_cls * 4 / 1e9
            log("low RAM: %.1f GB free; coarsening to %s (map ~%.1f GB)"
                % (free_gb, [round(x, 2) for x in work], soft_gb))
        cm.configuration["spacing"] = work.tolist()
        log("working spacing %s, map ~%.1f GB, %.1f GB free"
            % ([round(x, 2) for x in work], soft_gb, free_gb))
    except Exception as e:
        log("RAM check skipped:", e)

    # Wire the progressive reveal now that the working spacing is fixed. Guarded: any failure
    # here just means the mandible appears once at the end instead of growing bottom to top.
    if RP is not None and isinstance(predictor, RP):
        try:
            D = np.array(fine_img.GetDirection(), float).reshape(3, 3)
            predictor.z_ascending = bool(D[2, 2] >= 0)          # tiles bottom to top: mandible forms first
            tb = list(predictor.plans_manager.transpose_backward)
            dbg = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug", "segmentation")
            predictor.reveal_cb = _make_reveal(fine_img, grow_dir, predictor.configuration_manager,
                                               tb, debug_dir=dbg)
            log("progressive reveal on (z_ascending=%s)" % predictor.z_ascending)
        except Exception as e:
            log("reveal wiring skipped:", e)

    log("PHASE preparing the model")   # fills the silent gap before the sliding window starts

    # Run the prediction IN-PROCESS (predict_single_npy_array) instead of predict_from_files.
    # predict_from_files spawns multiprocessing workers for preprocessing and export; when the
    # parent runs low on memory those workers get orphaned and keep holding RAM, which then
    # starves every later run (this is what made back-to-back segmentations cascade into
    # failure). The single-array path does everything in this one process, so nothing leaks.
    out_path = os.path.join(out_dir, "CASE.nii.gz")
    io = SimpleITKIO()
    try:
        img, props = io.read_images([os.path.join(in_dir, "CASE_0000.nii.gz")])
        seg = predictor.predict_single_npy_array(img, props, None, None, False)   # (z,y,x) at fine geometry
        if paste is not None:
            # paste the head-crop labels back into the full bone-crop frame so the saved mask and
            # the edit bundle keep the geometry the rest of the pipeline expects
            Z0, Y0, X0 = paste
            gz, gy, gx = crop_img.GetSize()[2], crop_img.GetSize()[1], crop_img.GetSize()[0]
            full = np.zeros((gz, gy, gx), dtype=np.uint8)
            full[Z0:Z0 + seg.shape[0], Y0:Y0 + seg.shape[1], X0:X0 + seg.shape[2]] = seg.astype(np.uint8)
            out_img = sitk.GetImageFromArray(full)
            out_img.CopyInformation(crop_img)
            sitk.WriteImage(out_img, out_path)
        else:
            io.write_seg(seg, out_path, props)
    except Exception as e:
        # Same model as MOOSE, so a memory failure would fail there too: give a clear message
        # instead of falling back and failing again.
        raise RuntimeError("LOW_MEMORY: not enough free RAM to segment this scan at full "
                           "resolution. Close other apps (the browser especially) to free "
                           "memory, then try again. (%s)" % e)
    finally:
        # free the model + logits before the meshing step so it does not run out of memory
        try:
            del predictor
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
    if not os.path.exists(out_path):
        raise RuntimeError("dental nnU-Net produced no output")
    log("dental output:", out_path)
    return out_path


# ---------------------------------------------------------------- post-processing

def clean_components(mask, largest_only=False, rel=0.10, abs_min=50):
    """Keep connected components. largest_only -> just the biggest (one bone);
    otherwise keep every component >= max(abs_min, rel*largest) so bilateral or
    multi-part structures survive while isolated specks are dropped."""
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1)).astype(np.int64)
    if largest_only:
        keep = {int(np.argmax(sizes)) + 1}
    else:
        thr = max(abs_min, int(rel * sizes.max()))
        keep = {i + 1 for i, s in enumerate(sizes) if s >= thr}
    return np.isin(lab, list(keep))


def constrain_to(mask, host_mask, spacing, dilate_mm=4.0):
    """Keep only voxels of mask that fall inside host_mask dilated by dilate_mm."""
    iters = max(1, int(round(dilate_mm / float(min(spacing)))))
    host_d = ndimage.binary_dilation(host_mask, iterations=iters)
    return mask & host_d


def mask_to_mesh(mask, img, smooth_iter=10):
    """Marching-cubes a binary mask into a Trimesh in the image physical frame."""
    if mask.sum() < 50:
        return None
    m = np.pad(mask.astype(np.uint8), 1, mode="constant")
    verts, faces, _, _ = measure.marching_cubes(m, level=0.5)
    verts -= 1.0
    idx = verts[:, ::-1]  # (k,j,i) -> (i,j,k)
    sp = np.array(img.GetSpacing(), dtype=np.float64)
    origin = np.array(img.GetOrigin(), dtype=np.float64)
    D = np.array(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    phys = origin + (idx * sp) @ D.T
    mesh = trimesh.Trimesh(vertices=phys, faces=faces, process=True)
    if smooth_iter > 0:
        trimesh.smoothing.filter_taubin(mesh, lamb=0.5, nu=-0.53, iterations=smooth_iter)
    return mesh


def labels_to_stls(label_nii, region_cfg, out_dir):
    """Turn a MOOSE multilabel volume into per-structure STLs with the validated cleanup."""
    os.makedirs(out_dir, exist_ok=True)
    img = sitk.ReadImage(label_nii)
    arr = sitk.GetArrayFromImage(img)
    spacing = np.array(img.GetSpacing(), dtype=np.float64)
    present = set(int(v) for v in np.unique(arr) if v != 0)
    labels = region_cfg["labels"]
    largest_only = region_cfg["largest_only"]
    constrain = region_cfg["constrain_to_host"]
    stl_set = region_cfg.get("stl_labels")   # None => mesh every label; else only these

    # only the labels we actually mesh (+ any constraint hosts they need)
    mesh_labels = set(labels) if stl_set is None else set(stl_set)
    need = set(mesh_labels) | {constrain[l] for l in mesh_labels if l in constrain}

    keep_large = region_cfg.get("keep_large", set())
    masks = {}
    for lab in labels:
        if lab not in present or lab not in need:
            continue
        if lab in keep_large:
            # keep every component at least 40% of the biggest: both mandible segments of a defect
            # case survive, while small mislabeled neck/thorax pieces are dropped
            masks[lab] = clean_components(arr == lab, largest_only=False, rel=0.40)
        else:
            masks[lab] = clean_components(arr == lab, largest_only=(lab in largest_only))

    out = {}
    todo = [(lab, name) for lab, name in labels.items() if lab in masks and lab in mesh_labels]

    def _prio(name):                       # build the bone you need FIRST, the big skull last
        n = name.lower()
        if "mandible" in n or "fibula" in n:
            return 0
        if "canal" in n or "tibia" in n:
            return 1
        if "skull" in n:
            return 9
        return 5
    todo.sort(key=lambda t: _prio(t[1]))
    log("PHASE building %d 3D models" % len(todo))
    for lab, name in todo:
        log("PHASE building %s" % name)
        mask = masks[lab]
        host = constrain.get(lab)
        if host is not None and host in masks:
            if "canal" in name.lower():
                mask = ndimage.binary_closing(mask, iterations=1)   # heal the thin canal before clipping
            mask = constrain_to(mask, masks[host], spacing)
            mask = clean_components(mask, largest_only=False)
        # A secondary structure failing to mesh (a thin canal, say) must never sink the whole
        # run: the mandible is built first and already emitted, so keep going.
        try:
            mesh = mask_to_mesh(mask, img)
            if mesh is None:
                log("label", lab, name, "too small after cleanup"); continue
            path = os.path.join(out_dir, name + ".stl")
            mesh.export(path)
            log("wrote", name, len(mesh.vertices), "verts ->", path)
            log("STL_READY %s %s" % (name, path))   # UI loads it live
            out[name] = path
        except Exception as e:
            log("mesh skipped for %s: %s" % (name, e))
    return out


# ------------------------------------------------------------------- orchestration

def save_edit_bundle(ct_nifti, label_nii, cfg, out_dir):
    """Save a compact numpy bundle (CT + label mask + geometry + names) so the app can
    hand-edit the mask slice by slice without needing SimpleITK, then re-mesh."""
    os.makedirs(out_dir, exist_ok=True)
    ct_img = sitk.ReadImage(ct_nifti)
    lb_img = sitk.ReadImage(label_nii)
    ct = sitk.GetArrayFromImage(ct_img).astype(np.int16)          # (z,y,x)
    labels = sitk.GetArrayFromImage(lb_img).astype(np.uint8)
    path = os.path.join(out_dir, "bundle.npz")
    np.savez_compressed(
        path,
        ct=ct, labels=labels,
        spacing=np.array(ct_img.GetSpacing(), dtype=np.float64),
        origin=np.array(ct_img.GetOrigin(), dtype=np.float64),
        direction=np.array(ct_img.GetDirection(), dtype=np.float64),
        names=json.dumps({int(k): v for k, v in cfg["labels"].items()}),
    )
    log("edit bundle:", path)
    return path


GENERIC_LABELS = {1: "Bone", 2: "Nerve", 3: "Vessel", 4: "Tumour", 5: "Extra"}


def _prepare_local_series(dicom_folder, series_id, work_dir):
    """Copy just the chosen series to a local folder and return it (fast, off-drive)."""
    s = scan_series(dicom_folder)
    files = s.get(series_id, {}).get("files")
    if not files:
        raise ValueError("series not found under %s" % dicom_folder)
    return _copy_series_local(files, os.path.join(work_dir, "dicom_local"))


def ct_bundle(dicom_folder, series_id, work_dir):
    """Manual path: convert a chosen series to an editable bundle with an EMPTY mask,
    so the user can draw the segmentation from scratch, no model involved."""
    os.makedirs(work_dir, exist_ok=True)
    local = _prepare_local_series(dicom_folder, series_id, work_dir)
    nifti = series_to_nifti(local, series_id, os.path.join(work_dir, "input", "CT_manual.nii.gz"))
    ct_img = sitk.ReadImage(nifti)
    ct = sitk.GetArrayFromImage(ct_img).astype(np.int16)
    labels = np.zeros(ct.shape, dtype=np.uint8)
    out_dir = os.path.join(work_dir, "edit")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "bundle.npz")
    np.savez_compressed(
        path, ct=ct, labels=labels,
        spacing=np.array(ct_img.GetSpacing(), dtype=np.float64),
        origin=np.array(ct_img.GetOrigin(), dtype=np.float64),
        direction=np.array(ct_img.GetDirection(), dtype=np.float64),
        names=json.dumps(GENERIC_LABELS),
    )
    log("manual CT bundle:", path)
    return {"ok": True, "bundle": path, "names": GENERIC_LABELS}


def segment(dicom_folder, series_id, region, work_dir):
    if region not in REGIONS:
        raise ValueError("unknown region %r (expected head|leg)" % region)
    cfg = REGIONS[region]
    os.makedirs(work_dir, exist_ok=True)
    local = _prepare_local_series(dicom_folder, series_id, work_dir)
    log("PHASE reading the scan")
    nifti = series_to_nifti(local, series_id, os.path.join(work_dir, "input", "CT_%s.nii.gz" % region))
    if region == "head":
        # direct DentalSegmentator (fast, shows a percentage); MOOSE is the fallback
        label_nii = run_dental_nnunet(nifti, work_dir) or run_moose(nifti, cfg["moose_model"], work_dir)
    else:
        label_nii = run_moose(nifti, cfg["moose_model"], work_dir)
    stls = labels_to_stls(label_nii, cfg, os.path.join(work_dir, "stl"))
    log("PHASE preparing the editor")
    bundle = save_edit_bundle(nifti, label_nii, cfg, os.path.join(work_dir, "edit"))
    log("PHASE done")
    return {"ok": True, "region": region, "label_nii": label_nii, "stls": stls, "bundle": bundle}


def remesh(bundle_path, label_value, out_stl, largest_only=True, smooth_iter=10):
    """Re-mesh one label from a (possibly hand-edited) bundle into an STL in the CT frame.
    The edited mask, if present next to the bundle as ``labels_edited.npy``, wins."""
    b = np.load(bundle_path, allow_pickle=True)
    edited = os.path.join(os.path.dirname(bundle_path), "labels_edited.npy")
    labels = np.load(edited) if os.path.exists(edited) else b["labels"]
    # rebuild a minimal SimpleITK image just to carry the geometry for mask_to_mesh
    img = sitk.GetImageFromArray(labels)
    img.SetSpacing(tuple(float(x) for x in b["spacing"]))
    img.SetOrigin(tuple(float(x) for x in b["origin"]))
    img.SetDirection(tuple(float(x) for x in b["direction"]))
    mask = (labels == int(label_value))
    mask = clean_components(mask, largest_only=largest_only)
    mesh = mask_to_mesh(mask, img, smooth_iter=smooth_iter)
    if mesh is None:
        raise RuntimeError("label %s empty after edit" % label_value)
    os.makedirs(os.path.dirname(out_stl) or ".", exist_ok=True)
    mesh.export(out_stl)
    log("remesh label", label_value, "->", out_stl, len(mesh.vertices), "verts")
    return {"ok": True, "stl": out_stl, "verts": int(len(mesh.vertices))}


# ---------------------------------------------------------------------------- CLI

def main(argv):
    if len(argv) < 2:
        print(json.dumps({"ok": False, "error": "usage: segmentation.py <list-series|segment> ..."}))
        return 2
    cmd = argv[1]
    try:
        if cmd == "list-series":
            result = {"ok": True, "series": list_series(argv[2])}
        elif cmd == "segment":
            result = segment(argv[2], argv[3], argv[4], argv[5])
        elif cmd == "ctbundle":
            # ctbundle <folder> <series_id> <work_dir>  -> empty-mask bundle for manual drawing
            result = ct_bundle(argv[2], argv[3], argv[4])
        elif cmd == "remesh":
            # remesh <bundle.npz> <label_value> <out.stl> [keep-largest|keep-all]
            largest = (len(argv) <= 5) or (argv[5] != "keep-all")
            result = remesh(argv[2], int(argv[3]), argv[4], largest_only=largest)
        else:
            result = {"ok": False, "error": "unknown command %r" % cmd}
    except Exception as e:
        import traceback
        traceback.print_exc(file=sys.stderr)
        result = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
    # Tag the result line so the app can find it even if the ML stack prints to stdout.
    print("UNIGUIDE_RESULT " + json.dumps(result), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
