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
        # DentalSegmentator (Dataset112) native label indices, preserved by MOOSE.
        "labels": {1: "Skull", 2: "Mandible", 3: "UpperTeeth", 4: "LowerTeeth", 5: "MandibularCanal"},
        "largest_only": {1, 2},          # skull and mandible are one connected bone each
        "constrain_to_host": {5: 2},     # canal (5) must live inside the mandible (2)
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
    """Read one series and write it as an int16 NIfTI, preserving geometry."""
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(folder, series_id)
    if not files:
        raise ValueError("series %s not found in %s" % (series_id, folder))
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
    """Locate the DentalSegmentator (Dataset112) 3d_fullres trainer folder, or None."""
    bases = []
    try:
        import moosez
        bases.append(os.path.join(os.path.dirname(moosez.__file__), "models",
                                  "nnunet_trained_models", "Dataset112_DentalSegmentator_v100"))
    except Exception:
        pass
    bases.append(r"D:/uniguide_seg_models/Dataset112_DentalSegmentator_v100")
    for base in bases:
        if os.path.isdir(base):
            for name in os.listdir(base):
                if "3d_fullres" in name and os.path.isdir(os.path.join(base, name)):
                    return os.path.join(base, name)
    return None


def run_dental_nnunet(ct_nifti, work_dir):
    """Head path: run DentalSegmentator directly (nnU-Net, no test-time mirroring) so it is
    faster than MOOSE AND streams a real percentage. Returns the label map, or None to fall
    back to MOOSE if the model is not present."""
    model = _find_dental_model()
    if not model:
        log("dental model not found, falling back to MOOSE")
        return None
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    in_dir = os.path.join(work_dir, "nn_in")
    out_dir = os.path.join(work_dir, "nn_out")
    os.makedirs(in_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copyfile(ct_nifti, os.path.join(in_dir, "CASE_0000.nii.gz"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log("dental nnU-Net on", dev.type, "(no mirroring)")
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=False,
        perform_everything_on_device=(dev.type == "cuda" and False),
        device=dev, verbose=False, verbose_preprocessing=False, allow_tqdm=True)
    predictor.initialize_from_trained_model_folder(model, use_folds=(0,), checkpoint_name="checkpoint_final.pth")
    predictor.predict_from_files(in_dir, out_dir, save_probabilities=False, overwrite=True,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2)
    outs = glob.glob(os.path.join(out_dir, "*.nii.gz"))
    if not outs:
        raise RuntimeError("dental nnU-Net produced no output")
    log("dental output:", outs[0])
    return outs[0]


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

    # build cleaned masks first (hosts may be needed for constraints)
    masks = {}
    for lab in labels:
        if lab not in present:
            continue
        masks[lab] = clean_components(arr == lab, largest_only=(lab in largest_only))

    out = {}
    for lab, name in labels.items():
        if lab not in masks:
            log("label", lab, name, "absent"); continue
        mask = masks[lab]
        host = constrain.get(lab)
        if host is not None and host in masks:
            mask = constrain_to(mask, masks[host], spacing)
            mask = clean_components(mask, largest_only=False)
        mesh = mask_to_mesh(mask, img)
        if mesh is None:
            log("label", lab, name, "too small after cleanup"); continue
        path = os.path.join(out_dir, name + ".stl")
        mesh.export(path)
        log("wrote", name, len(mesh.vertices), "verts ->", path)
        out[name] = path
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
    nifti = series_to_nifti(local, series_id, os.path.join(work_dir, "input", "CT_%s.nii.gz" % region))
    if region == "head":
        # direct DentalSegmentator (fast, shows a percentage); MOOSE is the fallback
        label_nii = run_dental_nnunet(nifti, work_dir) or run_moose(nifti, cfg["moose_model"], work_dir)
    else:
        label_nii = run_moose(nifti, cfg["moose_model"], work_dir)
    stls = labels_to_stls(label_nii, cfg, os.path.join(work_dir, "stl"))
    bundle = save_edit_bundle(nifti, label_nii, cfg, os.path.join(work_dir, "edit"))
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
    print(json.dumps(result))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
