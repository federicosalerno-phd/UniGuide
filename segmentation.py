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

def list_series(root):
    """Enumerate DICOM series ANYWHERE under ``root`` (recursively), richest first.

    The user can point at the patient folder, or any folder above the DICOM, and we find
    every series in the subtree. Each entry carries the exact ``folder`` it lives in so the
    caller can read it, plus a ``rel`` path shown in the UI so the user knows what it is.
    """
    reader = sitk.ImageSeriesReader()
    out, seen = [], set()
    for dirpath, _dirs, files in os.walk(root):
        if not files:
            continue
        try:
            ids = reader.GetGDCMSeriesIDs(dirpath)
        except Exception:
            continue
        if not ids:
            continue
        try:
            rel = os.path.relpath(dirpath, root)
        except Exception:
            rel = dirpath
        log("scanning:", rel)
        for sid in ids:
            if (dirpath, sid) in seen:
                continue
            seen.add((dirpath, sid))
            fnames = reader.GetGDCMSeriesFileNames(dirpath, sid)
            if not fnames:
                continue
            fr = sitk.ImageFileReader()
            fr.SetFileName(fnames[0])
            try:
                fr.ReadImageInformation()
            except Exception:
                continue

            def meta(k):
                try:
                    return fr.GetMetaData(k).strip()
                except Exception:
                    return ""

            out.append({
                "series_id": sid,
                "folder": dirpath,
                "rel": rel,
                "description": meta("0008|103e"),
                "modality": meta("0008|0060"),
                "thickness": meta("0018|0050"),
                "slices": len(fnames),
            })
    out.sort(key=lambda r: -r["slices"])
    return out


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


def segment(dicom_folder, series_id, region, work_dir):
    if region not in REGIONS:
        raise ValueError("unknown region %r (expected head|leg)" % region)
    cfg = REGIONS[region]
    os.makedirs(work_dir, exist_ok=True)
    nifti = series_to_nifti(dicom_folder, series_id, os.path.join(work_dir, "input", "CT_%s.nii.gz" % region))
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
