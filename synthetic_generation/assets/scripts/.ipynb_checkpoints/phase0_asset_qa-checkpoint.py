#!/usr/bin/env python3

import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- must match generate_dataset.py exactly --------------------------------
ASSETS_DIR = f"{PROJECT_DIR}/assets"
CLASS_IDS = {"crop": 0, "weed": 1}
CLASS_TARGET_FOOTPRINT = {"crop": 0.018, "weed": 0.09}
FOCAL_LENGTH = 24.0
HORIZ_APERTURE = 20.955
# ----------------------------------------------------------------------------

from isaacsim import SimulationApp  # noqa: E402

QA_W, QA_H = 1024, 1024
simulation_app = SimulationApp({"headless": True, "width": QA_W, "height": QA_H})

import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import numpy as np
import glob
import csv
from PIL import Image

OUT_DIR = f"{PROJECT_DIR}/output/asset_qa"
os.makedirs(f"{OUT_DIR}/renders", exist_ok=True)

# Normalise every asset to the same size for QA so the test is fair across
# classes. Camera height chosen so this spans ~600 px -> plenty of detail.
QA_FOOTPRINT = 0.10
QA_TARGET_PX = 400.0
YAWS = [0, 45, 90, 135, 180, 225, 270, 315]

# thresholds -> a row is flagged if any of these trip
MIN_VISIBLE_PX = 400          # at 600-px nominal span, anything less is broken
MAX_SEM_OVER_VIS = 1.35       # segmentation may exceed RGB slightly (AA only)
MIN_BOX_FILL = 0.12           # visible px / annotator box area
MAX_FOOTPRINT_ERR = 0.15      # measured vs requested world footprint

stage = omni.usd.get_context().get_stage()


# ---------- copied verbatim from the production generator -------------------
def discover_assets(assets_dir):
    found = {}
    for cls in CLASS_IDS:
        pattern = os.path.join(assets_dir, cls, "**", "*.usd*")
        paths = sorted(
            p for p in glob.glob(pattern, recursive=True)
            if p.lower().endswith((".usd", ".usda", ".usdc"))
        )
        found[cls] = paths
    return found


def bounds_from_points(st, root=None):
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mn = np.array([np.inf] * 3)
    mx = np.array([-np.inf] * 3)
    found = False
    rng_iter = (
        Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
        if root is not None else
        Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
    )
    for prim in rng_iter:
        if not prim.IsA(UsdGeom.PointBased):
            continue
        pts = UsdGeom.PointBased(prim).GetPointsAttr().Get()
        if not pts:
            continue
        arr = np.array(pts, dtype=np.float64)
        lmn, lmx = arr.min(axis=0), arr.max(axis=0)
        m = xf_cache.GetLocalToWorldTransform(prim)
        for cx in (lmn[0], lmx[0]):
            for cy in (lmn[1], lmx[1]):
                for cz in (lmn[2], lmx[2]):
                    w = m.Transform(Gf.Vec3d(cx, cy, cz))
                    mn = np.minimum(mn, [w[0], w[1], w[2]])
                    mx = np.maximum(mx, [w[0], w[1], w[2]])
                    found = True
    if not found:
        return None
    return Gf.Range3d(Gf.Vec3d(*mn), Gf.Vec3d(*mx))


def compute_bounds(st):
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    candidates = []
    if st.GetDefaultPrim():
        candidates.append(("authored extents", st.GetDefaultPrim()))
    candidates.append(("authored extents (pseudo-root)", st.GetPseudoRoot()))
    for label, prim in candidates:
        rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not rng.IsEmpty():
            return rng, label
    rng = bounds_from_points(st)
    if rng is not None and not rng.IsEmpty():
        return rng, "raw mesh points"
    return None, "no geometry points found anywhere in the file"


def measure_asset(usd_path, target_footprint):
    src = Usd.Stage.Open(usd_path)
    if src is None:
        return None, f"cannot open {usd_path}"
    src_up = UsdGeom.GetStageUpAxis(src)
    src_mpu = UsdGeom.GetStageMetersPerUnit(src)
    rng, how = compute_bounds(src)
    if rng is None:
        return None, how
    mn, mx = Gf.Vec3d(rng.GetMin()), Gf.Vec3d(rng.GetMax())
    y_up = (src_up == UsdGeom.Tokens.y)
    rot = Gf.Matrix4d(1.0)
    if y_up:
        rot.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0))
    corners = [Gf.Vec3d(x, y, z)
               for x in (mn[0], mx[0])
               for y in (mn[1], mx[1])
               for z in (mn[2], mx[2])]
    rc = [rot.Transform(c) for c in corners]
    rmn = Gf.Vec3d(*(min(c[i] for c in rc) for i in range(3)))
    rmx = Gf.Vec3d(*(max(c[i] for c in rc) for i in range(3)))
    raw_fp = max(rmx[0] - rmn[0], rmx[1] - rmn[1])
    if raw_fp <= 0:
        return None, "degenerate bounds"
    scale = target_footprint / raw_fp
    offset = Gf.Vec3d(-0.5 * (rmn[0] + rmx[0]) * scale,
                      -0.5 * (rmn[1] + rmx[1]) * scale,
                      -rmn[2] * scale)
    return {
        "path": usd_path,
        "name": os.path.splitext(os.path.basename(usd_path))[0],
        "scale": scale, "offset": offset, "y_up": y_up,
        "raw_footprint": raw_fp, "mpu": src_mpu, "how": how,
    }, None
# ---------------------------------------------------------------------------


assets = discover_assets(ASSETS_DIR)
CAM_H = QA_TARGET_PX * (QA_FOOTPRINT / QA_TARGET_PX)  # placeholder, set below
# Solve: GSD = footprint/target_px  and  GSD = (ap/f)*H/W  ->  H
GSD_QA = QA_FOOTPRINT / QA_TARGET_PX
CAM_H = GSD_QA * QA_W / (HORIZ_APERTURE / FOCAL_LENGTH)
print(f"[qa] camera height {CAM_H:.3f} m, GSD {GSD_QA*1000:.4f} mm/px, "
      f"nominal span {QA_TARGET_PX:.0f} px")

with rep.new_layer():
    # NO ground plane. Background stays empty so the rgb alpha channel gives
    # an exact foreground mask.
    camera = rep.create.camera(
        position=(0.0, 0.0, CAM_H),
        rotation=(0.0, 0.0, 0.0),
        focal_length=FOCAL_LENGTH,
        horizontal_aperture=HORIZ_APERTURE,
        clipping_range=(0.005, 10.0),
    )
    rp = rep.create.render_product(camera, resolution=(QA_W, QA_H))

    rep.create.light(light_type="Distant", intensity=4000,
                     rotation=(55, 0, 45), color=(1, 1, 1))
    rep.create.light(light_type="Dome", intensity=600, color=(1, 1, 1))

    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    seg_annot = rep.AnnotatorRegistry.get_annotator("semantic_segmentation")
    rgb_annot.attach([rp])
    bbox_annot.attach([rp])
    seg_annot.attach([rp])

for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Camera):
        cg = UsdGeom.Camera(prim)
        cg.GetHorizontalApertureAttr().Set(HORIZ_APERTURE)
        cg.GetVerticalApertureAttr().Set(HORIZ_APERTURE * (QA_H / QA_W))


def build_subject(info, yaw_deg):
    """Rebuild the subject prim from scratch each time. Slower but immune to
    stale-transform effects, which is the point of a QA harness."""
    root = "/World/QASubject"
    if stage.GetPrimAtPath(root):
        stage.RemovePrim(Sdf.Path(root))
    xf = UsdGeom.Xform.Define(stage, root)
    xf.AddRotateZOp().Set(float(yaw_deg))
    xf.AddTranslateOp().Set(info["offset"])
    if info["y_up"]:
        xf.AddRotateXOp().Set(90.0)
    s = info["scale"]
    xf.AddScaleOp().Set(Gf.Vec3f(s, s, s))
    geo = UsdGeom.Xform.Define(stage, f"{root}/geo")
    geo.GetPrim().GetReferences().AddReference(info["path"])
    # Semantics via USD so we do not depend on a rep node context here.
    from pxr import Semantics
    sem = Semantics.SemanticsAPI.Apply(xf.GetPrim(), "Semantics")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set("subject")
    return root


def foreground_mask(rgb):
    """Exact if the renderer gives us alpha; excess-green fallback if not."""
    if rgb.shape[2] == 4 and rgb[:, :, 3].min() < 250:
        return rgb[:, :, 3] > 16, "alpha"
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    return (2 * g - r - b) > 20, "excess_green"


def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


rows = []
thumbs = []

for cls, paths in assets.items():
    for path in paths:
        info, err = measure_asset(path, QA_FOOTPRINT)
        if info is None:
            rows.append(dict(cls=cls, asset=os.path.basename(path), yaw="-",
                             status="UNREADABLE", note=err))
            print(f"[{cls}] {os.path.basename(path)}: UNREADABLE ({err})")
            continue

        for yaw in YAWS:
            build_subject(info, yaw)
            rep.orchestrator.step(rt_subframes=24)
            rgb = rgb_annot.get_data()
            bb = bbox_annot.get_data()
            seg = seg_annot.get_data()

            fg, fg_src = foreground_mask(rgb)
            vis_px = int(fg.sum())
            vis_box = mask_bbox(fg)

            # semantic mask: everything that is not the background id
            seg_arr = seg["data"] if isinstance(seg, dict) else seg
            seg_arr = np.asarray(seg_arr)
            if seg_arr.ndim == 3:
                seg_arr = seg_arr[:, :, 0]
            bg_id = np.bincount(seg_arr.ravel()).argmax()
            sem_mask = seg_arr != bg_id
            sem_px = int(sem_mask.sum())

            ann_box = None
            if len(bb["data"]) > 0:
                b0 = bb["data"][0]
                ann_box = (float(b0["x_min"]), float(b0["y_min"]),
                           float(b0["x_max"]), float(b0["y_max"]))

            ann_area = ((ann_box[2] - ann_box[0]) * (ann_box[3] - ann_box[1])
                        if ann_box else 0.0)
            fill = vis_px / ann_area if ann_area > 0 else 0.0
            sem_over_vis = sem_px / vis_px if vis_px > 0 else float("inf")

            meas_fp = ((max(vis_box[2] - vis_box[0], vis_box[3] - vis_box[1])
                        * GSD_QA) if vis_box else 0.0)
            fp_err = abs(meas_fp - QA_FOOTPRINT) / QA_FOOTPRINT

            flags = []
            if vis_px < MIN_VISIBLE_PX:
                flags.append("INVISIBLE")
            if sem_over_vis > MAX_SEM_OVER_VIS:
                flags.append("SEG>RGB")
            if ann_box is not None and fill < MIN_BOX_FILL:
                flags.append("HOLLOW_BOX")
            if ann_box is None:
                flags.append("NO_BOX")
            if fp_err > MAX_FOOTPRINT_ERR:
                flags.append("FOOTPRINT")

            tag = f"{cls}_{info['name']}_yaw{yaw:03d}"
            Image.fromarray(rgb[:, :, :3]).save(
                f"{OUT_DIR}/renders/{tag}.png")
            if flags:
                thumbs.append((tag, rgb[:, :, :3], ann_box, ",".join(flags)))

            rows.append(dict(
                cls=cls, asset=info["name"], yaw=yaw,
                status="FLAG:" + ",".join(flags) if flags else "OK",
                visible_px=vis_px, sem_px=sem_px,
                sem_over_vis=round(sem_over_vis, 3),
                box_fill=round(fill, 3),
                measured_footprint_m=round(meas_fp, 4),
                requested_footprint_m=QA_FOOTPRINT,
                footprint_err_pct=round(100 * fp_err, 1),
                raw_footprint_m=round(info["raw_footprint"], 4),
                metersPerUnit=info["mpu"],
                bounds_from=info["how"],
                fg_source=fg_src,
                note="",
            ))
            print(f"[{cls}] {info['name']:<26s} yaw={yaw:3d}  "
                  f"vis={vis_px:7d} sem/vis={sem_over_vis:5.2f} "
                  f"fill={fill:5.2f} fp={meas_fp*100:5.1f}cm  "
                  f"{'FLAG:' + ','.join(flags) if flags else 'ok'}")

# ------------------------------------------------------------- outputs -----
fields = ["cls", "asset", "yaw", "status", "visible_px", "sem_px",
          "sem_over_vis", "box_fill", "measured_footprint_m",
          "requested_footprint_m", "footprint_err_pct", "raw_footprint_m",
          "metersPerUnit", "bounds_from", "fg_source", "note"]
with open(f"{OUT_DIR}/asset_qa.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)

if thumbs:
    TH = 256
    cols = min(6, len(thumbs))
    rowsn = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * TH, rowsn * TH), (30, 30, 30))
    for k, (tag, arr, box, flg) in enumerate(thumbs):
        im = Image.fromarray(arr).resize((TH, TH), Image.LANCZOS)
        sheet.paste(im, ((k % cols) * TH, (k // cols) * TH))
    sheet.save(f"{OUT_DIR}/contact_sheet.png")

n_flag = sum(1 for r in rows if str(r.get("status", "")).startswith("FLAG"))
print("\n" + "=" * 74)
print(f"STEP 2: {len(rows)} renders, {n_flag} flagged "
      f"({'PASS' if n_flag == 0 else 'FAIL - quarantine or fix the flagged assets'})")
print(f"wrote {OUT_DIR}/asset_qa.csv")
if thumbs:
    print(f"wrote {OUT_DIR}/contact_sheet.png")
print("=" * 74)

simulation_app.close()
