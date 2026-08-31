import time

_T0 = time.time()

from isaacsim import SimulationApp

IMAGE_W, IMAGE_H = 1920, 1088
FINAL_W, FINAL_H = 960, 544
NUM_IMAGES = 500

simulation_app = SimulationApp(
    {"headless": True, "width": IMAGE_W, "height": IMAGE_H}
)

import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, UsdLux, UsdShade, Sdf, Gf
import numpy as np
import math
import glob
import os
import random
import json
from PIL import Image


def elapsed():
    return time.time() - _T0


def hms(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


PROJECT_DIR = "/isaac_sim_new_model_fixed"
ASSETS_DIR = f"{PROJECT_DIR}/assets"
OUTPUT_DIR = "/isaac_output"
SOIL_TEX_DIR = f"{ASSETS_DIR}/ground/soil_textures"

CLASS_IDS = {"crop": 0, "weed": 1}
COCO_CATEGORY_ID_START = 1

WRITE_COCO = True
WRITE_YOLO = True
WRITE_TAO_SPEC = True
VAL_FRACTION = 0.2
SPLIT_SEED = 1234

# Draw boxes on a few frames so coordinate bugs are visible immediately
DEBUG_OVERLAY_COUNT = 12

# Plant scale
CROP_SIZE_MODE = "preserve"

# Growth stages
CROP_STAGES = {
    "cotyledon": {"weight": 0.30, "fp": (0.025, 0.055), "assets": ["v14"]},
    "early": {"weight": 0.55, "fp": (0.06, 0.11), "assets": ["v09", "v12", "v13"]},
    "mid": {
        "weight": 0.15,
        "fp": (0.12, 0.20),
        "assets": ["v01", "v02", "v03", "v08", "v10", "v11"],
    },
}

CROP_MAX_STRETCH = 1.6
CROP_FOOTPRINT_CLAMP = (0.02, 0.55)
CENTER_ON_CROWN = True
CROWN_PRIM_NAMES = ("crown",)

CLASS_TARGET_FOOTPRINT = {
    "crop": 0.18,
    "weed": 0.025,
}

CROP_SCALE_JITTER = (0.85, 1.15)
WEED_FOOTPRINT_RANGE = (0.008, 0.032)

# Scene layout
CROP_ROW_SPACING = 0.50
CROP_IN_ROW_SPACING = 0.22
CROP_ROW_JITTER = 0.030
CROP_ALONG_JITTER = 0.045
CROP_MISSING_PROB = 0.18

WEEDS_PER_FRAME = (2, 10)

MAX_CROP_SLOTS = 20
MAX_WEED_SLOTS = 12
VARIANTS_PER_SLOT = 6

# Clutter: straw and stones
CLUTTER_ENABLED = True

STRAW_PER_FRAME = (0, 22)
STRAW_LENGTH = (0.020, 0.090)
STRAW_WIDTH = (0.002, 0.004)
STRAW_BEND = (0.0, 0.12)

STRAW_COLORS_SRGB = [
    (0.81, 0.79, 0.71),
    (0.72, 0.67, 0.55),
    (0.63, 0.58, 0.46),
    (0.55, 0.49, 0.38),
    (0.86, 0.83, 0.74),
]
CLUTTER_COLOR_JITTER = 0.10
STRAW_ROUGHNESS = (0.88, 0.98)

STONES_PER_FRAME = (0, 30)
STONE_SIZE = (0.005, 0.025)
STONE_COLORS_SRGB = [
    (0.60, 0.56, 0.50),
    (0.48, 0.45, 0.40),
    (0.40, 0.37, 0.33),
    (0.33, 0.31, 0.28),
    (0.52, 0.47, 0.39),
]
STONE_ROUGHNESS = (0.86, 0.97)
STONE_BURIAL = (0.10, 0.40)

MAX_STRAW_SLOTS = 24
MAX_STONE_SLOTS = 32

LAYOUT_MARGIN = 0.07
EDGE_CLIP_CROPS = 2
EDGE_CLIP_WEED_PROB = 0.15

# Camera
FOCAL_LENGTH = 24.0
HORIZ_APERTURE = 20.955
CAM_HEIGHT = (1.05, 1.35)
CAM_XY_JITTER = 0.05
CAM_TILT = 2.0

GROUND_SIZE = 12.0
CAM_TILE_RANGE = 2

MUD_TILES = 0.35

UV_CENTER = 0.5
UV_JITTER = 0.04

MIN_BOX_PX_OUT = 5.0

_SCALE_X = IMAGE_W / float(FINAL_W)
_SCALE_Y = IMAGE_H / float(FINAL_H)
assert abs(_SCALE_X - _SCALE_Y) < 1e-6, (
    f"render/output scale differs per axis ({_SCALE_X} vs {_SCALE_Y}); "
    f"MIN_BOX_PX must be split into x and y before using this configuration"
)
MIN_BOX_PX = MIN_BOX_PX_OUT * _SCALE_X

SOIL_DETAIL_FRACTION = 0.40
SOIL_DETAIL_FLOOR = 0.60

SOIL_RESTEP_MAX = 4
SOIL_RETRY_SAME = 2
SOIL_MAX_ATTEMPTS = 4
WARMUP_MAX_STEPS = 12

# Plant residency
USE_INSTANCING = False

SETTLE_UPDATES = 3
PLANT_RESTEP_MAX = 3
PLANT_YIELD_MIN = 0.60
PLANT_VERIFY_MIN_PLACED = 3

SAVE_DISCARDED_FRAMES = True

PLANT_MIN_GREEN = 1.0
PLANT_MIN_BOXES = 1

# Run robustness
MAX_CONSECUTIVE_FAILURES = 8
MAX_FAILURE_FRACTION = 0.25

COCO_FLUSH_EVERY = 50

MEASURE_CACHE = True
MEASURE_CACHE_PATH = f"{OUTPUT_DIR}/asset_measurements.json"

# Lighting
HARSH_PROB = 0.72

LIGHT_HARSH = {
    "sun_intensity": (4200.0, 7000.0),
    "sun_angle": (0.35, 1.10),
    "sun_zenith": (24.0, 44.0),
    "sun_color": (1.00, 0.96, 0.88),
    "dome_intensity": (650.0, 950.0),
    "dome_color": (0.45, 0.55, 0.78),
    "fill_intensity": (290.0, 480.0),
}

LIGHT_OVERCAST = {
    "sun_intensity": (250.0, 1100.0),
    "sun_angle": (12.0, 26.0),
    "sun_zenith": (10.0, 45.0),
    "sun_color": (0.97, 0.97, 0.98),
    "dome_intensity": (750.0, 1500.0),
    "dome_color": (0.72, 0.74, 0.78),
    "fill_intensity": (0.0, 120.0),
}

# First-run verification
RULER_CUBE = False
RULER_SIZE = 0.10

SOIL_EXCLUDE = ("crack", "playa", "desert", "lakebed", "drought")

SOIL_REF_RGB = (157.0, 151.0, 135.0)
SOIL_TINT_CLAMP = 0.40
SOIL_TINT_CEIL = 1.60
SOIL_AUTO_TINT = True

SOIL_TINT_ITERS = 5
SOIL_TINT_GAIN = 2.5
SOIL_TINT_TOL = 0.015

SOIL_TINTS = [
    (0.878, 0.926, 1.000),
    (0.850, 0.910, 1.000),
    (0.910, 0.945, 1.000),
    (0.790, 0.833, 0.900),
    (0.700, 0.740, 0.800),
    (0.940, 0.960, 0.985),
]

SPLITS = ("train", "val")
for _split in SPLITS:
    os.makedirs(f"{OUTPUT_DIR}/images/{_split}", exist_ok=True)
    if WRITE_YOLO:
        os.makedirs(f"{OUTPUT_DIR}/labels/{_split}", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/annotations", exist_ok=True)
if DEBUG_OVERLAY_COUNT:
    os.makedirs(f"{OUTPUT_DIR}/debug_overlays", exist_ok=True)

_split_rng = random.Random(SPLIT_SEED)
_indices = list(range(NUM_IMAGES))
_split_rng.shuffle(_indices)
_n_val = max(1, int(round(NUM_IMAGES * VAL_FRACTION))) if NUM_IMAGES > 1 else 0
_val_set = set(_indices[:_n_val])
frame_split = {
    i: ("val" if i in _val_set else "train") for i in range(NUM_IMAGES)
}
print(f"[split] train={NUM_IMAGES - _n_val} val={_n_val} (seed {SPLIT_SEED})")

# COCO accumulators
coco_images = {s: [] for s in SPLITS}
coco_annotations = {s: [] for s in SPLITS}
coco_ann_id = {s: 1 for s in SPLITS}

coco_category_id = {
    name: idx + COCO_CATEGORY_ID_START for name, idx in CLASS_IDS.items()
}
COCO_NUM_CLASSES = max(coco_category_id.values()) + 1

if WRITE_TAO_SPEC and COCO_CATEGORY_ID_START < 1:
    raise ValueError(
        "TAO reserves category_id 0 for background. "
        "Set COCO_CATEGORY_ID_START = 1, or disable WRITE_TAO_SPEC."
    )

stage = omni.usd.get_context().get_stage()
rng = random.Random()

EXPOSURE_MODE = "auto"

TONEMAP_OP_ACES = 5
MANUAL_EXPOSURE = {
    "film_iso": 100.0,
    "camera_shutter": 1.0 / 125.0,
    "f_number": 8.0,
}


def configure_exposure(mode):
    if mode == "auto":
        print("[light] exposure: auto")
        return
    try:
        import carb

        s = carb.settings.get_settings()
        s.set("/rtx/post/histogram/enabled", False)
        s.set("/rtx/post/tonemap/op", TONEMAP_OP_ACES)
        s.set("/rtx/post/tonemap/filmIso", MANUAL_EXPOSURE["film_iso"])
        s.set(
            "/rtx/post/tonemap/cameraShutter", MANUAL_EXPOSURE["camera_shutter"]
        )
        s.set("/rtx/post/tonemap/fNumber", MANUAL_EXPOSURE["f_number"])
        print(
            f"[light] exposure: manual, ACES, iso={MANUAL_EXPOSURE['film_iso']:.0f} "
            f"shutter=1/{1.0 / MANUAL_EXPOSURE['camera_shutter']:.0f} "
            f"f/{MANUAL_EXPOSURE['f_number']:.1f}"
        )
    except Exception as exc:
        print(f"[WARN] could not set exposure settings: {exc}")


configure_exposure(EXPOSURE_MODE)


# Semantics API
def apply_semantics(prim, label):
    try:
        from pxr import UsdSemantics

        api = UsdSemantics.LabelsAPI.Apply(prim, "class")
        api.CreateLabelsAttr().Set([label])
        return
    except Exception:
        pass

    try:
        from pxr import Semantics

        sem = Semantics.SemanticsAPI.Apply(prim, "Semantics")
        sem.CreateSemanticTypeAttr().Set("class")
        sem.CreateSemanticDataAttr().Set(label)
        return
    except Exception as exc:
        raise RuntimeError(f"no usable semantics API: {exc}")


# Asset discovery
def discover_assets(assets_dir):
    found = {}
    for cls in CLASS_IDS:
        pattern = os.path.join(assets_dir, cls, "**", "*.usd*")
        paths = sorted(
            p
            for p in glob.glob(pattern, recursive=True)
            if p.lower().endswith((".usd", ".usda", ".usdc"))
        )
        found[cls] = paths
    return found


def discover_soil_textures(tex_dir):
    if not os.path.isdir(tex_dir):
        return []

    role_markers = {
        "diffuse": ("_diff", "_albedo", "_basecolor", "_base_color"),
        "normal": ("_nor_gl", "_normal_gl", "_normal", "_nor"),
        "roughness": ("_roughness", "_rough"),
        "specular": ("_specular", "_spec"),
        "displacement": ("_displacement", "_height", "_disp"),
    }

    supported_extensions = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".exr")
    soil_sets = []

    for root, _, filenames in os.walk(tex_dir):
        roles = {}
        for filename in sorted(filenames):
            lower = filename.lower()
            if not lower.endswith(supported_extensions):
                continue

            full_path = os.path.join(root, filename)
            for role, markers in role_markers.items():
                if not any(marker in lower for marker in markers):
                    continue
                if role == "normal" and lower.endswith(".exr"):
                    print("[WARN] skipping EXR normal map")
                    break
                roles[role] = full_path
                break

        if "diffuse" not in roles:
            continue

        soil_sets.append(
            {
                "name": os.path.basename(root),
                "directory": root,
                "diffuse": roles["diffuse"],
                "normal": roles.get("normal", ""),
                "roughness": roles.get("roughness", ""),
                "specular": roles.get("specular", ""),
                "displacement": roles.get("displacement", ""),
            }
        )

    soil_sets.sort(key=lambda item: item["name"])

    if SOIL_EXCLUDE:
        kept, dropped = [], []
        for soil in soil_sets:
            if any(k in soil["name"].lower() for k in SOIL_EXCLUDE):
                dropped.append(soil["name"])
            else:
                kept.append(soil)

        if dropped:
            print(f"[soil] excluded set by SOIL_EXCLUDE: {', '.join(dropped)}")
        if not kept:
            print("[WARN] SOIL_EXCLUDE removed every soil texture set")
            kept = soil_sets
        soil_sets = kept

    return soil_sets


def bounds_from_points(st, root=None):
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mn = np.array([np.inf] * 3)
    mx = np.array([-np.inf] * 3)
    found = False

    rng_iter = (
        Usd.PrimRange(root, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
        if root is not None
        else Usd.PrimRange.Stage(
            st, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        )
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
        rng_ = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not rng_.IsEmpty():
            return rng_, label

    rng_ = bounds_from_points(st)
    if rng_ is not None and not rng_.IsEmpty():
        return rng_, "raw mesh points"

    return None, "no geometry points found anywhere in the file"


def find_crown_bounds(st, rot):
    dflt = st.GetDefaultPrim()
    for prim in Usd.PrimRange.Stage(
        st, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    ):
        if dflt and prim == dflt:
            continue

        nm = prim.GetName().lower()
        if not any(k in nm for k in CROWN_PRIM_NAMES):
            continue

        rng_ = bounds_from_points(st, prim)
        if rng_ is None or rng_.IsEmpty():
            continue

        mn, mx = Gf.Vec3d(rng_.GetMin()), Gf.Vec3d(rng_.GetMax())
        corners = [
            Gf.Vec3d(x, y, z)
            for x in (mn[0], mx[0])
            for y in (mn[1], mx[1])
            for z in (mn[2], mx[2])
        ]
        rc = [rot.Transform(c) for c in corners]
        cx = 0.5 * (min(c[0] for c in rc) + max(c[0] for c in rc))
        cy = 0.5 * (min(c[1] for c in rc) + max(c[1] for c in rc))
        return cx, cy, prim.GetName()

    return None


def measure_asset(usd_path, target_footprint, cls="weed"):
    src = Usd.Stage.Open(usd_path)
    if src is None:
        return None, f"cannot open {usd_path}"

    src_up = UsdGeom.GetStageUpAxis(src)
    src_mpu = UsdGeom.GetStageMetersPerUnit(src)
    rng_, how = compute_bounds(src)
    if rng_ is None:
        return None, how

    mn, mx = Gf.Vec3d(rng_.GetMin()), Gf.Vec3d(rng_.GetMax())

    y_up = src_up == UsdGeom.Tokens.y
    rot = Gf.Matrix4d(1.0)
    if y_up:
        rot.SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), 90.0))

    corners = [
        Gf.Vec3d(x, y, z)
        for x in (mn[0], mx[0])
        for y in (mn[1], mx[1])
        for z in (mn[2], mx[2])
    ]
    rc = [rot.Transform(c) for c in corners]
    rmn = Gf.Vec3d(*(min(c[i] for c in rc) for i in range(3)))
    rmx = Gf.Vec3d(*(max(c[i] for c in rc) for i in range(3)))

    raw_fp = max(rmx[0] - rmn[0], rmx[1] - rmn[1])
    if raw_fp <= 0:
        return None, "degenerate bounds"

    if cls == "crop" and CROP_SIZE_MODE == "preserve":
        scale = 1.0
        scale_note = "native"
        if CROP_FOOTPRINT_CLAMP:
            lo, hi = CROP_FOOTPRINT_CLAMP
            if raw_fp > hi:
                scale = hi / raw_fp
                scale_note = f"CLAMPED down from {raw_fp * 1000:.0f}mm"
            elif raw_fp < lo:
                scale = lo / raw_fp
                scale_note = f"CLAMPED up from {raw_fp * 1000:.0f}mm"
    else:
        scale = target_footprint / raw_fp
        scale_note = "normalized"

    cx = 0.5 * (rmn[0] + rmx[0])
    cy = 0.5 * (rmn[1] + rmx[1])
    center_note = "bbox"

    if CENTER_ON_CROWN:
        cr = find_crown_bounds(src, rot)
        if cr is not None:
            ccx, ccy, cname = cr
            offset_mm = math.hypot(ccx - cx, ccy - cy) * 1000.0
            cx, cy = ccx, ccy
            center_note = f"crown '{cname}' ({offset_mm:.0f}mm from bbox)"

    offset = Gf.Vec3d(-cx * scale, -cy * scale, -rmn[2] * scale)

    return {
        "path": usd_path,
        "name": os.path.splitext(os.path.basename(usd_path))[0],
        "scale": scale,
        "offset": offset,
        "y_up": y_up,
        "raw_footprint": raw_fp,
        "final_footprint": raw_fp * scale,
        "raw_height": rmx[2] - rmn[2],
        "final_height": (rmx[2] - rmn[2]) * scale,
        "mpu": src_mpu,
        "how": how,
        "scale_note": scale_note,
        "center_note": center_note,
    }, None

# Discovery run
assets = discover_assets(ASSETS_DIR)
soil_sets = discover_soil_textures(SOIL_TEX_DIR)

gsd_min = (HORIZ_APERTURE / FOCAL_LENGTH) * CAM_HEIGHT[0] / IMAGE_W
gsd_max = (HORIZ_APERTURE / FOCAL_LENGTH) * CAM_HEIGHT[1] / IMAGE_W
ground_w_min = gsd_min * IMAGE_W
ground_w_max = gsd_max * IMAGE_W

_meas_cache = {}
_meas_dirty = []
if MEASURE_CACHE:
    try:
        with open(MEASURE_CACHE_PATH) as f:
            _meas_cache = json.load(f)
        print(
            f"[cache] loaded {len(_meas_cache)} asset measurement from "
            f"{MEASURE_CACHE_PATH}"
        )
    except Exception:
        _meas_cache = {}


def _measure_key(path, cls):
    try:
        stat = os.stat(path)
        stamp = f"{stat.st_mtime:.0f}:{stat.st_size}"
    except OSError:
        stamp = "missing"

    return "|".join(
        [
            path,
            cls,
            stamp,
            CROP_SIZE_MODE,
            str(CROP_FOOTPRINT_CLAMP),
            str(CENTER_ON_CROWN),
            str(CROWN_PRIM_NAMES),
            str(CLASS_TARGET_FOOTPRINT[cls]),
        ]
    )


def measure_asset_cached(path, target, cls):
    if not MEASURE_CACHE:
        return measure_asset(path, target, cls=cls)

    key = _measure_key(path, cls)
    hit = _meas_cache.get(key)
    if hit is not None:
        info = dict(hit)
        info["offset"] = Gf.Vec3d(*hit["offset"])
        return info, None

    info, err = measure_asset(path, target, cls=cls)
    if info is not None:
        stored = dict(info)
        stored["offset"] = [float(v) for v in info["offset"]]
        _meas_cache[key] = stored
        _meas_dirty.append(1)

    return info, err


templates = {}
for cls, paths in assets.items():
    templates[cls] = []
    print(f"[{cls}] {len(paths)} assets")

    if not paths:
        print(f"[WARN] No assets found in {ASSETS_DIR}/{cls}")
        continue

    for path in paths:
        info, err = measure_asset_cached(
            path, CLASS_TARGET_FOOTPRINT[cls], cls
        )

        if info is None:
            print(f"  SKIP {os.path.basename(path)}: {err}")
            continue

        if cls == "crop":
            fp_lo = info["final_footprint"] * CROP_SCALE_JITTER[0]
            fp_hi = info["final_footprint"] * CROP_SCALE_JITTER[1]
        else:
            fp_lo, fp_hi = WEED_FOOTPRINT_RANGE

        px_min = fp_lo / gsd_max
        px_max = fp_hi / gsd_min
        print(
            f"  {info['name']}: "
            f"{info['final_footprint']:.3f} m, "
            f"scale={info['scale']:.3f}, "
            f"~{px_min * FINAL_W / IMAGE_W:.0f}-"
            f"{px_max * FINAL_W / IMAGE_W:.0f}px"
        )

        if info["raw_footprint"] > 1.0:
            print(
                f"  [WARN] Large raw footprint: "
                f"{info['raw_footprint']:.2f} m"
            )

        templates[cls].append(info)

if MEASURE_CACHE and _meas_dirty:
    try:
        os.makedirs(os.path.dirname(MEASURE_CACHE_PATH), exist_ok=True)
        with open(MEASURE_CACHE_PATH, "w") as f:
            json.dump(_meas_cache, f)
        print(f"[cache] saved {len(_meas_dirty)} measurements")
    except Exception as exc:
        print(f"[WARN] Could not save measurement cache: {exc}")

print(f"[soil] {len(soil_sets)} texture sets")
print(
    f"[camera] {FOCAL_LENGTH} mm, "
    f"{CAM_HEIGHT[0]}-{CAM_HEIGHT[1]} m, "
    f"GSD {gsd_min * 1000:.3f}-{gsd_max * 1000:.3f} mm/px"
)

_tile_period = 1.0 / (MUD_TILES * 1.10)
_worst_span = ground_w_max * math.sqrt(2.0)
if _worst_span >= _tile_period:
    print(
        f"[WARN] Soil texture period ({_tile_period:.2f} m) "
        f"is smaller than visible span ({_worst_span:.2f} m)"
    )

if RULER_CUBE:
    print(
        f"[ruler] expected size: "
        f"{RULER_SIZE / gsd_max:.0f}-"
        f"{RULER_SIZE / gsd_min:.0f}px"
    )

active_classes = [c for c in CLASS_IDS if templates.get(c)]
assert active_classes, "No assets found. Check ASSETS_DIR and the folder structure."

if "crop" not in active_classes:
    print("[WARN] No crop assets; generating weed-only data")


# Materials
def make_soil_material(stage, path, soil, color, tiling=1.0):
    mtl = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/Shader"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

    shader.CreateInput(
        "diffuse_color_constant", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*color))
    shader.CreateInput(
        "diffuse_tint", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(1, 1, 1))
    shader.CreateInput(
        "reflection_roughness_constant", Sdf.ValueTypeNames.Float
    ).Set(0.95)

    diffuse = soil["diffuse"] if soil else ""
    normal = (soil.get("normal") or "") if soil else ""
    rough = (soil.get("roughness") or "") if soil else ""

    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(diffuse)
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(normal)
    shader.CreateInput(
        "reflectionroughness_texture", Sdf.ValueTypeNames.Asset
    ).Set(rough)

    shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(
        1.0 if normal else 0.0
    )
    shader.CreateInput(
        "reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float
    ).Set(1.0 if rough else 0.0)

    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(tiling, tiling)
    )
    shader.CreateInput("texture_translate", Sdf.ValueTypeNames.Float2).Set(
        Gf.Vec2f(UV_CENTER, UV_CENTER)
    )
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)

    mtl.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return {
        "soil": soil,
        "material": mtl,
        "shader": shader,
        "has_normal": bool(normal),
        "has_roughness": bool(rough),
        "name": soil["name"] if soil else "flat_brown",
        "baseline_detail": None,
        "tint_correction": None,
    }


# Scene build
ground = rep.create.plane(scale=(GROUND_SIZE, GROUND_SIZE, 1))

ground_prim = None
for prim in stage.Traverse():
    if prim.GetTypeName() == "Mesh" and "Plane" in prim.GetName():
        ground_prim = prim

assert ground_prim is not None, "Ground plane prim not found"

_bbox = UsdGeom.BBoxCache(
    Usd.TimeCode.Default(), ["default", "render", "proxy", "guide"]
).ComputeWorldBound(ground_prim).ComputeAlignedRange()

GROUND_HALF = min(
    abs(_bbox.GetMin()[0]),
    abs(_bbox.GetMax()[0]),
    abs(_bbox.GetMin()[1]),
    abs(_bbox.GetMax()[1]),
)
print(
    f"[ground] plane measured half-extent {GROUND_HALF:.2f} m "
    f"(GROUND_SIZE={GROUND_SIZE} m nominal)"
)

if GROUND_HALF < GROUND_SIZE / 2.0 - 1e-3:
    print(
        "[ground] plane is smaller than GROUND_SIZE; "
        "camera placement uses the measured extent"
    )

soil_materials = []
if soil_sets:
    for i, soil in enumerate(soil_sets):
        soil_materials.append(
            make_soil_material(
                stage,
                f"/World/Looks/Soil_{i:02d}",
                soil=soil,
                color=(0.35, 0.22, 0.10),
                tiling=MUD_TILES,
            )
        )
else:
    soil_materials.append(
        make_soil_material(
            stage,
            "/World/Looks/Soil_flat",
            soil=None,
            color=(0.35, 0.22, 0.10),
            tiling=MUD_TILES,
        )
    )

ground_binding_api = UsdShade.MaterialBindingAPI(ground_prim)
_bound_soil = {"entry": None}


def bind_soil_material(entry):
    if _bound_soil["entry"] is entry:
        return False
    ground_binding_api.Bind(
        entry["material"], UsdShade.Tokens.strongerThanDescendants
    )
    _bound_soil["entry"] = entry
    return True


bind_soil_material(soil_materials[0])
print(f"[soil] built {len(soil_materials)} materials")

if RULER_CUBE:
    rep.create.cube(
        position=(0.0, 0.0, RULER_SIZE / 2.0),
        scale=(RULER_SIZE, RULER_SIZE, RULER_SIZE),
    )

# Camera (USD)
cam_xform = UsdGeom.Xform.Define(stage, "/World/CameraRig")
cam_translate = cam_xform.AddTranslateOp()
cam_rotate = cam_xform.AddRotateXYZOp()
cam_translate.Set(Gf.Vec3d(0.0, 0.0, CAM_HEIGHT[1]))
cam_rotate.Set(Gf.Vec3f(0.0, 0.0, 0.0))

cam = UsdGeom.Camera.Define(stage, "/World/CameraRig/Camera")
cam.CreateFocalLengthAttr(FOCAL_LENGTH)
cam.CreateHorizontalApertureAttr(HORIZ_APERTURE)
cam.CreateVerticalApertureAttr(HORIZ_APERTURE * (IMAGE_H / float(IMAGE_W)))
cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))

rp = rep.create.render_product(
    "/World/CameraRig/Camera", resolution=(IMAGE_W, IMAGE_H)
)

# Lights (USD)
sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
sun_xform = UsdGeom.Xformable(sun.GetPrim())
sun_rotate = sun_xform.AddRotateXYZOp()
sun.CreateIntensityAttr(5000.0)
sun.CreateAngleAttr(0.53)
sun.CreateColorAttr(Gf.Vec3f(1.0, 0.96, 0.88))

fill = UsdLux.DistantLight.Define(stage, "/World/Lights/Fill")
fill_xform = UsdGeom.Xformable(fill.GetPrim())
fill_rotate = fill_xform.AddRotateXYZOp()
fill_rotate.Set(Gf.Vec3f(-20.0, 0.0, 200.0))
fill.CreateIntensityAttr(120.0)
fill.CreateAngleAttr(20.0)
fill.CreateColorAttr(Gf.Vec3f(0.85, 0.90, 1.00))

dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
dome.CreateIntensityAttr(80.0)
dome.CreateColorAttr(Gf.Vec3f(0.45, 0.55, 0.78))

# Plant slots
def gpu_mem():
    try:
        import subprocess

        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode != 0:
            return None
        used, total = out.stdout.strip().split("\n")[0].split(",")
        return int(used), int(total)
    except Exception:
        return None


_gpu0 = gpu_mem()
if _gpu0:
    print(
        f"[gpu] {_gpu0[0]} / {_gpu0[1]} MB used at startup "
        f"({_gpu0[1] - _gpu0[0]} MB free)"
    )
else:
    print("[gpu] nvidia-smi unavailable")


def build_slots(cls, count):
    slots = []
    for s in range(count):
        root = f"/World/Plants/{cls}_slot_{s:02d}"
        slot = UsdGeom.Xform.Define(stage, root)
        t_op = slot.AddTranslateOp()
        r_op = slot.AddRotateZOp()
        s_op = slot.AddScaleOp()
        t_op.Set(Gf.Vec3d(0.0, 0.0, -50.0))
        r_op.Set(0.0)
        s_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

        apply_semantics(slot.GetPrim(), cls)

        variants = []
        variant_stage = []
        n_templates = len(templates[cls])
        k = min(VARIANTS_PER_SLOT, n_templates)

        if cls == "crop":
            by_stage = {}
            for i, template in enumerate(templates[cls]):
                if "stage" not in template:
                    raise RuntimeError(
                        f"Crop template {template['name']!r} is missing stage information."
                    )
                by_stage.setdefault(template["stage"], []).append(i)

            idxs = []
            for stage_name in CROP_STAGES:
                members = by_stage.get(stage_name)
                if members and len(idxs) < k:
                    idxs.append(members[s % len(members)])

            extra = sorted(by_stage.items(), key=lambda kv: -len(kv[1]))
            j = 1
            while len(idxs) < k and extra:
                added = False
                for _, members in extra:
                    if len(idxs) >= k:
                        break
                    cand = members[(s + j) % len(members)]
                    if cand not in idxs:
                        idxs.append(cand)
                        added = True
                j += 1
                if j > n_templates or not added:
                    break
        else:
            idxs = [((s * k) + j) % n_templates for j in range(k)]

        for i in idxs:
            info = templates[cls][i]
            vpath = f"{root}/var_{i}"
            v = UsdGeom.Xform.Define(stage, vpath)
            v.AddTranslateOp().Set(info["offset"])
            if info["y_up"]:
                v.AddRotateXOp().Set(90.0)
            sc = info["scale"]
            v.AddScaleOp().Set(Gf.Vec3f(sc, sc, sc))

            geo = UsdGeom.Xform.Define(stage, f"{vpath}/geo")
            geo.GetPrim().GetReferences().AddReference(info["path"])

            if USE_INSTANCING:
                geo.GetPrim().SetInstanceable(True)

            variant_stage.append(info.get("stage"))
            UsdGeom.Imageable(v).MakeInvisible()
            variants.append(v)

        UsdGeom.Imageable(slot).MakeInvisible()
        slots.append(
            {
                "xform": slot,
                "imageable": UsdGeom.Imageable(slot.GetPrim()),
                "t": t_op,
                "r": r_op,
                "s": s_op,
                "variants": variants,
                "variant_stage": variant_stage,
                "variant_fp": [templates[cls][i]["final_footprint"] for i in idxs],
                "active_variant": None,
            }
        )

    return slots


import re as _re


def stage_of_asset(name):
    match = _re.search(r"_v(\d{1,2})_", name) or _re.search(
        r"_v(\d{1,2})$", name
    )
    if not match:
        return None

    version = f"v{int(match.group(1)):02d}"
    for stage_name, stage_cfg in CROP_STAGES.items():
        if version in stage_cfg["assets"]:
            return stage_name

    return None


stage_names = list(CROP_STAGES.keys())
for info in templates.get("crop", []):
    stage_name = stage_of_asset(info["name"])

    if stage_name is None:
        fp = info["final_footprint"]
        stage_name = min(
            stage_names,
            key=lambda name: abs(
                fp - sum(CROP_STAGES[name]["fp"]) / 2.0
            ),
        )
        print(
            f"[WARN] {info['name']}: stage inferred from footprint "
            f"({fp:.3f} m -> {stage_name})"
        )

    info["stage"] = stage_name

for stage_name in stage_names:
    members = [
        t
        for t in templates.get("crop", [])
        if t["stage"] == stage_name
    ]

    if not members:
        print(f"[WARN] No crop assets for stage '{stage_name}'")
        continue

    lo, hi = CROP_STAGES[stage_name]["fp"]
    sizes = ", ".join(
        f"{t['final_footprint'] * 100:.0f}" for t in members
    )
    print(
        f"[stage] {stage_name}: {len(members)} assets, "
        f"target={lo * 100:.0f}-{hi * 100:.0f} cm, "
        f"native={sizes} cm"
    )

slot_pool = {}
if "crop" in active_classes:
    slot_pool["crop"] = build_slots("crop", MAX_CROP_SLOTS)
if "weed" in active_classes:
    slot_pool["weed"] = build_slots("weed", MAX_WEED_SLOTS)

print(
    f"[slots] crop={len(slot_pool.get('crop', []))}, "
    f"weed={len(slot_pool.get('weed', []))}"
)


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


_clutter_rng = random.Random(SPLIT_SEED + 991)


def jittered_linear(srgb, amount):
    out = []
    for c in srgb:
        j = c * (1.0 + _clutter_rng.uniform(-amount, amount))
        out.append(srgb_to_linear(min(max(j, 0.0), 1.0)))
    return tuple(out)


def make_clutter_material(stage, path, color, rough):
    mtl = UsdShade.Material.Define(stage, Sdf.Path(path))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/Shader"))
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput(
        "diffuse_color_constant", Sdf.ValueTypeNames.Color3f
    ).Set(Gf.Vec3f(*color))
    sh.CreateInput(
        "reflection_roughness_constant", Sdf.ValueTypeNames.Float
    ).Set(rough)
    mtl.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mtl


def build_straw_mesh(stage, path, length, width, bend, sides=6, segs=7):
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts, counts, idx = [], [], []

    for j in range(segs):
        t = j / (segs - 1.0)
        r = width * 0.5 * (
            0.45 + 0.55 * math.sin(math.pi * (0.25 + 0.75 * t))
        )
        z = -bend * length * math.sin(math.pi * t)
        x = (t - 0.5) * length
        for i in range(sides):
            a = 2.0 * math.pi * i / sides
            pts.append(
                Gf.Vec3f(
                    x,
                    float(r * math.cos(a)),
                    float(z + r * math.sin(a)),
                )
            )

    for j in range(segs - 1):
        for i in range(sides):
            i2 = (i + 1) % sides
            a0 = j * sides + i
            a1 = j * sides + i2
            b0 = (j + 1) * sides + i
            b1 = (j + 1) * sides + i2
            counts.append(4)
            idx.extend([a0, a1, b1, b0])

    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateDoubleSidedAttr(True)
    lowest = min(p[2] for p in pts)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, float(-lowest))
    )
    return mesh


def build_stone_mesh(stage, path, size, seed=0, subdiv=1):
    r = size * 0.5
    t = (1.0 + math.sqrt(5.0)) / 2.0
    base = [
        (-1, t, 0),
        (1, t, 0),
        (-1, -t, 0),
        (1, -t, 0),
        (0, -1, t),
        (0, 1, t),
        (0, -1, -t),
        (0, 1, -t),
        (t, 0, -1),
        (t, 0, 1),
        (-t, 0, -1),
        (-t, 0, 1),
    ]
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    verts = [Gf.Vec3f(*v) for v in base]

    for _ in range(subdiv):
        cache, new_faces = {}, []

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key in cache:
                return cache[key]
            m = (verts[a] + verts[b]) * 0.5
            verts.append(m)
            cache[key] = len(verts) - 1
            return cache[key]

        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [
                (a, ab, ca),
                (b, bc, ab),
                (c, ca, bc),
                (ab, bc, ca),
            ]
        faces = new_faces

    lrng = random.Random(seed)
    pts = []
    for v in verts:
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
        k = r * lrng.uniform(0.72, 1.12)
        pts.append(
            Gf.Vec3f(
                float(v[0] / n * k),
                float(v[1] / n * k),
                float(v[2] / n * k * 0.60),
            )
        )

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    idx = []
    for face in faces:
        idx.extend(face)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateDoubleSidedAttr(False)
    lowest = min(p[2] for p in pts)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, float(-lowest))
    )
    return mesh


def build_clutter_slots(kind, count, n_variants=5):
    slots = []
    for s in range(count):
        root = f"/World/Clutter/{kind}_slot_{s:02d}"
        slot = UsdGeom.Xform.Define(stage, root)
        t_op = slot.AddTranslateOp()
        rz_op = slot.AddRotateZOp()
        ry_op = slot.AddRotateYOp()
        sc_op = slot.AddScaleOp()
        t_op.Set(Gf.Vec3d(0.0, 0.0, -50.0))
        rz_op.Set(0.0)
        ry_op.Set(0.0)
        sc_op.Set(Gf.Vec3f(1.0, 1.0, 1.0))

        variants = []
        for i in range(n_variants):
            vpath = f"{root}/var_{i}"
            v = UsdGeom.Xform.Define(stage, vpath)

            if kind == "straw":
                lo, hi = STRAW_LENGTH
                length = lo + (hi - lo) * (i / max(n_variants - 1, 1))
                width = (STRAW_WIDTH[0] + STRAW_WIDTH[1]) * 0.5
                bend = STRAW_BEND[0] + (STRAW_BEND[1] - STRAW_BEND[0]) * (
                    (i * 0.37) % 1.0
                )
                build_straw_mesh(stage, f"{vpath}/geo", length, width, bend)
                col = jittered_linear(
                    STRAW_COLORS_SRGB[i % len(STRAW_COLORS_SRGB)],
                    CLUTTER_COLOR_JITTER,
                )
                rough = _clutter_rng.uniform(*STRAW_ROUGHNESS)
                burial = 0.0
            else:
                lo, hi = STONE_SIZE
                size = lo + (hi - lo) * (i / max(n_variants - 1, 1))
                build_stone_mesh(
                    stage, f"{vpath}/geo", size, seed=s * 17 + i
                )
                col = jittered_linear(
                    STONE_COLORS_SRGB[i % len(STONE_COLORS_SRGB)],
                    CLUTTER_COLOR_JITTER,
                )
                rough = _clutter_rng.uniform(*STONE_ROUGHNESS)
                burial = size * _clutter_rng.uniform(*STONE_BURIAL)

            mtl = make_clutter_material(
                stage, f"/World/Looks/{kind}_{s:02d}_{i}", col, rough
            )

            if burial:
                UsdGeom.Xform(v).AddTranslateOp().Set(
                    Gf.Vec3d(0.0, 0.0, -float(burial))
                )

            geo_prim = stage.GetPrimAtPath(f"{vpath}/geo")
            UsdShade.MaterialBindingAPI(geo_prim).Bind(
                mtl, UsdShade.Tokens.strongerThanDescendants
            )
            UsdGeom.Imageable(v).MakeInvisible()
            variants.append(v)

        UsdGeom.Imageable(slot).MakeInvisible()
        slots.append(
            {
                "xform": slot,
                "imageable": UsdGeom.Imageable(slot.GetPrim()),
                "t": t_op,
                "rz": rz_op,
                "ry": ry_op,
                "s": sc_op,
                "variants": variants,
                "active_variant": None,
            }
        )

    return slots


def hide_clutter(slot):
    slot["imageable"].MakeInvisible()
    slot["t"].Set(Gf.Vec3d(0.0, 0.0, -50.0))
    if slot["active_variant"] is not None:
        UsdGeom.Imageable(slot["active_variant"]).MakeInvisible()
        slot["active_variant"] = None


def place_clutter(slot, x, y, yaw, tilt, scale_mult, variant_idx):
    if slot["active_variant"] is not None:
        UsdGeom.Imageable(slot["active_variant"]).MakeInvisible()

    v = slot["variants"][variant_idx]
    UsdGeom.Imageable(v).MakeVisible()
    slot["active_variant"] = v
    slot["t"].Set(Gf.Vec3d(x, y, 0.0))
    slot["rz"].Set(float(yaw))
    slot["ry"].Set(float(tilt))
    slot["s"].Set(Gf.Vec3f(scale_mult, scale_mult, scale_mult))
    slot["imageable"].MakeVisible()


clutter_pool = {}
if CLUTTER_ENABLED:
    clutter_pool["straw"] = build_clutter_slots("straw", MAX_STRAW_SLOTS)
    clutter_pool["stone"] = build_clutter_slots("stone", MAX_STONE_SLOTS)

    print(
        f"[clutter] straw={len(clutter_pool['straw'])}, "
        f"stone={len(clutter_pool['stone'])}"
    )
    print(
        f"[clutter] straw: {STRAW_LENGTH[0] * 1000:.0f}-"
        f"{STRAW_LENGTH[1] * 1000:.0f} mm, "
        f"~{STRAW_LENGTH[0] / gsd_max * FINAL_W / IMAGE_W:.0f}-"
        f"{STRAW_LENGTH[1] / gsd_min * FINAL_W / IMAGE_W:.0f}px"
    )
    print(
        f"[clutter] stone: {STONE_SIZE[0] * 1000:.0f}-"
        f"{STONE_SIZE[1] * 1000:.0f} mm, "
        f"~{STONE_SIZE[0] / gsd_max * FINAL_W / IMAGE_W:.0f}-"
        f"{STONE_SIZE[1] / gsd_min * FINAL_W / IMAGE_W:.0f}px"
    )

    def _has_semantics(prim):
        for schema in prim.GetAppliedSchemas():
            if "Semantic" in schema or "Labels" in schema:
                return True
        return False

    tagged = []
    root = stage.GetPrimAtPath("/World/Clutter")
    if root and root.IsValid():
        for prim in Usd.PrimRange(root):
            if _has_semantics(prim):
                tagged.append(str(prim.GetPath()))

    if tagged:
        print(f"[FATAL] {len(tagged)} clutter prim(s) carry semantics")
        for item in tagged[:5]:
            print(f"  {item}")
        raise SystemExit("clutter must remain unlabelled")

    print("[clutter] verified: no semantics")


def hide_slot(slot):
    slot["imageable"].MakeInvisible()
    slot["t"].Set(Gf.Vec3d(0.0, 0.0, -50.0))
    if slot["active_variant"] is not None:
        UsdGeom.Imageable(slot["active_variant"]).MakeInvisible()
        slot["active_variant"] = None


def place_slot(slot, x, y, yaw_deg, scale_mult, variant_idx):
    if slot["active_variant"] is not None:
        UsdGeom.Imageable(slot["active_variant"]).MakeInvisible()

    v = slot["variants"][variant_idx]
    UsdGeom.Imageable(v).MakeVisible()
    slot["active_variant"] = v

    slot["t"].Set(Gf.Vec3d(x, y, 0.0))
    slot["r"].Set(float(yaw_deg))
    slot["s"].Set(Gf.Vec3f(scale_mult, scale_mult, scale_mult))
    slot["imageable"].MakeVisible()


def pick_frame_stage(rng):
    names = [
        name
        for name in CROP_STAGES
        if any(
            t["stage"] == name for t in templates.get("crop", [])
        )
    ]
    if not names:
        return None

    weights = [CROP_STAGES[name]["weight"] for name in names]
    total = sum(weights)
    value = rng.uniform(0.0, total)
    acc = 0.0
    for name, weight in zip(names, weights):
        acc += weight
        if value <= acc:
            return name

    return names[-1]


def sample_stage_footprint(rng, stage_name, native_fp):
    lo, hi = CROP_STAGES[stage_name]["fp"]
    allowed_lo = native_fp / CROP_MAX_STRETCH
    allowed_hi = native_fp * CROP_MAX_STRETCH
    inter_lo = max(lo, allowed_lo)
    inter_hi = min(hi, allowed_hi)

    if inter_lo <= inter_hi:
        fp = rng.uniform(inter_lo, inter_hi)
    else:
        fp = allowed_hi if allowed_hi < lo else allowed_lo

    return min(
        max(fp, CROP_FOOTPRINT_CLAMP[0]),
        CROP_FOOTPRINT_CLAMP[1],
    )

# Layout
def to_world(points, yaw_deg, ox, oy):
    a = math.radians(yaw_deg)
    ca, sa = math.cos(a), math.sin(a)
    return [
        (ox + u * ca - v * sa, oy + u * sa + v * ca)
        for (u, v) in points
    ]


def crop_row_layout(rng, half_extent):
    half_w, half_h = half_extent
    reach_x = half_w + LAYOUT_MARGIN
    reach_y = half_h + LAYOUT_MARGIN

    theta = rng.uniform(0.0, math.pi)
    dx, dy = math.cos(theta), math.sin(theta)
    px, py = -dy, dx

    support = half_w * abs(px) + half_h * abs(py)
    base_lim = min(CROP_ROW_SPACING / 2.0, support)
    base = rng.uniform(-base_lim, base_lim)
    phase = rng.uniform(0.0, CROP_IN_ROW_SPACING)

    diag = math.hypot(reach_x, reach_y)
    k_max = int(diag / CROP_ROW_SPACING) + 1
    i_max = int(diag / CROP_IN_ROW_SPACING) + 1

    out = []
    for k in range(-k_max, k_max + 1):
        off = base + k * CROP_ROW_SPACING
        if abs(off) > diag:
            continue

        for i in range(-i_max, i_max + 1):
            s = phase + i * CROP_IN_ROW_SPACING
            if rng.random() < CROP_MISSING_PROB:
                continue

            js = rng.uniform(-CROP_ALONG_JITTER, CROP_ALONG_JITTER)
            jo = rng.uniform(-CROP_ROW_JITTER, CROP_ROW_JITTER)
            x = dx * (s + js) + px * (off + jo)
            y = dy * (s + js) + py * (off + jo)

            if abs(x) > reach_x or abs(y) > reach_y:
                continue
            out.append((x, y))

    inside = [
        p for p in out
        if abs(p[0]) <= half_w and abs(p[1]) <= half_h
    ]
    edge = [p for p in out if p not in inside]
    rng.shuffle(inside)
    rng.shuffle(edge)

    keep = inside[:MAX_CROP_SLOTS]
    room = MAX_CROP_SLOTS - len(keep)
    edge_allow = min(EDGE_CLIP_CROPS, room, max(1, len(keep)))
    keep += edge[:edge_allow]
    rng.shuffle(keep)
    return keep, math.degrees(theta)


def clutter_layout(rng, half_extent, n, avoid, avoid_r):
    half_w, half_h = half_extent
    reach_x = half_w + LAYOUT_MARGIN
    reach_y = half_h + LAYOUT_MARGIN
    out = []

    for _ in range(n):
        x = y = 0.0
        for _try in range(8):
            x = rng.uniform(-reach_x, reach_x)
            y = rng.uniform(-reach_y, reach_y)
            if all(
                (x - ax) ** 2 + (y - ay) ** 2 > avoid_r ** 2
                for ax, ay in avoid
            ):
                break
        out.append((x, y))

    return out


def weed_layout(rng, half_extent):
    n = rng.randint(WEEDS_PER_FRAME[0], WEEDS_PER_FRAME[1])
    n = min(n, MAX_WEED_SLOTS)
    half_w, half_h = half_extent
    pts = []

    for _ in range(n):
        x = rng.uniform(-half_w, half_w)
        y = rng.uniform(-half_h, half_h)

        if rng.random() < EDGE_CLIP_WEED_PROB:
            if rng.random() < 0.5:
                edge = half_w + rng.uniform(0.0, LAYOUT_MARGIN)
                x = edge if rng.random() < 0.5 else -edge
            else:
                edge = half_h + rng.uniform(0.0, LAYOUT_MARGIN)
                y = edge if rng.random() < 0.5 else -edge

        pts.append((x, y))

    return pts


# Lighting
def apply_lighting(rng):
    harsh = rng.random() < HARSH_PROB
    cfg = LIGHT_HARSH if harsh else LIGHT_OVERCAST

    zenith = rng.uniform(*cfg["sun_zenith"])
    azim = rng.uniform(0.0, 360.0)
    sun_rotate.Set(Gf.Vec3f(float(zenith), 0.0, float(azim)))
    sun.GetIntensityAttr().Set(rng.uniform(*cfg["sun_intensity"]))
    sun.GetAngleAttr().Set(rng.uniform(*cfg["sun_angle"]))
    sun.GetColorAttr().Set(Gf.Vec3f(*cfg["sun_color"]))

    fill.GetIntensityAttr().Set(rng.uniform(*cfg["fill_intensity"]))
    fill_rotate.Set(
        Gf.Vec3f(-20.0, 0.0, float((azim + 180.0) % 360.0))
    )

    dome.GetIntensityAttr().Set(rng.uniform(*cfg["dome_intensity"]))
    dome.GetColorAttr().Set(Gf.Vec3f(*cfg["dome_color"]))

    return "harsh" if harsh else "overcast"


def apply_soil(rng, entry=None):
    if entry is None:
        entry = rng.choice(soil_materials)

    rebound = bind_soil_material(entry)
    shader = entry["shader"]

    if entry["has_normal"]:
        shader.GetInput("bump_factor").Set(rng.uniform(1.10, 1.80))

    if not entry["has_roughness"]:
        shader.GetInput("reflection_roughness_constant").Set(
            rng.uniform(0.72, 0.98)
        )

    tint = np.array(rng.choice(SOIL_TINTS), dtype=np.float32)
    correction = entry.get("tint_correction")
    if correction is not None:
        tint = tint * correction

    shader.GetInput("diffuse_tint").Set(
        Gf.Vec3f(*[float(v) for v in tint])
    )

    ts = MUD_TILES * rng.uniform(0.85, 1.10)
    shader.GetInput("texture_scale").Set(Gf.Vec2f(ts, ts))
    shader.GetInput("texture_translate").Set(
        Gf.Vec2f(
            UV_CENTER + rng.uniform(-UV_JITTER, UV_JITTER),
            UV_CENTER + rng.uniform(-UV_JITTER, UV_JITTER),
        )
    )

    return entry, ts, rebound


def apply_camera(rng, tiling):
    h = rng.uniform(*CAM_HEIGHT)
    period = 1.0 / max(tiling, 1e-6)

    width = (HORIZ_APERTURE / FOCAL_LENGTH) * h
    height = width * (IMAGE_H / float(IMAGE_W))
    span = math.hypot(width, height)
    margin = (period - span) / 2.0
    jit = max(0.0, margin * 0.85)

    safe = GROUND_HALF - span / 2.0 - jit - 0.05
    if safe <= 0.0:
        raise SystemExit(
            f"Ground plane too small: half-extent={GROUND_HALF:.2f} m, "
            f"required span={span:.2f} m"
        )

    k_lim = int((safe / period) - 0.5)
    k_lim = max(0, min(CAM_TILE_RANGE, k_lim))

    kx = rng.randint(-k_lim, k_lim)
    ky = rng.randint(-k_lim, k_lim)
    cx = (kx + 0.5) * period + rng.uniform(-jit, jit)
    cy = (ky + 0.5) * period + rng.uniform(-jit, jit)

    yaw = rng.uniform(0.0, 360.0)

    cam_translate.Set(Gf.Vec3d(cx, cy, h))
    cam_rotate.Set(
        Gf.Vec3f(
            rng.uniform(-CAM_TILT, CAM_TILT),
            rng.uniform(-CAM_TILT, CAM_TILT),
            float(yaw),
        )
    )

    half_w = (HORIZ_APERTURE / FOCAL_LENGTH) * h / 2.0
    half_h = half_w * (IMAGE_H / float(IMAGE_W))
    return h, (half_w, half_h), margin, (cx, cy), yaw


# Annotators
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
rgb_annot.attach([rp])
bbox_annot.attach([rp])


def soil_chroma(rgb):
    a = rgb[:, :, :3].astype(np.float32)
    m = np.array(
        [a[:, :, 0].mean(), a[:, :, 1].mean(), a[:, :, 2].mean()]
    )
    return m / max(m.sum(), 1e-6)


def tint_toward_reference(measured_chroma):
    ref = np.array(SOIL_REF_RGB, dtype=np.float32)
    ref = ref / ref.sum()
    step = ref / np.maximum(measured_chroma, 1e-6)
    step = np.power(step / step.max(), SOIL_TINT_GAIN)

    lum = float(
        0.2126 * step[0] + 0.7152 * step[1] + 0.0722 * step[2]
    )
    step = step / max(lum, 1e-6)
    return np.clip(step, SOIL_TINT_CLAMP, SOIL_TINT_CEIL)


def chroma_error(measured_chroma):
    ref = np.array(SOIL_REF_RGB, dtype=np.float32)
    ref = ref / ref.sum()
    return float(np.abs(measured_chroma - ref).max())


def soil_detail(rgb):
    a = rgb[:, :, :3].astype(np.float32)
    lum = (
        0.2126 * a[:, :, 0]
        + 0.7152 * a[:, :, 1]
        + 0.0722 * a[:, :, 2]
    )

    total = a[:, :, 0] + a[:, :, 1] + a[:, :, 2] + 1e-6
    exg = (2.0 * a[:, :, 1] - a[:, :, 0] - a[:, :, 2]) / total
    med = float(np.median(lum))

    keep = (exg <= 0.08) & (lum > 0.55 * med)
    gx = np.abs(np.diff(lum, axis=1))[:-1, :]
    gy = np.abs(np.diff(lum, axis=0))[:, :-1]
    grad = gx + gy
    mask = keep[:-1, :-1]
    lum_t = lum[:-1, :-1]

    if mask.sum() < 5000:
        return 0.0

    return float(
        grad[mask].mean() / max(lum_t[mask].mean(), 1e-6) * 100.0
    )


RENDER_MODE = "PathTracing"
DLSS_EXEC_MODE = 2

PATHTRACING_TOTAL_SPP = 128
PATHTRACING_SPP = 8
PROBE_TOTAL_SPP = 16

RT_SUBFRAMES = (
    max(1, PATHTRACING_TOTAL_SPP // PATHTRACING_SPP)
    if RENDER_MODE == "PathTracing"
    else 8
)
PROBE_SUBFRAMES = (
    max(1, PROBE_TOTAL_SPP // PATHTRACING_SPP)
    if RENDER_MODE == "PathTracing"
    else 2
)


def configure_renderer():
    try:
        import carb.settings

        settings = carb.settings.get_settings()
        settings.set("/rtx/rendermode", RENDER_MODE)

        if RENDER_MODE == "PathTracing":
            settings.set("/rtx/pathtracing/spp", PATHTRACING_SPP)
            settings.set(
                "/rtx/pathtracing/totalSpp", PATHTRACING_TOTAL_SPP
            )
            settings.set(
                "/rtx/pathtracing/fractionalCutoutOpacity", True
            )
            print(
                f"[render] PathTracing, totalSpp={PATHTRACING_TOTAL_SPP}, "
                f"spp={PATHTRACING_SPP}, subframes={RT_SUBFRAMES}"
            )
        else:
            settings.set("/rtx/post/dlss/execMode", DLSS_EXEC_MODE)
            print(
                f"[render] RaytracedLighting, "
                f"DLSS execMode={DLSS_EXEC_MODE}"
            )

    except Exception as exc:
        print(f"[WARN] Renderer configuration failed: {exc}")


configure_renderer()

_T_WARMUP = time.time()
SOIL_QC_ENABLED = bool(soil_sets)

if not SOIL_QC_ENABLED:
    print("[WARN] no soil textures found - ground will be flat brown")

sun_rotate.Set(Gf.Vec3f(30.0, 0.0, 45.0))
sun.GetIntensityAttr().Set(5500.0)
sun.GetAngleAttr().Set(0.6)
dome.GetIntensityAttr().Set(1000.0)
fill.GetIntensityAttr().Set(450.0)
cam_translate.Set(Gf.Vec3d(0.5, 0.5, CAM_HEIGHT[1]))
cam_rotate.Set(Gf.Vec3f(0.0, 0.0, 0.0))

usable_materials = []
for entry in soil_materials:
    bind_soil_material(entry)
    best = 0.0
    steps = 0

    for attempt in range(WARMUP_MAX_STEPS):
        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
        steps += 1
        detail = soil_detail(rgb_annot.get_data())
        best = max(best, detail)
        if detail >= SOIL_DETAIL_FLOOR and attempt >= 1:
            break

    entry["baseline_detail"] = best
    ok = best >= SOIL_DETAIL_FLOOR

    if ok and SOIL_AUTO_TINT:
        mean_palette = np.mean(
            np.array(SOIL_TINTS, dtype=np.float32), axis=0
        )
        corr = np.ones(3, dtype=np.float32)
        sat0 = None
        ch = None
        tint_iter = 0

        for tint_iter in range(SOIL_TINT_ITERS):
            tint = np.clip(
                corr * mean_palette, SOIL_TINT_CLAMP, SOIL_TINT_CEIL
            )
            entry["shader"].GetInput("diffuse_tint").Set(
                Gf.Vec3f(*[float(v) for v in tint])
            )
            rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
            ch = soil_chroma(rgb_annot.get_data())

            if sat0 is None:
                sat0 = (
                    (ch.max() - ch.min()) / max(ch.max(), 1e-6) * 100.0
                )

            if chroma_error(ch) <= SOIL_TINT_TOL:
                break

            corr = np.clip(
                corr * tint_toward_reference(ch),
                SOIL_TINT_CLAMP,
                SOIL_TINT_CEIL,
            )

        if ch is None:
            ch = soil_chroma(rgb_annot.get_data())
        sat1 = (ch.max() - ch.min()) / max(ch.max(), 1e-6) * 100.0
        entry["tint_correction"] = corr
        print(
            f"  {entry['name']}: detail={best:.2f}, steps={steps}, "
            f"sat={sat0:.1f}%->{sat1:.1f}%, iter={tint_iter + 1}"
        )
    else:
        print(
            f"  {entry['name']}: detail={best:.2f}, steps={steps}, "
            f"{'OK' if ok else 'dropped'}"
        )

    if ok:
        usable_materials.append(entry)

if not usable_materials and not SOIL_QC_ENABLED:
    usable_materials = list(soil_materials)

if not usable_materials:
    print("[ERROR] No soil texture became resident during warm-up")
    simulation_app.close()
    raise SystemExit(1)

if len(usable_materials) < len(soil_materials):
    dropped = len(soil_materials) - len(usable_materials)
    print(f"[WARN] Dropped {dropped} soil material(s)")

soil_materials = usable_materials
print(
    f"[warmup] {hms(time.time() - _T_WARMUP)}, "
    f"materials={len(soil_materials)}"
)
print(f"[startup] {hms(elapsed())}")

_T_FRAMES = time.time()

# Main loop
class_counts = {c: 0 for c in CLASS_IDS}
regime_counts = {"harsh": 0, "overcast": 0}
lum_log = []
retry_log = []
attempts_used = {}


def pick_other_soil(current):
    others = [e for e in soil_materials if e is not current]
    return rng.choice(others) if others else current


def settle_scene(n=SETTLE_UPDATES):
    for _ in range(max(0, n)):
        try:
            simulation_app.update()
        except Exception:
            return


def count_plant_boxes():
    try:
        bb = bbox_annot.get_data()
    except Exception:
        return 0

    labels = bb["info"]["idToLabels"]
    n = 0

    for box in bb["data"]:
        sem_id = int(box["semanticId"])
        label = labels.get(sem_id, labels.get(str(sem_id), {}))
        if isinstance(label, dict):
            label = label.get("class", "")

        if not any(c in str(label) for c in CLASS_IDS):
            continue

        w = min(float(box["x_max"]), float(IMAGE_W)) - max(
            0.0, float(box["x_min"])
        )
        h = min(float(box["y_max"]), float(IMAGE_H)) - max(
            0.0, float(box["y_min"])
        )
        if w >= MIN_BOX_PX and h >= MIN_BOX_PX:
            n += 1

    return n


def count_visible(extents, half_w, half_h):
    min_overlap = (
        MIN_BOX_PX
        * (HORIZ_APERTURE / FOCAL_LENGTH)
        * CAM_HEIGHT[1]
        / IMAGE_W
    )
    n = 0

    for lx, ly, fp in extents:
        r = max(fp, 1e-6) / 2.0
        vis_w = min(half_w, lx + r) - max(-half_w, lx - r)
        vis_h = min(half_h, ly + r) - max(-half_h, ly - r)
        if vis_w >= min_overlap and vis_h >= min_overlap:
            n += 1

    return n

yield_window = []
yield_overshoot = []
consecutive_failures = 0
failed_frames = []

for frame_idx in range(NUM_IMAGES):
    t_frame = time.time()
    done = frame_idx

    if done >= 3:
        rate = (time.time() - _T_FRAMES) / done
        eta = rate * (NUM_IMAGES - done)
        print(
            f"Frame {frame_idx + 1}/{NUM_IMAGES} "
            f"({rate:.1f}s/frame, ~{hms(eta)} remaining)"
        )
    else:
        print(f"Frame {frame_idx + 1}/{NUM_IMAGES}")

    forced_soil = None
    committed = False
    attempt = 0

    while attempt < SOIL_MAX_ATTEMPTS and not committed:
        attempt += 1

        frame_stage = pick_frame_stage(rng)
        soil_entry, tiling, rebound = apply_soil(rng, forced_soil)
        regime = apply_lighting(rng)
        (
            cam_h,
            half_extent,
            seam_margin,
            (cam_x, cam_y),
            cam_yaw,
        ) = apply_camera(rng, tiling)
        half_w, half_h = half_extent

        ox, oy = cam_x, cam_y

        # Crops: rows
        n_crop = 0
        n_crop_in = 0
        row_deg = None
        crop_positions_used = []
        crop_extents = []

        if "crop" in slot_pool:
            positions, row_deg = crop_row_layout(rng, half_extent)
            world_pos = to_world(positions, cam_yaw, ox, oy)

            for slot, (wx, wy), (lx, ly) in zip(
                slot_pool["crop"], world_pos, positions
            ):
                candidates = [
                    i
                    for i, stage_name in enumerate(slot["variant_stage"])
                    if stage_name == frame_stage
                ]
                if not candidates:
                    candidates = list(range(len(slot["variants"])))

                variant_idx = rng.choice(candidates)
                native = max(slot["variant_fp"][variant_idx], 1e-6)
                target = sample_stage_footprint(
                    rng, frame_stage, native
                )

                place_slot(
                    slot,
                    wx,
                    wy,
                    yaw_deg=rng.uniform(0.0, 360.0),
                    scale_mult=target / native,
                    variant_idx=variant_idx,
                )

                crop_positions_used.append((lx, ly))
                crop_extents.append((lx, ly, target))

            n_crop = min(len(positions), len(slot_pool["crop"]))
            n_crop_in = count_visible(
                crop_extents[:n_crop], half_w, half_h
            )

            for slot in slot_pool["crop"][n_crop:]:
                hide_slot(slot)

        # Weeds: scattered
        n_weed = 0
        n_weed_in = 0

        if "weed" in slot_pool:
            positions = weed_layout(rng, half_extent)
            world_pos = to_world(positions, cam_yaw, ox, oy)
            base_fp = CLASS_TARGET_FOOTPRINT["weed"]
            weed_extents = []

            for slot, (wx, wy), (lx, ly) in zip(
                slot_pool["weed"], world_pos, positions
            ):
                fp = rng.uniform(*WEED_FOOTPRINT_RANGE)
                place_slot(
                    slot,
                    wx,
                    wy,
                    yaw_deg=rng.uniform(0.0, 360.0),
                    scale_mult=fp / base_fp,
                    variant_idx=rng.randrange(len(slot["variants"])),
                )
                weed_extents.append((lx, ly, fp))

            n_weed = min(len(positions), len(slot_pool["weed"]))
            n_weed_in = count_visible(
                weed_extents[:n_weed], half_w, half_h
            )

            for slot in slot_pool["weed"][n_weed:]:
                hide_slot(slot)

        # Clutter: unlabelled straw and stones
        n_straw = 0
        n_stone = 0

        if CLUTTER_ENABLED:
            avoid = list(crop_positions_used)

            n_straw = rng.randint(*STRAW_PER_FRAME)
            n_straw = min(n_straw, len(clutter_pool["straw"]))
            points = clutter_layout(
                rng, half_extent, n_straw, avoid, 0.06
            )
            world_points = to_world(points, cam_yaw, ox, oy)
            n_variants = (
                len(clutter_pool["straw"][0]["variants"])
                if n_straw
                else 1
            )

            for slot, (wx, wy) in zip(
                clutter_pool["straw"], world_points
            ):
                place_clutter(
                    slot,
                    wx,
                    wy,
                    yaw=rng.uniform(0.0, 360.0),
                    tilt=rng.uniform(-8.0, 8.0),
                    scale_mult=rng.uniform(0.75, 1.35),
                    variant_idx=rng.randrange(n_variants),
                )

            for slot in clutter_pool["straw"][n_straw:]:
                hide_clutter(slot)

            n_stone = rng.randint(*STONES_PER_FRAME)
            n_stone = min(n_stone, len(clutter_pool["stone"]))
            points = clutter_layout(
                rng, half_extent, n_stone, avoid, 0.05
            )
            world_points = to_world(points, cam_yaw, ox, oy)
            n_variants = (
                len(clutter_pool["stone"][0]["variants"])
                if n_stone
                else 1
            )

            for slot, (wx, wy) in zip(
                clutter_pool["stone"], world_points
            ):
                place_clutter(
                    slot,
                    wx,
                    wy,
                    yaw=rng.uniform(0.0, 360.0),
                    tilt=0.0,
                    scale_mult=rng.uniform(0.7, 1.4),
                    variant_idx=rng.randrange(n_variants),
                )

            for slot in clutter_pool["stone"][n_stone:]:
                hide_clutter(slot)

        soil_name = soil_entry["name"]
        row_txt = f"{row_deg:.0f}deg" if row_deg is not None else "n/a"
        attempt_txt = "" if attempt == 1 else f", attempt={attempt}"

        print(
            f"[frame] light={regime}, soil={soil_name}, stage={frame_stage}, "
            f"cam_h={cam_h:.3f} m, "
            f"size={half_w * 2:.3f}x{half_h * 2:.3f} m, "
            f"row={row_txt}{attempt_txt}"
        )
        print(
            f"[objects] crop={n_crop}, weed={n_weed}, "
            f"straw={n_straw}, stone={n_stone}"
        )
        print(f"[visible] crop={n_crop_in}, weed={n_weed_in}")

        if seam_margin <= 0:
            print(
                f"[WARN] Soil seam may be visible: "
                f"margin={seam_margin * 1000:.0f} mm"
            )

        settle_scene()

        rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
        rgb = rgb_annot.get_data()

        # Texture residency gate
        baseline = soil_entry["baseline_detail"] or SOIL_DETAIL_FLOOR
        threshold = max(
            SOIL_DETAIL_FLOOR, baseline * SOIL_DETAIL_FRACTION
        )
        detail = soil_detail(rgb)

        resteps = 0
        while (
            SOIL_QC_ENABLED
            and detail < threshold
            and resteps < SOIL_RESTEP_MAX
        ):
            resteps += 1
            rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
            rgb = rgb_annot.get_data()
            detail = soil_detail(rgb)

        if resteps and detail >= threshold:
            print(
                f"[settle] ground resolved after {resteps} extra step(s)"
            )

        if SOIL_QC_ENABLED and detail < threshold:
            print("[WARN] ground still reads as untextured")
            retry_log.append(
                {
                    "frame": frame_idx,
                    "attempt": attempt,
                    "soil": soil_name,
                    "detail": round(detail, 3),
                    "threshold": round(threshold, 3),
                    "resteps": resteps,
                }
            )

            if attempt < SOIL_RETRY_SAME:
                forced_soil = soil_entry
                print("[retry] same soil")
            else:
                forced_soil = pick_other_soil(soil_entry)
                if forced_soil is soil_entry:
                    print("[retry] reusing the only available soil")
                else:
                    print(
                        f"[retry] switching soil -> "
                        f"{forced_soil['name']}"
                    )
            continue

        # Plant residency settle
        expected_in = n_crop_in + n_weed_in
        plant_resteps = 0

        if expected_in >= PLANT_VERIFY_MIN_PLACED:
            need = max(
                1,
                int(math.ceil(expected_in * PLANT_YIELD_MIN)),
            )
            have = count_plant_boxes()

            while have < need and plant_resteps < PLANT_RESTEP_MAX:
                plant_resteps += 1
                settle_scene(SETTLE_UPDATES)
                rep.orchestrator.step(rt_subframes=PROBE_SUBFRAMES)
                have = count_plant_boxes()

            if plant_resteps:
                rep.orchestrator.step(rt_subframes=RT_SUBFRAMES)
                rgb = rgb_annot.get_data()
                have = count_plant_boxes()
                print(
                    f"[settle] plants={have}/{expected_in}, "
                    f"extra_steps={plant_resteps}"
                )

        a = rgb[:, :, :3].astype(np.float32)
        lum = (
            0.2126 * a[:, :, 0]
            + 0.7152 * a[:, :, 1]
            + 0.0722 * a[:, :, 2]
        )
        med = float(np.median(lum))
        deep_frac = float((lum < 0.30 * med).mean() * 100.0)
        shadow_frac = float((lum < 0.55 * med).mean() * 100.0)
        clipped_black = float((lum < 1.0).mean() * 100.0)
        stats = (
            float(lum.mean()),
            float(lum.std()),
            float(np.percentile(lum, 1)),
        )

        red = a[:, :, 0]
        green = a[:, :, 1]
        blue = a[:, :, 2]
        total = red + green + blue + 1e-6
        exg = (2.0 * green - red - blue) / total
        green_mask = exg > 0.08
        green_frac = float(green_mask.mean() * 100.0)
        soil_mask = ~green_mask

        if soil_mask.sum() > 1000:
            sr = float(red[soil_mask].mean())
            sg = float(green[soil_mask].mean())
            sb = float(blue[soil_mask].mean())
            smax, smin = max(sr, sg, sb), min(sr, sg, sb)
            soil_sat = float(
                (smax - smin) / max(smax, 1e-6) * 100.0
            )
        else:
            sr = sg = sb = soil_sat = float("nan")

        gpu = gpu_mem()
        gpu_txt = f", gpu={gpu[0]}/{gpu[1]} MB" if gpu else ""
        print(
            f"[metrics] detail={detail:.2f}, "
            f"lum={stats[0]:.1f}/{stats[1]:.1f}/{stats[2]:.1f}, "
            f"shadow={shadow_frac:.1f}%, deep={deep_frac:.1f}%, "
            f"black={clipped_black:.2f}%{gpu_txt}"
        )
        print(
            f"[color] green={green_frac:.2f}%, "
            f"soil={sr:.0f},{sg:.0f},{sb:.0f}, sat={soil_sat:.1f}%"
        )

        if frame_idx == 0 and (
            stats[1] < 3.0 or stats[0] > 250.0 or stats[0] < 5.0
        ):
            print(
                f"[ERROR] Invalid first frame: "
                f"mean={stats[0]:.1f}, std={stats[1]:.1f}"
            )

            if stats[0] > 250.0:
                print("[ERROR] Frame is overexposed")
            elif stats[0] < 5.0:
                print("[ERROR] Frame is underexposed")
            else:
                print("[ERROR] Frame has very low contrast")

            debug_path = f"{OUTPUT_DIR}/debug_frame_000.png"
            Image.fromarray(rgb[:, :, :3]).resize(
                (FINAL_W, FINAL_H), Image.LANCZOS
            ).save(debug_path)
            print(f"[debug] saved {debug_path}")
            simulation_app.close()
            raise SystemExit(1)

        split = frame_split[frame_idx]
        file_name = f"frame_{frame_idx:05d}.png"

        bb = bbox_annot.get_data()
        boxes = bb["data"]
        id_to_labels = bb["info"]["idToLabels"]

        sx = FINAL_W / float(IMAGE_W)
        sy = FINAL_H / float(IMAGE_H)

        frame_counts = {c: 0 for c in CLASS_IDS}
        yolo_lines = []
        overlay_boxes = []
        frame_annotations = []

        for box in boxes:
            sem_id = int(box["semanticId"])
            label = id_to_labels.get(
                sem_id, id_to_labels.get(str(sem_id), {})
            )
            if isinstance(label, dict):
                label = label.get("class", "")
            label = str(label)

            cls_name = next(
                (c for c in CLASS_IDS if c in label), None
            )
            if cls_name is None:
                continue

            x1 = float(box["x_min"])
            y1 = float(box["y_min"])
            x2 = float(box["x_max"])
            y2 = float(box["y_max"])

            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2 = min(float(IMAGE_W), x2)
            y2 = min(float(IMAGE_H), y2)
            w, h = x2 - x1, y2 - y1

            if w < MIN_BOX_PX or h < MIN_BOX_PX:
                continue

            # COCO coordinates in the saved image
            fx1, fy1 = x1 * sx, y1 * sy
            fw, fh = w * sx, h * sy
            fx1 = min(max(0.0, fx1), FINAL_W)
            fy1 = min(max(0.0, fy1), FINAL_H)
            fw = min(fw, FINAL_W - fx1)
            fh = min(fh, FINAL_H - fy1)

            if fw <= 0.0 or fh <= 0.0:
                continue

            if WRITE_COCO:
                bx = round(fx1, 2)
                by = round(fy1, 2)
                bw = round(fw, 2)
                bh = round(fh, 2)
                frame_annotations.append(
                    {
                        "image_id": frame_idx,
                        "category_id": coco_category_id[cls_name],
                        "bbox": [bx, by, bw, bh],
                        "area": round(bw * bh, 2),
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )

            if WRITE_YOLO:
                yolo_lines.append(
                    f"{CLASS_IDS[cls_name]} "
                    f"{(x1 + x2) / 2.0 / IMAGE_W:.6f} "
                    f"{(y1 + y2) / 2.0 / IMAGE_H:.6f} "
                    f"{w / IMAGE_W:.6f} "
                    f"{h / IMAGE_H:.6f}"
                )

            overlay_boxes.append((fx1, fy1, fw, fh, cls_name))
            frame_counts[cls_name] += 1

        # Plant render gate
        placed_total = n_crop + n_weed
        n_boxes = len(overlay_boxes)

        if expected_in >= PLANT_VERIFY_MIN_PLACED:
            plants_missing = n_boxes < max(
                1, int(math.ceil(expected_in * PLANT_YIELD_MIN))
            )
        else:
            plants_missing = (
                placed_total >= 3
                and n_boxes == 0
                and green_frac < PLANT_MIN_GREEN
            )

        if expected_in:
            ratio = n_boxes / float(expected_in)
            if ratio > 1.0:
                yield_overshoot.append(frame_idx)
            yield_window.append(min(1.0, ratio))
            del yield_window[:-20]

        if plants_missing:
            print(
                f"[WARN] Plants not visible: "
                f"crop={n_crop_in}/{n_crop}, "
                f"weed={n_weed_in}/{n_weed}, boxes={n_boxes}, "
                f"green={green_frac:.2f}%, resteps={plant_resteps}"
            )

            if SAVE_DISCARDED_FRAMES:
                try:
                    os.makedirs(
                        f"{OUTPUT_DIR}/debug_discarded", exist_ok=True
                    )
                    Image.fromarray(rgb[:, :, :3]).resize(
                        (FINAL_W, FINAL_H), Image.LANCZOS
                    ).save(
                        f"{OUTPUT_DIR}/debug_discarded/"
                        f"f{frame_idx:05d}_a{attempt}_{soil_name}_"
                        f"{frame_stage}_c{n_crop}w{n_weed}_"
                        f"b{n_boxes}_g{green_frac:.2f}.png"
                    )
                except Exception as exc:
                    print(
                        f"[WARN] could not save discarded frame: {exc}"
                    )

            retry_log.append(
                {
                    "frame": frame_idx,
                    "attempt": attempt,
                    "soil": soil_name,
                    "reason": "plants_not_rendered",
                    "placed": placed_total,
                    "in_frame": expected_in,
                    "boxes": n_boxes,
                    "green": round(green_frac, 3),
                    "plant_resteps": plant_resteps,
                }
            )
            forced_soil = soil_entry
            continue

        # Writes
        img = Image.fromarray(rgb[:, :, :3])
        img = img.resize((FINAL_W, FINAL_H), Image.LANCZOS)
        img.save(f"{OUTPUT_DIR}/images/{split}/{file_name}")

        if WRITE_COCO:
            for ann in frame_annotations:
                ann["id"] = coco_ann_id[split]
                coco_ann_id[split] += 1
                coco_annotations[split].append(ann)

            coco_images[split].append(
                {
                    "id": frame_idx,
                    "file_name": file_name,
                    "width": FINAL_W,
                    "height": FINAL_H,
                }
            )

        if WRITE_YOLO:
            with open(
                f"{OUTPUT_DIR}/labels/{split}/frame_{frame_idx:05d}.txt",
                "w",
            ) as f:
                f.write(
                    "\n".join(yolo_lines)
                    + ("\n" if yolo_lines else "")
                )

        if frame_idx < DEBUG_OVERLAY_COUNT and overlay_boxes:
            from PIL import ImageDraw

            overlay = img.copy().convert("RGB")
            draw = ImageDraw.Draw(overlay)
            for fx1, fy1, fw, fh, class_name in overlay_boxes:
                colour = (
                    (0, 255, 0)
                    if class_name == "crop"
                    else (255, 64, 64)
                )
                draw.rectangle(
                    [fx1, fy1, fx1 + fw, fy1 + fh],
                    outline=colour,
                    width=2,
                )
                draw.text(
                    (fx1 + 2, max(0, fy1 - 10)),
                    class_name,
                    fill=colour,
                )
            overlay.save(
                f"{OUTPUT_DIR}/debug_overlays/"
                f"frame_{frame_idx:05d}.png"
            )

        if (
            frame_idx == 0
            and (n_crop + n_weed) > 0
            and not overlay_boxes
        ):
            print(
                "Plants were placed but the annotator returned no usable boxes."
            )
            print(
                f"  placed crop={n_crop} weed={n_weed}, boxes=0"
            )
            raise SystemExit(1)

        for cls_name in CLASS_IDS:
            class_counts[cls_name] += frame_counts[cls_name]

        regime_counts[regime] += 1
        lum_log.append(
            {
                "frame": frame_idx,
                "regime": regime,
                "stage": frame_stage,
                "mean": stats[0],
                "std": stats[1],
                "p1": stats[2],
                "median": med,
                "deep_shadow_pct": deep_frac,
                "shadow_pct": shadow_frac,
                "clipped_black_pct": clipped_black,
                "green_pct": green_frac,
                "soil_sat_pct": soil_sat,
                "soil_rgb": [sr, sg, sb],
                "soil_detail": detail,
                "soil": soil_name,
                "attempts": attempt,
                "resteps": resteps,
            }
        )
        attempts_used[frame_idx] = attempt
        committed = True

        print(
            f"[{split}] labels: "
            + ", ".join(
                f"{c}={frame_counts[c]}" for c in CLASS_IDS
            )
            + f", plants={n_boxes}/{expected_in}, "
            + f"time={time.time() - t_frame:.1f}s"
        )

    if committed:
        consecutive_failures = 0
    else:
        consecutive_failures += 1
        failed_frames.append(frame_idx)

        reasons = [
            r.get("reason", "untextured_ground")
            for r in retry_log
            if r.get("frame") == frame_idx
        ]
        plant_fail_count = reasons.count("plants_not_rendered")

        print(
            f"[SKIP] frame {frame_idx} not committed after "
            f"{SOIL_MAX_ATTEMPTS} attempts "
            f"({plant_fail_count} plant, "
            f"{len(reasons) - plant_fail_count} ground)"
        )

        tried = frame_idx + 1
        fatal = None

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            fatal = f"{consecutive_failures} consecutive frames failed"
        elif (
            tried >= 40
            and len(failed_frames) / float(tried)
            > MAX_FAILURE_FRACTION
        ):
            fatal = (
                f"{len(failed_frames)} of {tried} frames failed "
                f"({100.0 * len(failed_frames) / tried:.0f}%)"
            )

        if fatal:
            print(f"[ERROR] {fatal}")

            if plant_fail_count:
                print(
                    f"[ERROR] Plant residency failure. "
                    f"Check SETTLE_UPDATES, PLANT_RESTEP_MAX, slot count, "
                    f"and {OUTPUT_DIR}/debug_discarded/"
                )
            else:
                print(
                    f"[ERROR] Ground texture failure. "
                    f"Check files in {SOIL_TEX_DIR} and GPU memory usage."
                )

            break

        continue

    if len(yield_window) >= 20 and (frame_idx + 1) % 20 == 0:
        recent = sum(yield_window[-10:]) / 10.0
        earlier = sum(yield_window[-20:-10]) / 10.0
        print(
            f"[yield] last10={recent * 100:.0f}%, "
            f"prev10={earlier * 100:.0f}%"
        )
        if recent < 0.75 and recent < earlier - 0.10:
            print("[WARN] Render yield is decreasing")

    # Save intermediate COCO annotations during long runs.
    if (
        WRITE_COCO
        and COCO_FLUSH_EVERY
        and (frame_idx + 1) % COCO_FLUSH_EVERY == 0
    ):
        for split in SPLITS:
            with open(
                f"{OUTPUT_DIR}/annotations/instances_{split}.json",
                "w",
            ) as f:
                json.dump(
                    {
                        "info": {
                            "description": "partial flush",
                            "version": "1.0",
                        },
                        "licenses": [],
                        "images": coco_images[split],
                        "annotations": coco_annotations[split],
                        "categories": [
                            {
                                "id": coco_category_id[name],
                                "name": name,
                                "supercategory": "plant",
                            }
                            for name in sorted(
                                CLASS_IDS,
                                key=lambda name: CLASS_IDS[name],
                            )
                        ],
                    },
                    f,
                )

        print(f"[coco] flushed at frame {frame_idx + 1}")

# Summary
coco_paths = {}
if WRITE_COCO:
    categories = [
        {
            "id": coco_category_id[name],
            "name": name,
            "supercategory": "plant",
        }
        for name in sorted(CLASS_IDS, key=lambda n: CLASS_IDS[n])
    ]

    for split in SPLITS:
        doc = {
            "info": {
                "description": "Synthetic sugar beet crop/weed, Isaac Sim",
                "version": "1.0",
            },
            "licenses": [],
            "images": coco_images[split],
            "annotations": coco_annotations[split],
            "categories": categories,
        }
        path = f"{OUTPUT_DIR}/annotations/instances_{split}.json"
        with open(path, "w") as f:
            json.dump(doc, f)
        coco_paths[split] = path
        print(
            f"[coco] {split}: {len(coco_images[split])} images, "
            f"{len(coco_annotations[split])} annotations -> {path}"
        )

    if WRITE_TAO_SPEC:
        ordered = sorted(CLASS_IDS, key=lambda n: coco_category_id[n])

        classmap_path = f"{OUTPUT_DIR}/annotations/classmap.txt"
        with open(classmap_path, "w") as f:
            for name in ordered:
                f.write(f"{name}\n")

        eval_ids = [coco_category_id[n] for n in ordered]
        color_lines = "\n".join(
            f"    {n}: {'green' if n == 'crop' else 'red'}"
            for n in ordered
        )

        tao_yml = f"""results_dir: /results
model:
  backbone: resnet_50
  train_backbone: true
  num_queries: 300
  num_select: 300
  num_feature_levels: 3
  return_interm_indices: [1, 2, 3]

dataset:
  train_data_sources:
    - image_dir: {OUTPUT_DIR}/images/train
      json_file: {OUTPUT_DIR}/annotations/instances_train.json
  val_data_sources:
    image_dir: {OUTPUT_DIR}/images/val
    json_file: {OUTPUT_DIR}/annotations/instances_val.json
  test_data_sources:
    image_dir: {OUTPUT_DIR}/images/val
    json_file: {OUTPUT_DIR}/annotations/instances_val.json
  infer_data_sources:
    image_dir: [{OUTPUT_DIR}/images/val]
    classmap: {classmap_path}

  num_classes: {COCO_NUM_CLASSES}
  remap_mscoco_category: False
  eval_class_ids: {eval_ids}
  batch_size: 4
  workers: 8
  dataset_type: serialized

  augmentation:
    train_spatial_size: [{FINAL_H}, {FINAL_W}]
    eval_spatial_size: [{FINAL_H}, {FINAL_W}]
    multi_scales: [480, 512, 544, 576, 608, 640, 672, 704]
    distortion_prob: 0.8
    iou_crop_prob: 0.8
    preserve_aspect_ratio: false

train:
  num_gpus: 1
  num_epochs: 100
  checkpoint_interval: 5
  validation_interval: 5
  precision: fp32
  optim:
    optimizer: AdamW
    lr: 0.0002
    lr_backbone: 0.00002
    weight_decay: 0.0001
    lr_scheduler: MultiStep
    lr_steps: [70, 90]
    lr_decay: 0.1

evaluate:
  checkpoint: /results/train/model_epoch_099.pth
  conf_threshold: 0.0

inference:
  checkpoint: /results/train/model_epoch_099.pth
  conf_threshold: 0.5
  input_width: {FINAL_W}
  input_height: {FINAL_H}
  color_map:
{color_lines}
"""

        with open(f"{OUTPUT_DIR}/tao_rtdetr_train.yaml", "w") as f:
            f.write(tao_yml)

        print(
            f"[tao] config written, classes={len(ordered)}, "
            f"num_classes={COCO_NUM_CLASSES}, eval_ids={eval_ids}"
        )

# Validate generated COCO annotations.
try:
    from pycocotools.coco import COCO

    for split, path in coco_paths.items():
        coco = COCO(path)
        ann_ids = coco.getAnnIds()
        anns = coco.loadAnns(ann_ids)

        bad = [
            ann
            for ann in anns
            if ann["bbox"][2] <= 0 or ann["bbox"][3] <= 0
        ]

        areas = sorted(ann["area"] for ann in anns)
        n_img = len(coco.getImgIds())
        n_empty = sum(
            1
            for image_id in coco.getImgIds()
            if not coco.getAnnIds(imgIds=image_id)
        )

        print(
            f"[coco] {split}: images={n_img}, negatives={n_empty}, "
            f"boxes={len(anns)}, invalid={len(bad)}"
        )

        if areas:
            print(
                f"[coco] {split} area: "
                f"min={areas[0]:.0f}, "
                f"median={areas[len(areas) // 2]:.0f}, "
                f"max={areas[-1]:.0f}"
            )

        for cat in coco.loadCats(coco.getCatIds()):
            n = len(coco.getAnnIds(catIds=[cat["id"]]))
            print(f"[coco] {cat['name']}: {n}")

except ImportError:
    print("[WARN] pycocotools not installed; COCO validation skipped")
except Exception as exc:
    print(f"[WARN] COCO validation failed: {exc}")

if WRITE_YOLO:
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for name, idx in sorted(CLASS_IDS.items(), key=lambda kv: kv[1]):
            f.write(f"  {idx}: {name}\n")

    print("[yolo] data.yaml written")

meta = {
    "num_images": NUM_IMAGES,
    "image_size": [FINAL_W, FINAL_H],
    "render_size": [IMAGE_W, IMAGE_H],
    "class_ids": CLASS_IDS,
    "annotation_formats": {
        "coco": WRITE_COCO,
        "yolo": WRITE_YOLO,
        "coco_category_id_start": COCO_CATEGORY_ID_START,
        "coco_num_classes": COCO_NUM_CLASSES,
    },
    "split": {
        "val_fraction": VAL_FRACTION,
        "seed": SPLIT_SEED,
        "train": sum(1 for v in frame_split.values() if v == "train"),
        "val": sum(1 for v in frame_split.values() if v == "val"),
    },
    "class_target_footprint_m": CLASS_TARGET_FOOTPRINT,
    "crop_scale_jitter": list(CROP_SCALE_JITTER),
    "weed_footprint_range_m": list(WEED_FOOTPRINT_RANGE),
    "layout": {
        "crop_row_spacing_m": CROP_ROW_SPACING,
        "crop_in_row_spacing_m": CROP_IN_ROW_SPACING,
        "crop_missing_prob": CROP_MISSING_PROB,
        "weeds_per_frame": list(WEEDS_PER_FRAME),
        "layout_margin_m": LAYOUT_MARGIN,
    },
    "camera": {
        "focal_length_mm": FOCAL_LENGTH,
        "horizontal_aperture_mm": HORIZ_APERTURE,
        "height_range_m": list(CAM_HEIGHT),
        "tilt_deg": CAM_TILT,
        "gsd_mm_per_px": [gsd_min * 1000, gsd_max * 1000],
        "ground_coverage_m": [ground_w_min, ground_w_max],
    },
    "lighting": {
        "harsh_prob": HARSH_PROB,
        "regime_counts": regime_counts,
        "reference_luminance": {
            "mean": 202.6,
            "std": 44.1,
            "p1": 52.7,
        },
        "per_frame": lum_log,
    },
    "assets": {
        c: [t["name"] for t in templates[c]] for c in CLASS_IDS
    },
    "total_boxes": class_counts,
    "ruler_cube": RULER_CUBE,
    "texture_qc": {
        "detail_fraction": SOIL_DETAIL_FRACTION,
        "detail_floor": SOIL_DETAIL_FLOOR,
        "retry_same": SOIL_RETRY_SAME,
        "max_attempts": SOIL_MAX_ATTEMPTS,
        "baselines": {
            entry["name"]: entry["baseline_detail"]
            for entry in soil_materials
        },
        "attempts_per_frame": attempts_used,
        "skipped_frames": failed_frames,
        "plant_yield_recent": (
            sum(yield_window) / len(yield_window)
            if yield_window
            else None
        ),
        "render_settings": {
            "mode": RENDER_MODE,
            "total_spp": PATHTRACING_TOTAL_SPP,
            "rt_subframes": RT_SUBFRAMES,
            "instancing": USE_INSTANCING,
            "settle_updates": SETTLE_UPDATES,
        },
        "discarded": retry_log,
        "frames_needing_retry": sum(
            1 for value in attempts_used.values() if value > 1
        ),
    },
}

with open(f"{OUTPUT_DIR}/dataset_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

with open(f"{OUTPUT_DIR}/classes.txt", "w") as f:
    for class_name, _ in sorted(
        CLASS_IDS.items(), key=lambda kv: kv[1]
    ):
        f.write(f"{class_name}\n")

if lum_log:
    harsh_log = [r for r in lum_log if r["regime"] == "harsh"]
    over_log = [r for r in lum_log if r["regime"] == "overcast"]

    mm = float(np.mean([r["mean"] for r in lum_log]))
    ms = float(np.mean([r["std"] for r in lum_log]))
    mp = float(np.mean([r["p1"] for r in lum_log]))

    print(
        f"[luminance] mean={mm:.1f}, std={ms:.1f}, "
        f"p1={mp:.1f}, exposure={EXPOSURE_MODE}"
    )

    for name, log in (("harsh", harsh_log), ("overcast", over_log)):
        if not log:
            continue

        mean = float(np.mean([r["mean"] for r in log]))
        std = float(np.mean([r["std"] for r in log]))
        p1 = float(np.mean([r["p1"] for r in log]))
        print(
            f"[luminance] {name}: mean={mean:.1f}, std={std:.1f}, "
            f"p1={p1:.1f}, ratio={p1 / max(mean, 1e-6):.2f}, "
            f"n={len(log)}"
        )

    if harsh_log:
        hm = float(np.mean([r["mean"] for r in harsh_log]))
        hs = float(np.mean([r["std"] for r in harsh_log]))
        hp = float(np.mean([r["p1"] for r in harsh_log]))
        hd = float(np.mean([r["deep_shadow_pct"] for r in harsh_log]))
        hc = float(np.mean([r["clipped_black_pct"] for r in harsh_log]))

        print(f"[shadow] deep={hd:.1f}%, black={hc:.2f}%")

        if hc > 0.20:
            print("[WARN] Harsh shadows are clipping")
        elif hd > 5.0:
            print("[WARN] Deep-shadow fraction is high")
        elif hd < 1.0 and hs < 35.0:
            print("[WARN] Harsh-light contrast is low")

        if hp / max(hm, 1e-6) > 0.40:
            print("[WARN] Harsh shadows are too bright")

    dark_frames = [r for r in lum_log if r["mean"] < 110.0]
    bright_frames = [r for r in lum_log if r["mean"] > 235.0]
    if dark_frames or bright_frames:
        print(
            f"[brightness] dark={len(dark_frames)}, "
            f"bright={len(bright_frames)}, total={len(lum_log)}"
        )

    gp = float(np.mean([r["green_pct"] for r in lum_log]))
    ss = float(np.nanmean([r["soil_sat_pct"] for r in lum_log]))
    sr = float(np.nanmean([r["soil_rgb"][0] for r in lum_log]))
    sg = float(np.nanmean([r["soil_rgb"][1] for r in lum_log]))
    sb = float(np.nanmean([r["soil_rgb"][2] for r in lum_log]))

    print(f"[green] mean={gp:.2f}%")

    by_stage = {}
    for row in lum_log:
        by_stage.setdefault(
            row.get("stage") or "n/a", []
        ).append(row["green_pct"])

    for stage_name in list(CROP_STAGES.keys()) + ["n/a"]:
        vals = by_stage.get(stage_name)
        if not vals:
            continue
        print(
            f"[green] {stage_name}: n={len(vals)}, "
            f"mean={np.mean(vals):.2f}%, "
            f"range={min(vals):.2f}-{max(vals):.2f}%"
        )

    print(
        f"[soil] rgb={sr:.0f},{sg:.0f},{sb:.0f}, sat={ss:.1f}%"
    )

    if gp < 4.0:
        print("[WARN] Vegetation coverage is low")
    if ss > 20.0:
        print("[WARN] Soil saturation is high")
    if ss < 9.0:
        print("[WARN] Soil saturation is low")

settled = sum(1 for r in lum_log if r.get("resteps"))

if retry_log:
    n_frames = len({r["frame"] for r in retry_log})
    print(f"[qc] retries={len(retry_log)}, frames={n_frames}")
else:
    print("[qc] no discarded frames")

if settled:
    total_resteps = sum(r.get("resteps", 0) for r in lum_log)
    print(
        f"[qc] settle frames={settled}, "
        f"extra steps={total_resteps}"
    )

if lum_log:
    details = [r["soil_detail"] for r in lum_log]
    print(
        f"[soil] detail min={min(details):.2f}, "
        f"median={sorted(details)[len(details) // 2]:.2f}, "
        f"max={max(details):.2f}"
    )

plant_fail = [
    r for r in retry_log if r.get("reason") == "plants_not_rendered"
]

if yield_window:
    plant_yield = sum(yield_window) / len(yield_window)
    print(
        f"[plants] render yield={plant_yield * 100:.0f}% "
        f"({len(yield_window)} frames)"
    )
    if plant_yield < 0.85:
        print("[WARN] Plant render yield below 85%")

if yield_overshoot:
    print(
        f"[WARN] Visibility count mismatch in "
        f"{len(yield_overshoot)} frame(s)"
    )

if plant_fail:
    failed_plant_frames = len({r["frame"] for r in plant_fail})
    print(
        f"[plants] discarded attempts={len(plant_fail)}, "
        f"frames={failed_plant_frames}"
    )

if failed_frames:
    print(
        f"[frames] skipped={len(failed_frames)}: "
        f"{failed_frames[:12]}"
    )

print(
    "[total] "
    + ", ".join(f"{c}={class_counts[c]}" for c in CLASS_IDS)
)
print(f"[regimes] {regime_counts}")

committed = len(attempts_used)
print(f"[frames] committed={committed}/{NUM_IMAGES}")

timing = (
    f"[timing] startup={hms(_T_FRAMES - _T0)}, "
    f"rendering={hms(time.time() - _T_FRAMES)}"
)
if committed:
    timing += f", {(time.time() - _T_FRAMES) / committed:.1f}s/frame"

print(timing)
print(f"[output] {OUTPUT_DIR}")

rep.orchestrator.wait_until_complete()
simulation_app.close()
