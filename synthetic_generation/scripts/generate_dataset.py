from isaacsim import SimulationApp

# config
IMAGE_W, IMAGE_H = 1920, 1920
FINAL_W, FINAL_H = 640, 640
NUM_IMAGES = 15

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

PROJECT_DIR = "/isaac_sim_new_model_fixed"
ASSETS_DIR = f"{PROJECT_DIR}/assets"
OUTPUT_DIR = "/isaac_output"
SOIL_TEX_DIR = f"{ASSETS_DIR}/ground/soil_textures"

CLASS_IDS = {"crop": 0, "weed": 1}

# nnotation format
COCO_CATEGORY_ID_START = 1

WRITE_COCO = True
WRITE_YOLO = True  # keep YOLO txt 
WRITE_TAO_SPEC = True # emit TAO rtdetr spec + classmap
VAL_FRACTION = 0.2
SPLIT_SEED = 1234

# draw boxes on a few frames for debugging
DEBUG_OVERLAY_COUNT = 3

# plant scale
CROP_SIZE_MODE = "preserve"

CROP_FOOTPRINT_CLAMP = (0.13, 0.30)
CENTER_ON_CROWN = True
CROWN_PRIM_NAMES = ("crown",)

CLASS_TARGET_FOOTPRINT = {
    "crop": 0.18,  
    "weed": 0.025,
}

CROP_SCALE_JITTER = (0.85, 1.15)   
WEED_FOOTPRINT_RANGE = (0.018, 0.045)

# scene layout
CROP_ROW_SPACING = 0.50  # meters between crop rows 
CROP_IN_ROW_SPACING = 0.22 # meters between plants within a row
CROP_ROW_JITTER = 0.030 # meters perpendicular to the row
CROP_ALONG_JITTER = 0.045 # meters along the row
CROP_MISSING_PROB = 0.12  # gaps where a seedling failed

WEEDS_PER_FRAME = (3, 16)

MAX_CROP_SLOTS = 10
MAX_WEED_SLOTS = 20 

# clutter: straw and stones
CLUTTER_ENABLED = True

STRAW_PER_FRAME = (0, 22) # 0 lets some frames stay clean
STRAW_LENGTH = (0.020, 0.090) # meters
STRAW_WIDTH = (0.002, 0.004) # meters

STRAW_BEND = (0.0, 0.12)
STRAW_COLORS = [             
    (0.78, 0.74, 0.63),
    (0.70, 0.65, 0.52),
    (0.85, 0.82, 0.72),
    (0.62, 0.57, 0.45),
]

STONES_PER_FRAME = (0, 30)
STONE_SIZE = (0.005, 0.025) # meters
STONE_COLORS = [                
    (0.58, 0.55, 0.49),
    (0.48, 0.46, 0.42),
    (0.66, 0.63, 0.57),
    (0.40, 0.38, 0.35),
]

MAX_STRAW_SLOTS = 24
MAX_STONE_SLOTS = 32

LAYOUT_MARGIN = 0.07

# camera
FOCAL_LENGTH = 24.0
HORIZ_APERTURE = 20.955
CAM_HEIGHT = (0.70, 0.90) 
CAM_XY_JITTER = 0.05
CAM_TILT = 2.0 

GROUND_SIZE = 80.0
MUD_TILES = 0.72
UV_CENTER = 0.5
UV_JITTER = 0.04

MIN_BOX_PX_OUT = 10.0
MIN_BOX_PX = MIN_BOX_PX_OUT * (IMAGE_W / float(FINAL_W))

# lighting
HARSH_PROB = 0.72

LIGHT_HARSH = {
    "sun_intensity": (4200.0, 7000.0),
    "sun_angle": (0.35, 1.10), 
    "sun_zenith": (24.0, 48.0),
    "sun_color": (1.00, 0.96, 0.88),
    "dome_intensity": (420.0, 650.0),
    "dome_color": (0.45, 0.55, 0.78),
    "fill_intensity": (220.0, 420.0),
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

# first-run verification 
RULER_CUBE = False # 10 cm cube at origin; set False for real runs
RULER_SIZE = 0.10          

SOIL_EXCLUDE = ("crack", "playa", "desert", "lakebed", "drought")

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
frame_split = {i: ("val" if i in _val_set else "train") for i in range(NUM_IMAGES)}
print(f"[split] train={NUM_IMAGES - _n_val} val={_n_val} (seed {SPLIT_SEED})")

# COCO accumulators, one per split.
coco_images = {s: [] for s in SPLITS}
coco_annotations = {s: [] for s in SPLITS}
coco_ann_id = {s: 1 for s in SPLITS}

# Maps internal 0-based class index->COCO category_id.
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
        print("[light] exposure: auto (RTX post-processing untouched)")
        return
    try:
        import carb
        s = carb.settings.get_settings()
        s.set("/rtx/post/histogram/enabled", False)
        s.set("/rtx/post/tonemap/op", TONEMAP_OP_ACES)
        s.set("/rtx/post/tonemap/filmIso", MANUAL_EXPOSURE["film_iso"])
        s.set("/rtx/post/tonemap/cameraShutter",
              MANUAL_EXPOSURE["camera_shutter"])
        s.set("/rtx/post/tonemap/fNumber", MANUAL_EXPOSURE["f_number"])
        print(f"[light] exposure: manual, ACES, iso="
              f"{MANUAL_EXPOSURE['film_iso']:.0f} "
              f"shutter=1/{1.0/MANUAL_EXPOSURE['camera_shutter']:.0f} "
              f"f/{MANUAL_EXPOSURE['f_number']:.1f}")
        print("[light] if frames come out white, drop film_iso or raise "
              "f_number; if black, do the reverse")
    except Exception as exc:
        print(f"[WARN] could not set exposure settings: {exc}")


configure_exposure(EXPOSURE_MODE)


# semantics API
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


# asset discovery 
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
                    print(
                        f"[WARN] skipping EXR normal map: {full_path}. "
                        "Convert it to PNG or download a PNG normal map."
                    )
                    break
                roles[role] = full_path
                break

        if "diffuse" not in roles:
            continue

        soil_sets.append({
            "name": os.path.basename(root),
            "directory": root,
            "diffuse": roles["diffuse"],
            "normal": roles.get("normal", ""),
            "roughness": roles.get("roughness", ""),
            "specular": roles.get("specular", ""),
            "displacement": roles.get("displacement", ""),
        })

    soil_sets.sort(key=lambda item: item["name"])

    if SOIL_EXCLUDE:
        kept, dropped = [], []
        for s_ in soil_sets:
            if any(k in s_["name"].lower() for k in SOIL_EXCLUDE):
                dropped.append(s_["name"])
            else:
                kept.append(s_)
        if dropped:
            print(f"[soil] excluded {len(dropped)} set(s) by SOIL_EXCLUDE: "
                  f"{', '.join(dropped)}")
        if not kept:
            print("[soil] [WARN] SOIL_EXCLUDE removed every texture set; "
                  "keeping them all so the run is not flat brown")
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
        rng_ = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if not rng_.IsEmpty():
            return rng_, label
    rng_ = bounds_from_points(st)
    if rng_ is not None and not rng_.IsEmpty():
        return rng_, "raw mesh points"
    return None, "no geometry points found anywhere in the file"


def find_crown_bounds(st, rot):
    dflt = st.GetDefaultPrim()
    for prim in Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies(
            Usd.PrimDefaultPredicate)):
        # Never match the stage root: its bounds are the whole plant, so
        # "centring on the crown" would silently become bbox centring.
        if dflt and prim == dflt:
            continue
        nm = prim.GetName().lower()
        if not any(k in nm for k in CROWN_PRIM_NAMES):
            continue
        rng_ = bounds_from_points(st, prim)
        if rng_ is None or rng_.IsEmpty():
            continue
        mn, mx = Gf.Vec3d(rng_.GetMin()), Gf.Vec3d(rng_.GetMax())
        corners = [Gf.Vec3d(x, y, z)
                   for x in (mn[0], mx[0])
                   for y in (mn[1], mx[1])
                   for z in (mn[2], mx[2])]
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

    # scale
    if cls == "crop" and CROP_SIZE_MODE == "preserve":
        scale = 1.0
        scale_note = "native"
        if CROP_FOOTPRINT_CLAMP:
            lo, hi = CROP_FOOTPRINT_CLAMP
            if raw_fp > hi:
                scale = hi / raw_fp
                scale_note = f"CLAMPED down from {raw_fp*1000:.0f}mm"
            elif raw_fp < lo:
                scale = lo / raw_fp
                scale_note = f"CLAMPED up from {raw_fp*1000:.0f}mm"
    else:
        scale = target_footprint / raw_fp
        scale_note = "normalized"

    # XY centre
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


# discovery run
assets = discover_assets(ASSETS_DIR)
soil_sets = discover_soil_textures(SOIL_TEX_DIR)

gsd_min = (HORIZ_APERTURE / FOCAL_LENGTH) * CAM_HEIGHT[0] / IMAGE_W
gsd_max = (HORIZ_APERTURE / FOCAL_LENGTH) * CAM_HEIGHT[1] / IMAGE_W
ground_w_min = gsd_min * IMAGE_W
ground_w_max = gsd_max * IMAGE_W

print("=" * 74)
print("ASSET DISCOVERY")
print("=" * 74)

templates = {}
for cls, paths in assets.items():
    templates[cls] = []
    print(f"\n[{cls}]  {len(paths)} file(s)  "
          f"base footprint {CLASS_TARGET_FOOTPRINT[cls]:.3f} m")
    if not paths:
        print(f"  (none found under {ASSETS_DIR}/{cls}/)")
        continue
    for p in paths:
        info, err = measure_asset(p, CLASS_TARGET_FOOTPRINT[cls], cls=cls)
        if info is None:
            print(f"  SKIP {os.path.basename(p)}: {err}")
            continue
        if cls == "crop":
            fp_lo = info["final_footprint"] * CROP_SCALE_JITTER[0]
            fp_hi = info["final_footprint"] * CROP_SCALE_JITTER[1]
        else:
            fp_lo, fp_hi = WEED_FOOTPRINT_RANGE
        px_min = fp_lo / gsd_max
        px_max = fp_hi / gsd_min
        warn = ""
        if info["raw_footprint"] > 1.0:
            warn = "  [WARN] >1 m raw: multi-variant pack?"
        if abs(info["mpu"] - 1.0) > 1e-6:
            warn += f"  [WARN] metersPerUnit={info['mpu']}"
        if cls == "crop" and not (0.05 <= info["final_footprint"] <= 0.40):
            warn += (f"  [WARN] final footprint "
                     f"{info['final_footprint']:.3f} m is outside the 0.05-0.40 m "
                     f"range plausible for beet; check CROP_SIZE_MODE")
        print(f"  {info['name']:<30s} raw {info['raw_footprint']:.3f} m "
              f"-> x{info['scale']:.3f} = {info['final_footprint']:.3f} m "
              f"({info['scale_note']})  h={info['final_height']:.3f} m")
        print(f"      {fp_lo:.3f}-{fp_hi:.3f} m after jitter -> "
              f"~{px_min * FINAL_W / IMAGE_W:.0f}-{px_max * FINAL_W / IMAGE_W:.0f} px "
              f"@{FINAL_W}   centre: {info['center_note']}{warn}")
        templates[cls].append(info)

print(f"\n[soil]  {len(soil_sets)} texture set(s) under {SOIL_TEX_DIR}")
for soil in soil_sets:
    available_maps = [
        role for role in
        ("diffuse", "normal", "roughness", "specular", "displacement")
        if soil.get(role)
    ]
    print(f"  {soil['name']:<28s} maps={','.join(available_maps)}")

print(f"\n[camera]  focal {FOCAL_LENGTH}mm, aperture {HORIZ_APERTURE}mm, "
      f"height {CAM_HEIGHT[0]}-{CAM_HEIGHT[1]} m")
print(f"          GSD {gsd_min*1000:.3f}-{gsd_max*1000:.3f} mm/px at {IMAGE_W} px")
print(f"          ground coverage {ground_w_min:.3f}-{ground_w_max:.3f} m")
print(f"          reference frame measured at ~0.45-0.55 m")

_tile_period = 1.0 / (MUD_TILES * 1.10)      # smallest period after jitter
_worst_span = ground_w_max * math.sqrt(2.0)
print(f"\n[soil]    tile period {_tile_period:.3f} m (at max tiling jitter)")
print(f"          worst-case visible span {_worst_span:.3f} m "
      f"(frame diagonal, camera yaws freely)")
if _worst_span >= _tile_period:
    print(f"          [WARN] span >= period: a texture tile boundary WILL "
          f"cross some frames as a hard straight seam.")
    print(f"          Lower MUD_TILES below "
          f"{1.0/(_worst_span*1.10):.2f} to avoid it.")
else:
    _margin = (_tile_period - _worst_span) / 2.0
    _jit = UV_JITTER * _tile_period
    print(f"          OK: window fits inside one tile "
          f"(offset margin {_margin*1000:.0f} mm each way)")
    if _jit > _margin:
        print(f"          [WARN] UV_JITTER moves the window {_jit*1000:.0f} mm, "
              f"more than the {_margin*1000:.0f} mm margin -> reduce "
              f"UV_JITTER below {_margin/_tile_period:.3f}")
    else:
        print(f"          UV offset {UV_CENTER} +/- {UV_JITTER} "
              f"({_jit*1000:.0f} mm) keeps boundaries out of frame")
if RULER_CUBE:
    print(f"          RULER CUBE ON: {RULER_SIZE*100:.0f} cm cube should measure "
          f"{RULER_SIZE/gsd_max:.0f}-{RULER_SIZE/gsd_min:.0f} px")
print("=" * 74)

active_classes = [c for c in CLASS_IDS if templates.get(c)]
assert active_classes, "No assets found. Check ASSETS_DIR and the folder tree."
if "crop" not in active_classes:
    print("\n[WARN] No crop assets: this run produces WEED-ONLY data.\n")


# materials
def make_mud_material(stage, path, color, tiling=1.0):
    mtl = UsdShade.Material.Define(stage, Sdf.Path(path))
    shader = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/Shader"))
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset("OmniPBR.mdl", "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")

    shader.CreateInput("diffuse_color_constant",
                       Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("diffuse_tint",
                       Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1, 1, 1))
    shader.CreateInput("reflection_roughness_constant",
                       Sdf.ValueTypeNames.Float).Set(0.95)

    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(1.0)
    shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset)
    shader.CreateInput("reflection_roughness_texture_influence",
                       Sdf.ValueTypeNames.Float).Set(0.0)

    shader.CreateInput("texture_scale",
                       Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(tiling, tiling))
    shader.CreateInput("texture_translate",
                       Sdf.ValueTypeNames.Float2).Set(
                           Gf.Vec2f(UV_CENTER, UV_CENTER))
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)

    mtl.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mtl, shader


# scene build
ground = rep.create.plane(scale=(GROUND_SIZE / 2.0, GROUND_SIZE / 2.0, 1))

ground_prim = None
for prim in stage.Traverse():
    if prim.GetTypeName() == "Mesh" and "Plane" in prim.GetName():
        ground_prim = prim
assert ground_prim is not None, "Ground plane prim not found"

mud_mtl, mud_shader = make_mud_material(
    stage, "/World/Looks/Mud",
    color=(0.35, 0.22, 0.10),
    tiling=MUD_TILES,
)
UsdShade.MaterialBindingAPI(ground_prim).Bind(
    mud_mtl, UsdShade.Tokens.strongerThanDescendants)

if RULER_CUBE:
    rep.create.cube(
        position=(0.0, 0.0, RULER_SIZE / 2.0),
        scale=(RULER_SIZE, RULER_SIZE, RULER_SIZE),
    )


# camera 
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

rp = rep.create.render_product("/World/CameraRig/Camera",
                               resolution=(IMAGE_W, IMAGE_H))


# lights
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


# plant slots
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
        for i, info in enumerate(templates[cls]):
            vpath = f"{root}/var_{i}"
            v = UsdGeom.Xform.Define(stage, vpath)
            v.AddTranslateOp().Set(info["offset"])
            if info["y_up"]:
                v.AddRotateXOp().Set(90.0)
            sc = info["scale"]
            v.AddScaleOp().Set(Gf.Vec3f(sc, sc, sc))

            geo = UsdGeom.Xform.Define(stage, f"{vpath}/geo")
            geo.GetPrim().GetReferences().AddReference(info["path"])

            UsdGeom.Imageable(v).MakeInvisible()
            variants.append(v)

        UsdGeom.Imageable(slot).MakeInvisible()
        slots.append({
            "xform": slot,
            "imageable": UsdGeom.Imageable(slot.GetPrim()),
            "t": t_op, "r": r_op, "s": s_op,
            "variants": variants,
            "active_variant": None,
        })
    return slots


slot_pool = {}
if "crop" in active_classes:
    slot_pool["crop"] = build_slots("crop", MAX_CROP_SLOTS)
if "weed" in active_classes:
    slot_pool["weed"] = build_slots("weed", MAX_WEED_SLOTS)

print(f"[slots] crop={len(slot_pool.get('crop', []))} "
      f"weed={len(slot_pool.get('weed', []))}")

def make_clutter_material(stage, path, color, rough):
    mtl = UsdShade.Material.Define(stage, Sdf.Path(path))
    sh = UsdShade.Shader.Define(stage, Sdf.Path(f"{path}/Shader"))
    sh.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    sh.SetSourceAsset("OmniPBR.mdl", "mdl")
    sh.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    sh.CreateInput("diffuse_color_constant",
                   Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    sh.CreateInput("reflection_roughness_constant",
                   Sdf.ValueTypeNames.Float).Set(rough)
    mtl.CreateSurfaceOutput("mdl").ConnectToSource(sh.ConnectableAPI(), "out")
    return mtl


def build_straw_mesh(stage, path, length, width, bend, sides=6, segs=7):
    """A tapered, slightly bent tube: what a straw fragment actually is.

    Built as explicit geometry rather than loaded from an asset, because at
    18-82 px a scanned model would carry no information a tapered tube does not.
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts, counts, idx = [], [], []
    for j in range(segs):
        t = j / (segs - 1.0)
        # taper toward both ends, thickest around a third along
        r = width * 0.5 * (0.45 + 0.55 * math.sin(math.pi * (0.25 + 0.75 * t)))
        z = -bend * length * math.sin(math.pi * t)     # sag
        x = (t - 0.5) * length
        for i in range(sides):
            a = 2.0 * math.pi * i / sides
            pts.append(Gf.Vec3f(x,
                                float(r * math.cos(a)),
                                float(z + r * math.sin(a))))
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
    # sit the straw on the ground rather than half-buried
    lowest = min(p[2] for p in pts)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, float(-lowest)))
    return mesh


def build_stone_mesh(stage, path, size, seed=0, subdiv=1):
    r = size * 0.5
    t = (1.0 + math.sqrt(5.0)) / 2.0
    base = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0),
            (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
            (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
             (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
             (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
             (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [Gf.Vec3f(*v) for v in base]
    for _ in range(subdiv):
        cache, new_faces = {}, []
        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k in cache:
                return cache[k]
            m = (verts[a] + verts[b]) * 0.5
            verts.append(m)
            cache[k] = len(verts) - 1
            return cache[k]
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc),
                          (ab, bc, ca)]
        faces = new_faces

    lrng = random.Random(seed)
    pts = []
    for v in verts:
        n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2) or 1.0
        k = r * lrng.uniform(0.72, 1.12)
        pts.append(Gf.Vec3f(float(v[0] / n * k),
                            float(v[1] / n * k),
                            float(v[2] / n * k * 0.60)))   # flattened
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(pts)
    mesh.CreateFaceVertexCountsAttr([3] * len(faces))
    idx = []
    for f in faces:
        idx.extend(f)
    mesh.CreateFaceVertexIndicesAttr(idx)
    mesh.CreateDoubleSidedAttr(False)
    lowest = min(p[2] for p in pts)
    UsdGeom.Xformable(mesh).AddTranslateOp().Set(
        Gf.Vec3d(0.0, 0.0, float(-lowest)))
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
                bend = STRAW_BEND[0] + (STRAW_BEND[1] - STRAW_BEND[0]) * \
                    ((i * 0.37) % 1.0)
                build_straw_mesh(stage, f"{vpath}/geo", length, width, bend)
                col = STRAW_COLORS[i % len(STRAW_COLORS)]
                rough = 0.85
            else:
                lo, hi = STONE_SIZE
                size = lo + (hi - lo) * (i / max(n_variants - 1, 1))
                build_stone_mesh(stage, f"{vpath}/geo", size, seed=s * 17 + i)
                col = STONE_COLORS[i % len(STONE_COLORS)]
                rough = 0.72
            mtl = make_clutter_material(
                stage, f"/World/Looks/{kind}_{s:02d}_{i}", col, rough)
            geo_prim = stage.GetPrimAtPath(f"{vpath}/geo")
            UsdShade.MaterialBindingAPI(geo_prim).Bind(
                mtl, UsdShade.Tokens.strongerThanDescendants)
            UsdGeom.Imageable(v).MakeInvisible()
            variants.append(v)

        UsdGeom.Imageable(slot).MakeInvisible()
        slots.append({
            "xform": slot,
            "imageable": UsdGeom.Imageable(slot.GetPrim()),
            "t": t_op, "rz": rz_op, "ry": ry_op, "s": sc_op,
            "variants": variants,
            "active_variant": None,
        })
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
    print(f"[clutter] straw={len(clutter_pool['straw'])} "
          f"stone={len(clutter_pool['stone'])}  "
          f"UNLABELLED (no semantics -> excluded from COCO)")
    _gsd_mm = gsd_max * 1000.0
    print(f"[clutter] straw {STRAW_LENGTH[0]*1000:.0f}-"
          f"{STRAW_LENGTH[1]*1000:.0f} mm -> "
          f"{STRAW_LENGTH[0]/gsd_max*FINAL_W/IMAGE_W:.0f}-"
          f"{STRAW_LENGTH[1]/gsd_min*FINAL_W/IMAGE_W:.0f} px at {FINAL_W}")
    print(f"[clutter] stone {STONE_SIZE[0]*1000:.0f}-"
          f"{STONE_SIZE[1]*1000:.0f} mm -> "
          f"{STONE_SIZE[0]/gsd_max*FINAL_W/IMAGE_W:.0f}-"
          f"{STONE_SIZE[1]/gsd_min*FINAL_W/IMAGE_W:.0f} px at {FINAL_W}")

    # Hard check: a clutter prim carrying semantics would be annotated as a
    # plant, which is worse than having no clutter at all. Verify none does.
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
        print(f"[clutter] [FATAL] {len(tagged)} clutter prim(s) carry "
              f"semantics and WOULD be annotated as plants:")
        for t in tagged[:5]:
            print(f"           {t}")
        raise SystemExit("clutter must remain unlabelled")
    print(f"[clutter] verified: no semantics on any clutter prim")


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


# layout
def crop_row_layout(rng, half_extent):
    reach = half_extent + LAYOUT_MARGIN
    theta = rng.uniform(0.0, math.pi)
    dx, dy = math.cos(theta), math.sin(theta)
    px, py = -dy, dx

    base = rng.uniform(-CROP_ROW_SPACING / 2.0, CROP_ROW_SPACING / 2.0)
    phase = rng.uniform(0.0, CROP_IN_ROW_SPACING)

    diag = reach * math.sqrt(2.0)
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
            if abs(x) > reach or abs(y) > reach:
                continue
            out.append((x, y))

    rng.shuffle(out)
    return out[:MAX_CROP_SLOTS], math.degrees(theta)


def clutter_layout(rng, half_extent, n, avoid, avoid_r):
    reach = half_extent + LAYOUT_MARGIN
    out = []
    for _ in range(n):
        for _try in range(8):
            x = rng.uniform(-reach, reach)
            y = rng.uniform(-reach, reach)
            if all((x - ax) ** 2 + (y - ay) ** 2 > avoid_r ** 2
                   for ax, ay in avoid):
                break
        out.append((x, y))
    return out


def weed_layout(rng, half_extent):
    reach = half_extent + LAYOUT_MARGIN
    n = rng.randint(WEEDS_PER_FRAME[0], WEEDS_PER_FRAME[1])
    n = min(n, MAX_WEED_SLOTS)
    return [(rng.uniform(-reach, reach), rng.uniform(-reach, reach))
            for _ in range(n)]


# lighting
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
    fill_rotate.Set(Gf.Vec3f(-20.0, 0.0, float((azim + 180.0) % 360.0)))

    dome.GetIntensityAttr().Set(rng.uniform(*cfg["dome_intensity"]))
    dome.GetColorAttr().Set(Gf.Vec3f(*cfg["dome_color"]))

    return "harsh" if harsh else "overcast"


def apply_soil(rng):
    if not soil_sets:
        return None
    sel = rng.choice(soil_sets)
    mud_shader.GetInput("diffuse_texture").Set(sel["diffuse"])
    mud_shader.GetInput("normalmap_texture").Set(sel["normal"] or "")

    if sel["normal"]:
        mud_shader.GetInput("bump_factor").Set(rng.uniform(1.10, 1.80))
    else:
        mud_shader.GetInput("bump_factor").Set(0.0)

    if sel["roughness"]:
        mud_shader.GetInput("reflectionroughness_texture").Set(sel["roughness"])
        mud_shader.GetInput(
            "reflection_roughness_texture_influence").Set(1.0)
    else:
        mud_shader.GetInput("reflectionroughness_texture").Set("")
        mud_shader.GetInput(
            "reflection_roughness_texture_influence").Set(0.0)
        mud_shader.GetInput(
            "reflection_roughness_constant").Set(rng.uniform(0.72, 0.98))

    mud_shader.GetInput("diffuse_tint").Set(Gf.Vec3f(*rng.choice(SOIL_TINTS)))

    ts = MUD_TILES * rng.uniform(0.85, 1.10)
    mud_shader.GetInput("texture_scale").Set(Gf.Vec2f(ts, ts))

    mud_shader.GetInput("texture_translate").Set(Gf.Vec2f(
        UV_CENTER + rng.uniform(-UV_JITTER, UV_JITTER),
        UV_CENTER + rng.uniform(-UV_JITTER, UV_JITTER),
    ))
    return sel, ts


def apply_soil_none(rng):
    return None, MUD_TILES


def apply_camera(rng, tiling):

    h = rng.uniform(*CAM_HEIGHT)
    period = 1.0 / max(tiling, 1e-6)

    span = (HORIZ_APERTURE / FOCAL_LENGTH) * h * math.sqrt(2.0)
    margin = (period - span) / 2.0

    jit = max(0.0, margin * 0.85)
    kx = rng.randint(-20, 20)
    ky = rng.randint(-20, 20)
    cx = (kx + 0.5) * period + rng.uniform(-jit, jit)
    cy = (ky + 0.5) * period + rng.uniform(-jit, jit)

    cam_translate.Set(Gf.Vec3d(cx, cy, h))
    cam_rotate.Set(Gf.Vec3f(
        rng.uniform(-CAM_TILT, CAM_TILT),
        rng.uniform(-CAM_TILT, CAM_TILT),
        rng.uniform(0.0, 360.0),
    ))
    return h, (HORIZ_APERTURE / FOCAL_LENGTH) * h / 2.0, margin, (cx, cy)


# annotators 
rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
rgb_annot.attach([rp])
bbox_annot.attach([rp])

if soil_sets:
    for soil in soil_sets:
        mud_shader.GetInput("diffuse_texture").Set(soil["diffuse"])
        mud_shader.GetInput("normalmap_texture").Set(soil["normal"] or "")
        if soil["roughness"]:
            mud_shader.GetInput(
                "reflectionroughness_texture").Set(soil["roughness"])
            mud_shader.GetInput(
                "reflection_roughness_texture_influence").Set(1.0)
        else:
            mud_shader.GetInput("reflectionroughness_texture").Set("")
            mud_shader.GetInput(
                "reflection_roughness_texture_influence").Set(0.0)
        rep.orchestrator.step(rt_subframes=48)
else:
    print("[WARN] no soil textures found - ground will be flat brown.")
    rep.orchestrator.step(rt_subframes=48)


# main loop 
class_counts = {c: 0 for c in CLASS_IDS}
regime_counts = {"harsh": 0, "overcast": 0}
lum_log = []

for frame_idx in range(NUM_IMAGES):
    print(f"Frame {frame_idx + 1}/{NUM_IMAGES}")

    selected_soil, tiling = apply_soil(rng)
    regime = apply_lighting(rng)
    regime_counts[regime] += 1
    cam_h, half_extent, seam_margin, (cam_x, cam_y) = apply_camera(rng, tiling)

    OX, OY = cam_x, cam_y

    # crops: rows
    n_crop = 0
    row_deg = None
    crop_positions_used = []
    if "crop" in slot_pool:
        positions, row_deg = crop_row_layout(rng, half_extent)
        n_var = len(templates["crop"])
        for slot, (x, y) in zip(slot_pool["crop"], positions):
            place_slot(
                slot, x + OX, y + OY,
                yaw_deg=rng.uniform(0.0, 360.0),
                scale_mult=rng.uniform(*CROP_SCALE_JITTER),
                variant_idx=rng.randrange(n_var),
            )
            crop_positions_used.append((x, y))
        n_crop = min(len(positions), len(slot_pool["crop"]))
        for slot in slot_pool["crop"][n_crop:]:
            hide_slot(slot)

    # weeds: scattered
    n_weed = 0
    if "weed" in slot_pool:
        positions = weed_layout(rng, half_extent)
        n_var = len(templates["weed"])
        base_fp = CLASS_TARGET_FOOTPRINT["weed"]
        for slot, (x, y) in zip(slot_pool["weed"], positions):
            fp = rng.uniform(*WEED_FOOTPRINT_RANGE)
            place_slot(
                slot, x + OX, y + OY,
                yaw_deg=rng.uniform(0.0, 360.0),
                scale_mult=fp / base_fp,
                variant_idx=rng.randrange(n_var),
            )
        n_weed = min(len(positions), len(slot_pool["weed"]))
        for slot in slot_pool["weed"][n_weed:]:
            hide_slot(slot)

    # clutter: unlabelled straw and stones
    n_straw = n_stone = 0
    if CLUTTER_ENABLED:
        avoid = [(x, y) for (x, y) in crop_positions_used]

        n_straw = rng.randint(*STRAW_PER_FRAME)
        n_straw = min(n_straw, len(clutter_pool["straw"]))
        pts = clutter_layout(rng, half_extent, n_straw, avoid, 0.06)
        nv = len(clutter_pool["straw"][0]["variants"]) if n_straw else 1
        for slot, (x, y) in zip(clutter_pool["straw"], pts):
            place_clutter(
                slot, x + OX, y + OY,
                yaw=rng.uniform(0.0, 360.0),
                tilt=rng.uniform(-8.0, 8.0),
                scale_mult=rng.uniform(0.75, 1.35),
                variant_idx=rng.randrange(nv),
            )
        for slot in clutter_pool["straw"][n_straw:]:
            hide_clutter(slot)

        n_stone = rng.randint(*STONES_PER_FRAME)
        n_stone = min(n_stone, len(clutter_pool["stone"]))
        pts = clutter_layout(rng, half_extent, n_stone, avoid, 0.05)
        nv = len(clutter_pool["stone"][0]["variants"]) if n_stone else 1
        for slot, (x, y) in zip(clutter_pool["stone"], pts):
            place_clutter(
                slot, x + OX, y + OY,
                yaw=rng.uniform(0.0, 360.0),
                tilt=0.0,
                scale_mult=rng.uniform(0.7, 1.4),
                variant_idx=rng.randrange(nv),
            )
        for slot in clutter_pool["stone"][n_stone:]:
            hide_clutter(slot)

    soil_name = selected_soil["name"] if selected_soil else "none"
    row_txt = f"{row_deg:.0f}deg" if row_deg is not None else "n/a"
    print(f"  light={regime}  soil={soil_name}  cam_h={cam_h:.3f}m  "
          f"frame={half_extent*2:.3f}m  row={row_txt}  "
          f"placed crop={n_crop} weed={n_weed} "
          f"straw={n_straw} stone={n_stone} (clutter unlabelled)")
    print(f"  tile period={1.0/max(tiling,1e-6):.3f}m  "
          f"cam at tile centre ({cam_x:+.2f},{cam_y:+.2f})  "
          f"seam margin={seam_margin*1000:+.0f}mm"
          f"{'  [WARN] NEGATIVE -> seam will appear' if seam_margin <= 0 else ''}")

    rep.orchestrator.step(rt_subframes=48)

    rgb = rgb_annot.get_data()
    if rgb[:, :, :3].std() < 3.0:
        print(f"  [WARN] frame {frame_idx} looks flat "
              f"({rgb.std():.2f} std) - retrying")
        rep.orchestrator.step(rt_subframes=48)
        rgb = rgb_annot.get_data()

    a = rgb[:, :, :3].astype(np.float32)
    lum = 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]
    med = float(np.median(lum))
    deep_frac = float((lum < 0.30 * med).mean() * 100.0)
    shadow_frac = float((lum < 0.55 * med).mean() * 100.0)
    clipped_black = float((lum < 1.0).mean() * 100.0)
    stats = (float(lum.mean()), float(lum.std()), float(np.percentile(lum, 1)))

    _R, _G, _B = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    _s = _R + _G + _B + 1e-6
    _exg = (2.0 * _G - _R - _B) / _s
    green_mask = _exg > 0.08
    green_frac = float(green_mask.mean() * 100.0)
    soil_mask = ~green_mask
    if soil_mask.sum() > 1000:
        sr = float(_R[soil_mask].mean())
        sg = float(_G[soil_mask].mean())
        sb = float(_B[soil_mask].mean())
        smax, smin = max(sr, sg, sb), min(sr, sg, sb)
        soil_sat = float((smax - smin) / max(smax, 1e-6) * 100.0)
    else:
        sr = sg = sb = soil_sat = float("nan")

    lum_log.append({"frame": frame_idx, "regime": regime,
                    "mean": stats[0], "std": stats[1], "p1": stats[2],
                    "median": med, "deep_shadow_pct": deep_frac,
                    "shadow_pct": shadow_frac,
                    "clipped_black_pct": clipped_black,
                    "green_pct": green_frac, "soil_sat_pct": soil_sat,
                    "soil_rgb": [sr, sg, sb]})
    print(f"  lum mean={stats[0]:.1f} std={stats[1]:.1f} p1={stats[2]:.1f}  "
          f"(ref 202.6 / 44.1 / 52.7)")
    print(f"  shadow {shadow_frac:.1f}% deep {deep_frac:.1f}% "
          f"crushed-black {clipped_black:.2f}%  "
          f"(ref 7.1% / 2.3% / 0.00%)")
    print(f"  green {green_frac:.2f}% (ref 8.0%)  "
          f"soil RGB {sr:.0f},{sg:.0f},{sb:.0f} sat {soil_sat:.1f}% "
          f"(ref 157,151,135 / 15.4%)")

    if frame_idx == 0 and (stats[1] < 3.0 or stats[0] > 250.0 or stats[0] < 5.0):
        print("=" * 74)
        print("ABORT: first frame is degenerate "
              f"(mean={stats[0]:.1f}, std={stats[1]:.1f})")
        if stats[0] > 250.0:
            print("  Fully clipped to white. This is exposure, not lighting.")
            print("  - EXPOSURE_MODE == 'manual': raise f_number or lower "
                  "film_iso, and make sure tonemap op is NOT 1 (Linear).")
            print("  - EXPOSURE_MODE == 'auto': lower sun_intensity in "
                  "LIGHT_HARSH, or check for a stray light in the stage.")
        elif stats[0] < 5.0:
            print("  Fully black. Check that the sun is above the horizon "
                  "and that EXPOSURE_MODE settings are not starving it.")
        else:
            print("  Flat but not clipped: usually textures still loading, "
                  "or the camera is looking away from the ground plane.")
        print(f"  Debug frame written to {OUTPUT_DIR}/debug_frame_000.png")
        print("=" * 74)
        Image.fromarray(rgb[:, :, :3]).resize(
            (FINAL_W, FINAL_H), Image.LANCZOS
        ).save(f"{OUTPUT_DIR}/debug_frame_000.png")
        simulation_app.close()
        raise SystemExit(1)

    split = frame_split[frame_idx]
    file_name = f"frame_{frame_idx:05d}.png"

    img = Image.fromarray(rgb[:, :, :3])
    img = img.resize((FINAL_W, FINAL_H), Image.LANCZOS)
    img.save(f"{OUTPUT_DIR}/images/{split}/{file_name}")

    bb = bbox_annot.get_data()
    boxes = bb["data"]
    id_to_labels = bb["info"]["idToLabels"]

    sx = FINAL_W / float(IMAGE_W)
    sy = FINAL_H / float(IMAGE_H)

    frame_counts = {c: 0 for c in CLASS_IDS}
    yolo_lines = []
    overlay_boxes = []

    for box in boxes:
        sem_id = int(box["semanticId"])
        label = id_to_labels.get(sem_id, id_to_labels.get(str(sem_id), {}))
        if isinstance(label, dict):
            label = label.get("class", "")
        label = str(label)

        cls_name = next((c for c in CLASS_IDS if c in label), None)
        if cls_name is None:
            continue

        x1, y1 = float(box["x_min"]), float(box["y_min"])
        x2, y2 = float(box["x_max"]), float(box["y_max"])

        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(IMAGE_W), x2), min(float(IMAGE_H), y2)
        w, h = x2 - x1, y2 - y1
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue

        # COCO: absolute pixels [x, y, w, h] 
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
            coco_annotations[split].append({
                "id": coco_ann_id[split],
                "image_id": frame_idx,
                "category_id": coco_category_id[cls_name],
                "bbox": [bx, by, bw, bh],
                "area": round(bw * bh, 2),
                "iscrowd": 0,
                "segmentation": [],
            })
            coco_ann_id[split] += 1

        # YOLO: normalized centre-x, centre-y, w, h
        if WRITE_YOLO:
            yolo_lines.append(
                f"{CLASS_IDS[cls_name]} "
                f"{(x1 + x2) / 2.0 / IMAGE_W:.6f} "
                f"{(y1 + y2) / 2.0 / IMAGE_H:.6f} "
                f"{w / IMAGE_W:.6f} {h / IMAGE_H:.6f}"
            )

        overlay_boxes.append((fx1, fy1, fw, fh, cls_name))
        frame_counts[cls_name] += 1
        class_counts[cls_name] += 1

    if WRITE_COCO:
        coco_images[split].append({
            "id": frame_idx,
            "file_name": file_name,
            "width": FINAL_W,
            "height": FINAL_H,
        })

    if WRITE_YOLO:
        with open(f"{OUTPUT_DIR}/labels/{split}/frame_{frame_idx:05d}.txt",
                  "w") as f:
            f.write("\n".join(yolo_lines) + ("\n" if yolo_lines else ""))

    if frame_idx < DEBUG_OVERLAY_COUNT and overlay_boxes:
        from PIL import ImageDraw
        ov = img.copy().convert("RGB")
        draw = ImageDraw.Draw(ov)
        for fx1, fy1, fw, fh, cname in overlay_boxes:
            colour = (0, 255, 0) if cname == "crop" else (255, 64, 64)
            draw.rectangle([fx1, fy1, fx1 + fw, fy1 + fh],
                           outline=colour, width=2)
            draw.text((fx1 + 2, max(0, fy1 - 10)), cname, fill=colour)
        ov.save(f"{OUTPUT_DIR}/debug_overlays/frame_{frame_idx:05d}.png")

    print(f"  [{split}] labels: "
          + ", ".join(f"{c}={frame_counts[c]}" for c in CLASS_IDS))

coco_paths = {}
if WRITE_COCO:
    categories = [
        {"id": coco_category_id[name], "name": name, "supercategory": "plant"}
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
        print(f"[coco] {split}: {len(coco_images[split])} images, "
              f"{len(coco_annotations[split])} annotations -> {path}")

    custom_yml = f"""task: detection

evaluator:
  type: CocoEvaluator
  iou_types: ['bbox', ]

num_classes: {COCO_NUM_CLASSES}
remap_mscoco_category: False

train_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {OUTPUT_DIR}/images/train
    ann_file: {OUTPUT_DIR}/annotations/instances_train.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: True
  num_workers: 4
  drop_last: True
  collate_fn:
    type: BatchImageCollateFunction

val_dataloader:
  type: DataLoader
  dataset:
    type: CocoDetection
    img_folder: {OUTPUT_DIR}/images/val
    ann_file: {OUTPUT_DIR}/annotations/instances_val.json
    return_masks: False
    transforms:
      type: Compose
      ops: ~
  shuffle: False
  num_workers: 4
  drop_last: False
  collate_fn:
    type: BatchImageCollateFunction
"""
    with open(f"{OUTPUT_DIR}/custom_detection.yml", "w") as f:
        f.write(custom_yml)
    print(f"[coco] wrote {OUTPUT_DIR}/custom_detection.yml "
          f"(num_classes={COCO_NUM_CLASSES}, category ids start at "
          f"{COCO_CATEGORY_ID_START})")

    # TAO Toolkit
    if WRITE_TAO_SPEC:
        ordered = sorted(CLASS_IDS, key=lambda n: coco_category_id[n])

        classmap_path = f"{OUTPUT_DIR}/annotations/classmap.txt"
        with open(classmap_path, "w") as f:
            for name in ordered:
                f.write(f"{name}\n")

        eval_ids = [coco_category_id[n] for n in ordered]
        color_lines = "\n".join(
            f"    {n}: {'green' if n == 'crop' else 'red'}" for n in ordered
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
        print(f"[tao]  wrote {OUTPUT_DIR}/tao_rtdetr_train.yaml")
        print(f"[tao]  wrote {classmap_path} "
              f"({len(ordered)} foreground classes: {', '.join(ordered)})")
        print(f"[tao]  num_classes={COCO_NUM_CLASSES} "
              f"(max category_id {max(eval_ids)} + 1; 0 reserved for "
              f"background)")
        print(f"[tao]  eval_class_ids={eval_ids} "
              f"(TAO defaults to [1], which would evaluate "
              f"'{ordered[0]}' only)")
        print(f"[tao]  augmentation.multi_scales capped at 704: the default "
              f"list runs to 800, upscaling {FINAL_W}px frames")

    try:
        from pycocotools.coco import COCO
        for split, path in coco_paths.items():
            c = COCO(path)
            ann_ids = c.getAnnIds()
            anns = c.loadAnns(ann_ids)
            bad = [a for a in anns
                   if a["bbox"][2] <= 0 or a["bbox"][3] <= 0]
            areas = sorted(a["area"] for a in anns)
            n_img = len(c.getImgIds())
            n_empty = sum(1 for i in c.getImgIds() if not c.getAnnIds(imgIds=i))
            print(f"[coco] {split} validated: {n_img} images "
                  f"({n_empty} negatives), {len(anns)} boxes, "
                  f"{len(bad)} degenerate")
            if areas:
                print(f"[coco] {split} box area px^2: "
                      f"min={areas[0]:.0f} median={areas[len(areas)//2]:.0f} "
                      f"max={areas[-1]:.0f}")
            for cat in c.loadCats(c.getCatIds()):
                n = len(c.getAnnIds(catIds=[cat["id"]]))
                print(f"[coco]   category {cat['id']} {cat['name']}: {n}")
    except ImportError:
        print("[coco] pycocotools not installed - skipping validation "
              "(pip install pycocotools to enable)")
    except Exception as exc:
        print(f"[coco] [WARN] validation failed: {exc}")

if WRITE_YOLO:
    with open(f"{OUTPUT_DIR}/data.yaml", "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("names:\n")
        for name, idx in sorted(CLASS_IDS.items(), key=lambda kv: kv[1]):
            f.write(f"  {idx}: {name}\n")
    print(f"[yolo] wrote {OUTPUT_DIR}/data.yaml")

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
        "reference_luminance": {"mean": 202.6, "std": 44.1, "p1": 52.7},
        "per_frame": lum_log,
    },
    "assets": {c: [t["name"] for t in templates[c]] for c in CLASS_IDS},
    "total_boxes": class_counts,
    "ruler_cube": RULER_CUBE,
}
with open(f"{OUTPUT_DIR}/dataset_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

with open(f"{OUTPUT_DIR}/classes.txt", "w") as f:
    for c, i in sorted(CLASS_IDS.items(), key=lambda kv: kv[1]):
        f.write(f"{c}\n")

if lum_log:
    harsh_log = [r for r in lum_log if r["regime"] == "harsh"]
    over_log = [r for r in lum_log if r["regime"] == "overcast"]
    mm = float(np.mean([r["mean"] for r in lum_log]))
    ms = float(np.mean([r["std"] for r in lum_log]))
    mp = float(np.mean([r["p1"] for r in lum_log]))
    print("=" * 74)
    print(f"LUMINANCE  mean={mm:.1f} std={ms:.1f} p1={mp:.1f}  "
          f"(exposure={EXPOSURE_MODE})")
    print(f"REFERENCE  mean=202.6 std=44.1 p1=52.7  "
          f"shadow ratio p1/mean=0.26")
    for name, log in (("harsh", harsh_log), ("overcast", over_log)):
        if not log:
            continue
        m = float(np.mean([r["mean"] for r in log]))
        s = float(np.mean([r["std"] for r in log]))
        p = float(np.mean([r["p1"] for r in log]))
        print(f"  {name:<9s} mean={m:.1f} std={s:.1f} p1={p:.1f} "
              f"p1/mean={p/max(m,1e-6):.2f}  n={len(log)}")

    if harsh_log:
        hm = float(np.mean([r["mean"] for r in harsh_log]))
        hs = float(np.mean([r["std"] for r in harsh_log]))
        hp = float(np.mean([r["p1"] for r in harsh_log]))
        hd = float(np.mean([r["deep_shadow_pct"] for r in harsh_log]))
        hc = float(np.mean([r["clipped_black_pct"] for r in harsh_log]))
        print(f"  harsh shadows: deep={hd:.1f}% (ref 2.3%)  "
              f"crushed-black={hc:.2f}% (ref 0.00%)")
        if hc > 0.20:
            print("  [HINT] shadows clipping to pure black -> RAISE "
                  "dome_intensity / fill_intensity in LIGHT_HARSH. Real "
                  "shadows keep visible soil texture inside them.")
        elif hd > 5.0:
            print("  [HINT] too much deep shadow -> raise dome_intensity a "
                  "little, or narrow sun_zenith toward 0 (higher sun).")
        elif hd < 1.0 and hs < 35.0:
            print("  [HINT] barely any shadow -> LOWER dome_intensity / "
                  "fill_intensity, or widen sun_zenith (lower sun).")
        if hs < 35.0 and hc <= 0.20:
            print("  [HINT] harsh-regime contrast below reference -> lower "
                  "dome_intensity / fill_intensity in LIGHT_HARSH.")
        if hp / max(hm, 1e-6) > 0.40:
            print("  [HINT] harsh-regime shadows too milky (reference ratio "
                  "is 0.26) -> same fix.")
    if EXPOSURE_MODE == "auto" and abs(mm - 202.6) > 40.0:
        print("  [NOTE] mean differs from the reference, but auto-exposure "
              "sets it. Switch EXPOSURE_MODE to 'manual' to control it.")

    gp = float(np.mean([r["green_pct"] for r in lum_log]))
    ss = float(np.nanmean([r["soil_sat_pct"] for r in lum_log]))
    sr = float(np.nanmean([r["soil_rgb"][0] for r in lum_log]))
    sg = float(np.nanmean([r["soil_rgb"][1] for r in lum_log]))
    sb = float(np.nanmean([r["soil_rgb"][2] for r in lum_log]))
    print(f"  green cover  {gp:.2f}%  (real reference 8.00%, range 0.9-41.0%)")
    print(f"  soil RGB     {sr:.0f},{sg:.0f},{sb:.0f}  sat {ss:.1f}%  "
          f"(real 157,151,135 / 15.4%)")
    if gp < 4.0:
        print("  [HINT] sparse vegetation -> raise WEEDS_PER_FRAME, or lower "
              "CROP_MISSING_PROB.")
    if ss > 20.0:
        print("  [HINT] soil still too saturated (orange) -> pick SOIL_TINTS "
              "entries that attenuate RED more than BLUE.")
    if ss < 9.0:
        print("  [HINT] soil is nearly grey -> ease off the red attenuation "
              "in SOIL_TINTS.")

print("=" * 74)
print("TOTALS: " + ", ".join(f"{c}={class_counts[c]}" for c in CLASS_IDS))
print(f"regimes: {regime_counts}")
print(f"wrote {OUTPUT_DIR}/dataset_meta.json")
print("=" * 74)

rep.orchestrator.wait_until_complete()
print("Done!")
simulation_app.close()
