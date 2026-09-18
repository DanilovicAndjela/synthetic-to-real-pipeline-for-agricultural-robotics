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
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import numpy as np
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

# Target real-world footprint per class, in meters. These drive the pixel
# sizes: from gsd_match.py, real CropOrWeed2 medians at 640 px are
# crop ~34 px, weed ~14 px, i.e. crop is ~2.4x larger than weed.
CLASS_TARGET_FOOTPRINT = {
    "crop": 0.09,   # sugar beet rosette
    "weed": 0.04,   # weeds: smaller, matches the 14 px median
}

# Per-instance random scale multiplier, applied on top of the target above.
CLASS_SCALE_JITTER = {
    "crop": (0.8, 1.3),
    "weed": (0.9, 1.5),
}

# --- scene layout ------------------------------------------------------------
# Crops grow in rows; weeds scatter anywhere (including between rows).
CROP_ROW_SPACING = 0.50         # meters between crop rows
CROP_IN_ROW_SPACING = 0.22      # meters between plants within a row
CROP_ROW_JITTER = 0.04          # meters of positional noise per crop plant
CROPS_PER_FRAME = (3, 8)
WEEDS_PER_FRAME = (3, 8)

SCATTER_X = (-0.4, 0.4)
SCATTER_Y = (-0.4, 0.4)

# --- camera ------------------------------------------------------------------
# Nadir view. GSD = (horiz_aperture / focal_length) * height / IMAGE_W
# With aperture 20.955mm, focal 24mm: ground width = 0.873 * height
FOCAL_LENGTH = 24.0
HORIZ_APERTURE = 20.955
CAM_HEIGHT = (0.8, 1.0)       # meters; see printout for resulting px sizes
CAM_XY_JITTER = 0.08
CAM_TILT = 2.0                  # degrees

GROUND_SIZE = 80.0
MUD_TILES = 1.0 / 2.0
MIN_BOX_PX = 4

# --- first-run verification --------------------------------------------------
RULER_CUBE = True               # 10 cm cube at origin; set False for real runs
RULER_SIZE = 0.10               # meters

SOIL_TINTS = [
    (1.00, 1.00, 1.00),
    (0.85, 0.80, 0.75),
    (1.10, 1.05, 0.95),
    (0.92, 0.88, 0.82),
]

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

stage = omni.usd.get_context().get_stage()


# ------------------------------------------------------- asset discovery ----
def discover_assets(assets_dir):
    """Walk the class-folder tree. Returns {class_name: [usd_paths]}."""
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
    """Group soil textures into PBR sets by filename prefix."""
    if not os.path.isdir(tex_dir):
        return []
    sets = {}
    for f in sorted(glob.glob(os.path.join(tex_dir, "*"))):
        name = os.path.basename(f).lower()
        # prefix = everything before the _diff_/_nor_/_rough_ marker
        for marker, role in (("_diff", "diffuse"), ("_nor", "normal"),
                             ("_rough", "roughness"), ("_spec", "specular"),
                             ("_disp", "displacement")):
            if marker in name:
                prefix = name.split(marker)[0]
                sets.setdefault(prefix, {})[role] = f
                break
    # a usable set needs at least a diffuse; EXR normals are skipped (patchy
    # decode support in the RTX renderer)
    out = []
    for prefix, roles in sorted(sets.items()):
        if "diffuse" not in roles:
            continue
        entry = {"diffuse": roles["diffuse"], "normal": "", "roughness": ""}
        for role in ("normal", "roughness"):
            p = roles.get(role, "")
            if p and not p.lower().endswith(".exr"):
                entry[role] = p
        out.append(entry)
    return out


def bounds_from_points(st, root=None):
    """World-space bounds from raw mesh points, for files with no extents."""
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
    """Open an asset, measure it, return the transform that normalizes it."""
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
    offset = Gf.Vec3d(
        -0.5 * (rmn[0] + rmx[0]) * scale,
        -0.5 * (rmn[1] + rmx[1]) * scale,
        -rmn[2] * scale,
    )

    return {
        "path": usd_path,
        "name": os.path.splitext(os.path.basename(usd_path))[0],
        "scale": scale,
        "offset": offset,
        "y_up": y_up,
        "raw_footprint": raw_fp,
        "raw_height": rmx[2] - rmn[2],
        "final_height": (rmx[2] - rmn[2]) * scale,
        "mpu": src_mpu,
        "how": how,
    }, None


# ---------------------------------------------------------- discovery run ----
assets = discover_assets(ASSETS_DIR)
soil_sets = discover_soil_textures(SOIL_TEX_DIR)

gsd_min = (HORIZ_APERTURE / FOCAL_LENGTH) * CAM_HEIGHT[0] / IMAGE_W
gsd_max = (HORIZ_APERTURE / FOCAL_LENGTH) * CAM_HEIGHT[1] / IMAGE_W

print("=" * 74)
print("ASSET DISCOVERY")
print("=" * 74)

templates = {}          # class -> list of measured asset dicts
for cls, paths in assets.items():
    templates[cls] = []
    print(f"\n[{cls}]  {len(paths)} file(s)  "
          f"target footprint {CLASS_TARGET_FOOTPRINT[cls]:.2f} m")
    if not paths:
        print(f"  (none found under {ASSETS_DIR}/{cls}/)")
        continue
    for p in paths:
        info, err = measure_asset(p, CLASS_TARGET_FOOTPRINT[cls])
        if info is None:
            print(f"  SKIP {os.path.basename(p)}: {err}")
            continue
        px_min = CLASS_TARGET_FOOTPRINT[cls] * CLASS_SCALE_JITTER[cls][0] / gsd_max
        px_max = CLASS_TARGET_FOOTPRINT[cls] * CLASS_SCALE_JITTER[cls][1] / gsd_min
        warn = ""
        if info["raw_footprint"] > 1.0:
            warn = "  [WARN] >1 m raw: multi-variant pack?"
        if abs(info["mpu"] - 1.0) > 1e-6:
            warn += f"  [WARN] metersPerUnit={info['mpu']}"
        print(f"  {info['name']:<28s} raw {info['raw_footprint']:.3f} m "
              f"-> x{info['scale']:.2f}  h={info['final_height']:.3f} m  "
              f"~{px_min:.0f}-{px_max:.0f} px{warn}")
        templates[cls].append(info)

print(f"\n[soil]  {len(soil_sets)} texture set(s) under {SOIL_TEX_DIR}")
for s in soil_sets:
    extras = [k for k in ("normal", "roughness") if s[k]]
    print(f"  {os.path.basename(s['diffuse']):<40s} + {extras}")

print(f"\n[camera]  focal {FOCAL_LENGTH}mm, aperture {HORIZ_APERTURE}mm, "
      f"height {CAM_HEIGHT[0]}-{CAM_HEIGHT[1]} m")
print(f"          GSD {gsd_min*1000:.2f}-{gsd_max*1000:.2f} mm/px at {IMAGE_W} px")
print(f"          ground coverage {gsd_min*IMAGE_W:.2f}-{gsd_max*IMAGE_W:.2f} m")
if RULER_CUBE:
    print(f"          RULER CUBE ON: {RULER_SIZE*100:.0f} cm cube should measure "
          f"{RULER_SIZE/gsd_max:.0f}-{RULER_SIZE/gsd_min:.0f} px")
print("=" * 74)

active_classes = [c for c in CLASS_IDS if templates.get(c)]
assert active_classes, "No assets found. Check ASSETS_DIR and the folder tree."
if "crop" not in active_classes:
    print("\n[WARN] No crop assets: this run produces WEED-ONLY data. Fine for a "
          "pipeline smoke test, but not trainable for CropOrWeed2.\n")


# ------------------------------------------------------------- materials ----
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
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
    shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)

    mtl.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return mtl, shader


def crop_row_positions(n, rng):
    """Generate n positions arranged in rows along Y, jittered."""
    positions = []
    x0 = rng.uniform(SCATTER_X[0], SCATTER_X[0] + CROP_ROW_SPACING)
    row_xs = []
    x = x0
    while x <= SCATTER_X[1]:
        row_xs.append(x)
        x += CROP_ROW_SPACING
    if not row_xs:
        row_xs = [0.0]
    for i in range(n):
        rx = row_xs[i % len(row_xs)]
        step = (i // len(row_xs)) * CROP_IN_ROW_SPACING
        y = SCATTER_Y[0] + (step % (SCATTER_Y[1] - SCATTER_Y[0]))
        positions.append((
            rx + rng.uniform(-CROP_ROW_JITTER, CROP_ROW_JITTER),
            y + rng.uniform(-CROP_ROW_JITTER, CROP_ROW_JITTER),
            0.0,
        ))
    return positions


with rep.new_layer():

    # ------------------------------------------------------------ ground ----
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

    # ------------------------------------------------- ruler cube (optional) --
    if RULER_CUBE:
        rep.create.cube(
            position=(0.0, 0.0, RULER_SIZE / 2.0),
            scale=(RULER_SIZE, RULER_SIZE, RULER_SIZE),
        )
        # deliberately no semantics -> never labeled, just visible for measuring

    # ------------------------------------------- normalized plant templates ----
    # One template prim per asset file. The wrapper Xform carries the
    # corrective transform; the reference sits on a CHILD prim so the
    # referenced asset's own xformOps don't collide with ours.
    template_paths = {}     # class -> [prim path]
    for cls in active_classes:
        template_paths[cls] = []
        for i, info in enumerate(templates[cls]):
            root = f"/World/Templates/{cls}_{i}"
            tmpl = UsdGeom.Xform.Define(stage, root)
            tmpl.AddTranslateOp().Set(info["offset"])
            if info["y_up"]:
                tmpl.AddRotateXOp().Set(90.0)
            s = info["scale"]
            tmpl.AddScaleOp().Set(Gf.Vec3f(s, s, s))

            geo = UsdGeom.Xform.Define(stage, f"{root}/geo")
            geo.GetPrim().GetReferences().AddReference(info["path"])

            UsdGeom.Imageable(tmpl).MakeInvisible()
            template_paths[cls].append(root)

    # ------------------------------------------------------------ camera ----
    camera = rep.create.camera(
        position=(0.0, 0.0, CAM_HEIGHT[0]),
        rotation=(0.0, 0.0, 0.0),
        focal_length=FOCAL_LENGTH,
        horizontal_aperture=HORIZ_APERTURE,
        clipping_range=(0.01, 100.0),
    )
    rp = rep.create.render_product(camera, resolution=(IMAGE_W, IMAGE_H))

    # ------------------------------------------------------------ lights ----
    sun = rep.create.light(
        light_type="Distant",
        intensity=5000,
        rotation=(55, 0, 45),
        color=(1.0, 0.95, 0.8),
    )
    fill = rep.create.light(
        light_type="Distant",
        intensity=1500,
        rotation=(-20, 0, 200),
        color=(0.85, 0.9, 1.0),
    )
    dome = rep.create.light(
        light_type="Dome",
        intensity=400,
        color=(0.42, 0.28, 0.15),
    )

    # ---------------------------------------------------- frame randomizer ----
    with rep.trigger.on_frame():

        # --- weeds: scattered anywhere, including between crop rows ---------
        if "weed" in active_classes:
            weeds = rep.randomizer.instantiate(
                template_paths["weed"],
                size=rep.distribution.uniform(
                    WEEDS_PER_FRAME[0], WEEDS_PER_FRAME[1] + 1),
                mode="scene_instance",
                with_replacements=True,
            )
            with weeds:
                rep.modify.semantics([("class", "weed")])
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (SCATTER_X[0], SCATTER_Y[0], 0.0),
                        (SCATTER_X[1], SCATTER_Y[1], 0.0),
                    ),
                    rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                    scale=rep.distribution.uniform(*CLASS_SCALE_JITTER["weed"]),
                )

        # --- crops: arranged in rows ----------------------------------------
        if "crop" in active_classes:
            crops = rep.randomizer.instantiate(
                template_paths["crop"],
                size=rep.distribution.uniform(
                    CROPS_PER_FRAME[0], CROPS_PER_FRAME[1] + 1),
                mode="reference",
                with_replacements=True,
            )
            with crops:
                rep.modify.semantics([("class", "crop")])
                rep.modify.pose(
                    position=rep.distribution.uniform(
                        (SCATTER_X[0], SCATTER_Y[0], 0.0),
                        (SCATTER_X[1], SCATTER_Y[1], 0.0),
                    ),
                    rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                    scale=rep.distribution.uniform(*CLASS_SCALE_JITTER["crop"]),
                )

        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (-CAM_XY_JITTER, -CAM_XY_JITTER, CAM_HEIGHT[0]),
                    (CAM_XY_JITTER, CAM_XY_JITTER, CAM_HEIGHT[1]),
                ),
                rotation=rep.distribution.uniform(
                    (-CAM_TILT, -CAM_TILT, 0),
                    (CAM_TILT, CAM_TILT, 360),
                ),
            )

        with sun:
            rep.modify.pose(
                rotation=rep.distribution.uniform((35, 0, 0), (75, 0, 360)))
            rep.modify.attribute(
                "intensity", rep.distribution.uniform(3500, 6500))

    # -------------------------------------------------------- annotators ----
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    rgb_annot.attach([rp])
    bbox_annot.attach([rp])


# Warm-up: let textures finish async loading before the frames we keep.
if soil_sets:
    for tex in soil_sets:
        mud_shader.GetInput("diffuse_texture").Set(tex["diffuse"])
        if tex["normal"]:
            mud_shader.GetInput("normalmap_texture").Set(tex["normal"])
        rep.orchestrator.step(rt_subframes=48)
else:
    print("[WARN] no soil textures found - ground will be flat brown.")
    rep.orchestrator.step(rt_subframes=48)


# ------------------------------------------------------------- main loop ----
class_counts = {c: 0 for c in CLASS_IDS}

for frame_idx in range(NUM_IMAGES):
    print(f"Frame {frame_idx + 1}/{NUM_IMAGES}")

    if soil_sets:
        tex = random.choice(soil_sets)
        mud_shader.GetInput("diffuse_texture").Set(tex["diffuse"])
        if tex["normal"]:
            mud_shader.GetInput("normalmap_texture").Set(tex["normal"])
        if tex["roughness"]:
            mud_shader.GetInput("reflectionroughness_texture").Set(tex["roughness"])
            mud_shader.GetInput("reflection_roughness_texture_influence").Set(1.0)
        else:
            mud_shader.GetInput("reflectionroughness_texture").Set("")
            mud_shader.GetInput("reflection_roughness_texture_influence").Set(0.0)

        tint = random.choice(SOIL_TINTS)
        mud_shader.GetInput("diffuse_tint").Set(Gf.Vec3f(*tint))

        jitter = random.uniform(0.85, 1.2)
        mud_shader.GetInput("texture_scale").Set(
            Gf.Vec2f(MUD_TILES * jitter, MUD_TILES * jitter))

    rep.orchestrator.step(rt_subframes=48)

    rgb = rgb_annot.get_data()
    if rgb[:, :, :3].std() < 3.0:
        print(f"  [WARN] frame {frame_idx} looks flat ({rgb.std():.2f} std) — retrying")
        rep.orchestrator.step(rt_subframes=48)
        rgb = rgb_annot.get_data()
    img = Image.fromarray(rgb[:, :, :3])
    img = img.resize((FINAL_W, FINAL_H), Image.LANCZOS) 
    img.save(f"{OUTPUT_DIR}/images/frame_{frame_idx:05d}.png")

    bb = bbox_annot.get_data()
    boxes = bb["data"]
    id_to_labels = bb["info"]["idToLabels"]

    frame_counts = {c: 0 for c in CLASS_IDS}
    lines = []
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

        # clip to image bounds (plants at the frame edge)
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(IMAGE_W), x2), min(float(IMAGE_H), y2)
        w, h = x2 - x1, y2 - y1
        if w < MIN_BOX_PX or h < MIN_BOX_PX:
            continue

        # YOLO format: class_id cx cy w h, all normalized 0-1
        cid = CLASS_IDS[cls_name]
        cx = (x1 + x2) / 2.0 / IMAGE_W
        cy = (y1 + y2) / 2.0 / IMAGE_H
        nw = w / IMAGE_W
        nh = h / IMAGE_H
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        frame_counts[cls_name] += 1
        class_counts[cls_name] += 1

    with open(f"{OUTPUT_DIR}/labels/frame_{frame_idx:05d}.txt", "w") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))

    print("  labels: " + ", ".join(f"{c}={frame_counts[c]}" for c in CLASS_IDS))

# --------------------------------------------------------------- summary ----
meta = {
    "num_images": NUM_IMAGES,
    "image_size": [FINAL_W, FINAL_H],
    "render_size": [IMAGE_W, IMAGE_H],
    "class_ids": CLASS_IDS,
    "class_target_footprint_m": CLASS_TARGET_FOOTPRINT,
    "class_scale_jitter": CLASS_SCALE_JITTER,
    "camera": {
        "focal_length_mm": FOCAL_LENGTH,
        "horizontal_aperture_mm": HORIZ_APERTURE,
        "height_range_m": list(CAM_HEIGHT),
        "tilt_deg": CAM_TILT,
        "gsd_mm_per_px": [gsd_min * 1000, gsd_max * 1000],
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

print("=" * 74)
print("TOTALS: " + ", ".join(f"{c}={class_counts[c]}" for c in CLASS_IDS))
print(f"wrote {OUTPUT_DIR}/dataset_meta.json")
print("=" * 74)

rep.orchestrator.wait_until_complete()
print("Done!")
simulation_app.close()
