from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 416, "height": 416})

import omni.replicator.core as rep
import omni.usd
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
import numpy as np
import glob
import os
import random
from PIL import Image

# ----------------------------------------------------------------- config ----
# Single-plant test build. The plant is normalized in-scene (no wrapper
# script): measured at startup, then rotated/scaled/offset so it stands
# upright at the right size with its base at z = 0.
PROJECT_DIR = "/isaac_sim_project_new_project"          # container mount point
ASSETS_DIR = f"{PROJECT_DIR}/assets"
OUTPUT_DIR = "/isaac_output"
TEX_DIR = f"{ASSETS_DIR}/textures/soil"     # soil textures (optional)

# The plant to test. Points at the .usda; textures/ must sit beside it.
PLANT_USD = f"{ASSETS_DIR}/dandelion_01/dandelion_01_4k.usda"
PLANT_TARGET_FOOTPRINT = 0.25               # meters, largest horizontal extent

NUM_IMAGES = 1
IMAGE_W, IMAGE_H = 416, 416

# --- first-look settings: fixed camera, fixed scale, easy to judge ----------
# Once the render looks right, widen these back out (suggested values in
# the comments) for the real dataset.
PLANT_SCALE = (1.0, 1.0)        # real dataset: (0.7, 1.2)
PLANTS_PER_FRAME = (8, 8)       # real dataset: (8, 25)
CAM_HEIGHT = (2.5, 2.5)         # real dataset: (2.5, 6.0)
CAM_XY_JITTER = 0.0             # real dataset: 0.3
CAM_TILT = 0.0                  # real dataset: 6.0

# GSD = 0.87 * height / 416 px -> at 2.5 m: ~5.2 mm/px
# A 0.25 m plant appears ~48 px across. Big enough to judge textures.
SCATTER_X = (-1.2, 1.2)
SCATTER_Y = (-1.2, 1.2)

GROUND_SIZE = 80.0              # meters
MUD_TILES = 1.0 / 2.0           # world-projected UVs are meters -> 2 m period

MIN_BOX_PX = 4                  # minimum bbox side in pixels

# Soil PBR sets. Any that don't exist on disk are skipped automatically;
# if none exist the ground falls back to a flat brown color.
SOIL_TEXTURE_SETS = [
    {
        "diffuse":   f"{TEX_DIR}/brown_mud_03_diff_4k.jpg",
        "normal":    f"{TEX_DIR}/brown_mud_03_nor_gl_4k.exr",
        "roughness": "",
    },
    {
        "diffuse":   f"{TEX_DIR}/muddy_tracks_diff_4k.jpg",
        "normal":    f"{TEX_DIR}/muddy_tracks_nor_gl_4k.exr",
        "roughness": f"{TEX_DIR}/muddy_tracks_rough_4k.exr",
    },
]
SOIL_TEXTURE_SETS = [t for t in SOIL_TEXTURE_SETS if os.path.exists(t["diffuse"])]

SOIL_TINTS = [
    (1.00, 1.00, 1.00),
    (0.85, 0.80, 0.75),
    (1.10, 1.05, 0.95),
]

os.makedirs(f"{OUTPUT_DIR}/images", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/labels", exist_ok=True)

stage = omni.usd.get_context().get_stage()


# ------------------------------------------------------- bounds utilities ----
def bounds_from_points(st, root=None):
    """World-space bounds from raw mesh points. Works even when the exporter
    authored no extent attributes (BBoxCache returns empty in that case)."""
    xf_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    mn = np.array([np.inf] * 3)
    mx = np.array([-np.inf] * 3)
    found = False
    rng_iter = (Usd.PrimRange(root, Usd.TraverseInstanceProxies(
                    Usd.PrimDefaultPredicate))
                if root is not None else
                Usd.PrimRange.Stage(st, Usd.TraverseInstanceProxies(
                    Usd.PrimDefaultPredicate)))
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
    """Bounds with fallback chain; returns (range, how) or (None, msg)."""
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


# ------------------------------------------------- measure + normalize plant --
assert os.path.exists(PLANT_USD), f"Plant USD not found: {PLANT_USD}"
tex_dir_beside_plant = os.path.join(os.path.dirname(PLANT_USD), "textures")
if not os.path.isdir(tex_dir_beside_plant):
    print(f"[WARN] no textures/ folder beside {os.path.basename(PLANT_USD)} "
          f"- the plant will likely render untextured.")

src = Usd.Stage.Open(PLANT_USD)
assert src is not None, f"Cannot open {PLANT_USD}"

src_up = UsdGeom.GetStageUpAxis(src)
src_mpu = UsdGeom.GetStageMetersPerUnit(src)
rng, how = compute_bounds(src)
assert rng is not None, f"{PLANT_USD}: {how}"

mn, mx = Gf.Vec3d(rng.GetMin()), Gf.Vec3d(rng.GetMax())
n_meshes = sum(1 for p in Usd.PrimRange.Stage(
    src, Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate))
    if p.IsA(UsdGeom.Mesh))

# corrective rotation: Y-up source -> Z-up stage
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
assert raw_fp > 0, "degenerate plant bounds"
plant_scale = PLANT_TARGET_FOOTPRINT / raw_fp
plant_offset = Gf.Vec3d(
    -0.5 * (rmn[0] + rmx[0]) * plant_scale,
    -0.5 * (rmn[1] + rmx[1]) * plant_scale,
    -rmn[2] * plant_scale,
)
final_h = (rmx[2] - rmn[2]) * plant_scale

gsd = 0.87 * CAM_HEIGHT[1] / IMAGE_W
apparent_px = PLANT_TARGET_FOOTPRINT * PLANT_SCALE[0] / gsd

print("=" * 70)
print("Plant asset check")
print("=" * 70)
print(f"  file            {PLANT_USD}")
print(f"  meshes          {n_meshes}   (bounds via {how})")
print(f"  metersPerUnit   {src_mpu}    upAxis  {src_up}")
print(f"  raw bounds      {mx[0]-mn[0]:.3f} x {mx[1]-mn[1]:.3f} x "
      f"{mx[2]-mn[2]:.3f} m")
print(f"  raw footprint   {raw_fp:.3f} m  ->  correction scale x{plant_scale:.3f}")
print(f"  after fixing    footprint {PLANT_TARGET_FOOTPRINT:.3f} m, "
      f"height {final_h:.3f} m, base at z=0"
      f"{', Y-up -> Z-up' if y_up else ''}")
print(f"  apparent size   ~{apparent_px:.0f} px at {CAM_HEIGHT[1]} m camera "
      f"(min plant scale {PLANT_SCALE[0]})")
if raw_fp > 1.0:
    print("  [WARN] raw footprint > 1 m: this file probably contains SEVERAL "
          "plant variants in a row. Each scattered instance would be the "
          "whole row. Export a single plant, or use the variant-splitting "
          "wrapper script.")
if abs(src_mpu - 1.0) > 1e-6:
    print(f"  [WARN] metersPerUnit={src_mpu}: sizes above may be off by "
          f"{1.0/src_mpu:.0f}x.")
if apparent_px < MIN_BOX_PX:
    print(f"  [WARN] ~{apparent_px:.0f} px is below the {MIN_BOX_PX} px label "
          f"filter -> plants would render but get no labels.")
print("=" * 70)


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


with rep.new_layer():

    # ------------------------------------------------------------ ground ----
    # No semantics on the ground -> never appears in the labels.
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

    # ------------------------------------------- normalized plant template ----
    # /World/PlantSource holds the corrective transform; the reference sits on
    # a CHILD prim because a referenced prim brings its own xformOps and the
    # two must not collide. This template is what gets cloned by the scatter.
    tmpl = UsdGeom.Xform.Define(stage, "/World/PlantSource")
    tmpl.AddTranslateOp().Set(plant_offset)
    if y_up:
        tmpl.AddRotateXOp().Set(90.0)
    tmpl.AddScaleOp().Set(Gf.Vec3f(plant_scale, plant_scale, plant_scale))

    geo = UsdGeom.Xform.Define(stage, "/World/PlantSource/geo")
    geo.GetPrim().GetReferences().AddReference(PLANT_USD)

    # keep the template itself out of the renders
    UsdGeom.Imageable(tmpl).MakeInvisible()

    # ------------------------------------------------------------ camera ----
    # Nadir view: in a Z-up stage, identity rotation looks straight down.
    camera = rep.create.camera(
        position=(0.0, 0.0, CAM_HEIGHT[0]),
        rotation=(0.0, 0.0, 0.0),
        focal_length=24.0,
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
        # Clone the normalized template. +1 because the integer uniform's
        # upper bound is effectively exclusive.
        instances = rep.randomizer.instantiate(
            ["/World/PlantSource"],
            size=rep.distribution.uniform(
                PLANTS_PER_FRAME[0], PLANTS_PER_FRAME[1] + 1),
            mode="scene_instance",
            with_replacements=True,
        )
        with instances:
            rep.modify.semantics([("class", "weed")])
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (SCATTER_X[0], SCATTER_Y[0], 0.0),
                    (SCATTER_X[1], SCATTER_Y[1], 0.0),
                ),
                rotation=rep.distribution.uniform((0, 0, 0), (0, 0, 360)),
                scale=rep.distribution.uniform(PLANT_SCALE[0], PLANT_SCALE[1]),
            )

        with camera:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (-CAM_XY_JITTER, -CAM_XY_JITTER, CAM_HEIGHT[0]),
                    ( CAM_XY_JITTER,  CAM_XY_JITTER, CAM_HEIGHT[1]),
                ),
                rotation=rep.distribution.uniform(
                    (-CAM_TILT, -CAM_TILT, 0),
                    ( CAM_TILT,  CAM_TILT, 360),
                ),
            )

        with sun:
            rep.modify.pose(
                rotation=rep.distribution.uniform((40, 0, 0), (70, 0, 360)))
            rep.modify.attribute(
                "intensity", rep.distribution.uniform(3500, 6500))

    # -------------------------------------------------------- annotators ----
    rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
    bbox_annot = rep.AnnotatorRegistry.get_annotator("bounding_box_2d_tight")
    rgb_annot.attach([rp])
    bbox_annot.attach([rp])


# Warm-up: one step per texture set so textures finish async loading before
# the frames we keep. At least one step regardless, for the plant assets.
if SOIL_TEXTURE_SETS:
    for tex in SOIL_TEXTURE_SETS:
        mud_shader.GetInput("diffuse_texture").Set(tex["diffuse"])
        mud_shader.GetInput("normalmap_texture").Set(tex["normal"])
        if tex["roughness"]:
            mud_shader.GetInput("reflectionroughness_texture").Set(
                tex["roughness"])
        rep.orchestrator.step(rt_subframes=16)
else:
    print("[WARN] no soil textures found - ground will be flat brown.")
    rep.orchestrator.step(rt_subframes=16)

# ------------------------------------------------------------- main loop ----
for frame_idx in range(NUM_IMAGES):
    print(f"Frame {frame_idx + 1}/{NUM_IMAGES}")

    if SOIL_TEXTURE_SETS:
        tex = random.choice(SOIL_TEXTURE_SETS)
        mud_shader.GetInput("diffuse_texture").Set(tex["diffuse"])
        mud_shader.GetInput("normalmap_texture").Set(tex["normal"])
        if tex["roughness"]:
            mud_shader.GetInput("reflectionroughness_texture").Set(
                tex["roughness"])
            mud_shader.GetInput(
                "reflection_roughness_texture_influence").Set(1.0)
        else:
            mud_shader.GetInput("reflectionroughness_texture").Set("")
            mud_shader.GetInput(
                "reflection_roughness_texture_influence").Set(0.0)

        tint = random.choice(SOIL_TINTS)
        mud_shader.GetInput("diffuse_tint").Set(Gf.Vec3f(*tint))

        # jitter the tiling so repeats don't land in the same world spots
        jitter = random.uniform(0.85, 1.2)
        mud_shader.GetInput("texture_scale").Set(
            Gf.Vec2f(MUD_TILES * jitter, MUD_TILES * jitter))

    rep.orchestrator.step(rt_subframes=16)

    rgb = rgb_annot.get_data()
    print(f"  Max: {rgb.max()} Mean: {rgb.mean():.1f}")
    Image.fromarray(rgb[:, :, :3]).save(
        f"{OUTPUT_DIR}/images/frame_{frame_idx:05d}.png")

    bb = bbox_annot.get_data()
    boxes = bb["data"]
    id_to_labels = bb["info"]["idToLabels"]

    written = 0
    with open(f"{OUTPUT_DIR}/labels/frame_{frame_idx:05d}.txt", "w") as f:
        for box in boxes:
            sem_id = int(box["semanticId"])
            label = id_to_labels.get(sem_id, id_to_labels.get(str(sem_id), {}))
            if isinstance(label, dict):
                label = label.get("class", "")
            if "weed" not in str(label):
                continue
            x1, y1 = float(box["x_min"]), float(box["y_min"])
            x2, y2 = float(box["x_max"]), float(box["y_max"])
            if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
                continue
            f.write(f"weed 0.0 0 0.0 {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f} "
                    f"0.0 0.0 0.0 0.0 0.0 0.0 0.0\n")
            written += 1

    print(f"  {written} labels")

rep.orchestrator.wait_until_complete()
print("Done!")
simulation_app.close()
