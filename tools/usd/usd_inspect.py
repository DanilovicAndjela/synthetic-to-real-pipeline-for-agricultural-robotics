#!/usr/bin/env python3

"""Inspect USD plant assets for geometry, materials, textures, and basic scale issues

Usage:
    python usd_inspect.py path/to/asset.usd 
"""

import os
import sys

try:
    from pxr import Usd, UsdGeom, UsdShade
except ImportError:
    sys.exit("usd-core not installed. Run:  pip install usd-core")

MAX_PLANT_HEIGHT_M = 1.0 # anything taller than this is suspicious for a weed
MIN_PLANT_HEIGHT_M = 0.01 # anything shorter than 1 cm is suspicious 
PIVOT_TOL_M = 0.02  # bbox bottom should be within 2 cm of the origin

USD_EXT = (".usd", ".usda", ".usdc", ".usdz")


def collect_files(args):
    files = []
    for a in args:
        if os.path.isdir(a):
            for root, _dirs, names in os.walk(a):
                for n in sorted(names):
                    if n.lower().endswith(USD_EXT):
                        files.append(os.path.join(root, n))
        elif a.lower().endswith(USD_EXT):
            files.append(a)
        else:
            print(f"skipping (not a USD file): {a}")
    return files


def axis_index(up_axis_token):
    return {"X": 0, "Y": 1, "Z": 2}.get(str(up_axis_token), 2)


def fmt_m(v):
    if abs(v) < 1.0:
        return f"{v * 100:.1f} cm"
    return f"{v:.3f} m"


def texture_paths_of_material(material):
    for prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(prim)
        if not shader:
            continue
        shader_id = shader.GetIdAttr().Get() if shader.GetIdAttr() else None
        for inp in shader.GetInputs():
            try:
                val = inp.Get()
            except Exception:
                continue
            if val is not None and hasattr(val, "path") and hasattr(val, "resolvedPath"):
                yield (prim.GetName(), str(shader_id), val.path, val.resolvedPath)


def shader_ids_of_material(material):
    ids = set()
    for prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(prim)
        if shader and shader.GetIdAttr():
            sid = shader.GetIdAttr().Get()
            if sid:
                ids.add(str(sid))
    return ids


def inspect(path):
    print("=" * 78)
    print(f"ASSET: {path}")
    print("=" * 78)
    findings = []  # (level, message)

    stage = Usd.Stage.Open(path)
    if stage is None:
        print("  FAIL: could not open stage")
        return ["FAIL"]
-
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    default_prim = stage.GetDefaultPrim()
    print(f"  metersPerUnit : {mpu}   (1 stage unit = {mpu} m)")
    print(f"  upAxis        : {up}")
    print(f"  defaultPrim   : {default_prim.GetPath() if default_prim else 'NOT SET'}")

    if not default_prim:
        findings.append(("WARN", "no defaultPrim set — referencing this file in "
                                 "Isaac Sim/Replicator may fail or pick the wrong root"))
    if mpu == 0.01:
        findings.append(("INFO", "stage is in CENTIMETERS (typical raw Blender export); "
                                 "fine if the bbox below is plausible, but be consistent "
                                 "across all assets"))

    # bbox
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    root = default_prim if default_prim else stage.GetPseudoRoot()
    bbox = cache.ComputeWorldBound(root)
    rng = bbox.ComputeAlignedRange()
    if rng.IsEmpty():
        print("  FAIL: empty bounding box (no imageable geometry found)")
        return ["FAIL"]

    mn, mx = rng.GetMin(), rng.GetMax()
    size = [(mx[i] - mn[i]) * mpu for i in range(3)]  #in meters
    ui = axis_index(up)
    height_m = size[ui]
    footprint = [size[i] for i in range(3) if i != ui]

    print(f"  world bbox min: ({mn[0]:.3f}, {mn[1]:.3f}, {mn[2]:.3f}) [stage units]")
    print(f"  world bbox max: ({mx[0]:.3f}, {mx[1]:.3f}, {mx[2]:.3f}) [stage units]")
    print(f"  size (meters) : X={fmt_m(size[0])}  Y={fmt_m(size[1])}  Z={fmt_m(size[2])}")
    print(f"  plant height  : {fmt_m(height_m)} (along up axis {up})")
    print(f"  footprint     : {fmt_m(footprint[0])} x {fmt_m(footprint[1])}")

    if height_m > MAX_PLANT_HEIGHT_M:
        findings.append(("FAIL", f"plant is {fmt_m(height_m)} tall — scale is almost "
                                 f"certainly wrong for a field weed"))
    elif height_m < MIN_PLANT_HEIGHT_M:
        findings.append(("FAIL", f"plant is only {fmt_m(height_m)} tall — scale or "
                                 f"units are probably wrong"))

    # origin check
    bottom_offset_m = mn[ui] * mpu
    print(f"  bbox bottom vs origin (soil-line check): {fmt_m(bottom_offset_m)}")
    if abs(bottom_offset_m) > PIVOT_TOL_M:
        findings.append(("WARN", f"origin is {fmt_m(abs(bottom_offset_m))} away from the "
                                 f"plant base — scattering in Replicator will bury or "
                                 f"float this plant; fix the pivot in Blender"))

    # meshes
    n_mesh = n_pts = n_faces = 0
    single_sided = []
    for prim in stage.Traverse():
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            continue
        n_mesh += 1
        pts = mesh.GetPointsAttr().Get()
        fvc = mesh.GetFaceVertexCountsAttr().Get()
        n_pts += len(pts) if pts else 0
        n_faces += len(fvc) if fvc else 0
        ds = mesh.GetDoubleSidedAttr().Get()
        if not ds:
            single_sided.append(prim.GetPath().pathString)

    print(f"  meshes        : {n_mesh}   points: {n_pts}   faces: {n_faces}")
    if n_faces > 500_000:
        findings.append(("WARN", f"{n_faces} faces — heavy for an asset you will "
                                 f"scatter dozens of times per scene; consider decimating"))
    if single_sided:
        findings.append(("WARN", f"{len(single_sided)}/{n_mesh} meshes are single-sided "
                                 f"(doubleSided=False) — leaves may vanish at grazing "
                                 f"angles; first few: {single_sided[:3]}"))

    # materials & textures
    layer_dir = os.path.dirname(os.path.abspath(path))
    bound_mats = {}
    unbound = []
    for prim in stage.Traverse():
        if not UsdGeom.Mesh(prim):
            continue
        mat, _rel = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat and mat.GetPrim().IsValid():
            bound_mats[mat.GetPath().pathString] = mat
        else:
            unbound.append(prim.GetPath().pathString)

    print(f"  materials     : {len(bound_mats)} bound"
          + (f", {len(unbound)} meshes WITHOUT material" if unbound else ""))
    if unbound:
        findings.append(("FAIL", f"meshes with no material binding (will render grey): "
                                 f"{unbound[:3]}{' ...' if len(unbound) > 3 else ''}"))

    any_texture = False
    for mpath, mat in bound_mats.items():
        ids = shader_ids_of_material(mat)
        print(f"    - {mpath}")
        print(f"        shaders: {sorted(ids) if ids else 'none found'}")
        known_ok = any(s.startswith("UsdPreviewSurface") or s.startswith("Usd")
                       for s in ids)
        if ids and not known_ok:
            findings.append(("WARN", f"material {mpath} uses non-UsdPreviewSurface "
                                     f"shaders {sorted(ids)} — check it actually "
                                     f"renders in Isaac Sim"))
        for shader_name, sid, apath, resolved in texture_paths_of_material(mat):
            any_texture = True
            exists = bool(resolved) and os.path.exists(resolved)
            if not exists and apath:
                cand = os.path.join(layer_dir, apath)
                exists = os.path.exists(cand)
            status = "OK" if exists else "MISSING"
            print(f"        texture [{status}]: {apath}")
            if not exists:
                findings.append(("FAIL", f"texture not found on disk: {apath} "
                                         f"(shader {shader_name})"))

    if bound_mats and not any_texture:
        findings.append(("WARN", "no texture files referenced by any material — if this "
                                 "asset relied on Blender procedural shaders, they did "
                                 "NOT survive export; bake textures in Blender"))

    # summary
    print("  " + "-" * 74)
    if not findings:
        print("  RESULT: PASS — no issues found")
    else:
        for level, msg in findings:
            print(f"  {level}: {msg}")
    print()
    return [lvl for lvl, _ in findings]


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    files = collect_files(sys.argv[1:])
    if not files:
        sys.exit("no USD files found")
    worst = 0
    for f in files:
        levels = inspect(f)
        if "FAIL" in levels:
            worst = 1
    sys.exit(worst)


if __name__ == "__main__":
    main()
