#!/usr/bin/env python3

"""
Repair missing UsdPreviewSurface material networks in USD assets
Usage: python usd_fix_material.py asset.usdc
"""

import argparse
import glob
import os
import sys

try:
    from pxr import Usd, UsdGeom, UsdShade, Sdf
except ImportError:
    sys.exit("usd-core not installed.")

USD_EXT = (".usd", ".usda", ".usdc", ".usdz")
IMG_EXT = (".png", ".jpg", ".jpeg", ".tga", ".tif", ".tiff", ".bmp", ".exr", ".hdr")

ROLE_KEYS = {
    "diffuse": ("diff", "albedo", "basecolor", "base_color", "_col", "color"),
    "alpha": ("alpha", "opacity", "_mask"),
    "roughness": ("rough",),
    "normal": ("nor_gl", "normal", "_nor", "_nrm"),
    "metallic": ("metal",),
}

WARN_EXR = True

def find_tex_dir(asset_path, override=None):
    if override:
        return os.path.abspath(override)
    base = os.path.dirname(os.path.abspath(asset_path))
    for name in ("textures", "Textures", "tex", "maps", "Maps"):
        cand = os.path.join(base, name)
        if os.path.isdir(cand):
            return cand
    return base


def list_textures(tex_dir):
    out = []
    for f in sorted(glob.glob(os.path.join(tex_dir, "*"))):
        if f.lower().endswith(IMG_EXT):
            out.append(f)
    return out


def pick(files, keys, prefer_non_exr=True):
    matches = [f for f in files if any(k in os.path.basename(f).lower() for k in keys)]
    if not matches:
        return None
    if prefer_non_exr:
        non_exr = [f for f in matches if not f.lower().endswith(".exr")]
        if non_exr:
            return non_exr[0]
    return matches[0]


def detect_uv_primvar(stage, default="st"):
    for prim in stage.Traverse():
        if not UsdGeom.Mesh(prim):
            continue
        for pv in UsdGeom.PrimvarsAPI(prim).GetPrimvars():
            if "float2" in str(pv.GetTypeName()):
                return pv.GetPrimvarName()
    return default


def material_has_shaders(material):
    for prim in Usd.PrimRange(material.GetPrim()):
        shader = UsdShade.Shader(prim)
        if shader and shader.GetIdAttr() and shader.GetIdAttr().Get():
            return True
    return False


def textures_for_material(all_tex, mat_name):
    mn = mat_name.lower()
    subset = [f for f in all_tex if mn in os.path.basename(f).lower()]
    return subset if subset else all_tex


def build_network(stage, mat, tex_files, asset_dir, uv_name, verbose=True):
    mat_path = mat.GetPath().pathString

    diffuse = pick(tex_files, ROLE_KEYS["diffuse"])
    alpha = pick(tex_files, ROLE_KEYS["alpha"])
    rough = pick(tex_files, ROLE_KEYS["roughness"])
    normal = pick(tex_files, ROLE_KEYS["normal"])
    metal = pick(tex_files, ROLE_KEYS["metallic"])

    if not diffuse:
        print(f"    SKIP {mat_path}: no diffuse texture found")
        return False

    def rel(p):
        return os.path.relpath(p, asset_dir)

    if verbose:
        for role, p in (("diffuse", diffuse), ("alpha", alpha),
                        ("roughness", rough), ("normal", normal), ("metallic", metal)):
            if p:
                flag = "  <-- EXR, convert to PNG" if (WARN_EXR and p.lower().endswith(".exr")) else ""
                print(f"    {role:9s}: {rel(p)}{flag}")

    pbr = UsdShade.Shader.Define(stage, mat_path + "/PBRShader")
    pbr.CreateIdAttr("UsdPreviewSurface")
    pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

    st = UsdShade.Shader.Define(stage, mat_path + "/stReader")
    st.CreateIdAttr("UsdPrimvarReader_float2")
    st.CreateInput("varname", Sdf.ValueTypeNames.Token).Set(uv_name)
    stout = st.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    def tex(name, path, colorspace, out_type, out_name):
        t = UsdShade.Shader.Define(stage, f"{mat_path}/{name}")
        t.CreateIdAttr("UsdUVTexture")
        t.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(rel(path))
        t.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(stout)
        t.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set(colorspace)
        t.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        t.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        return t, t.CreateOutput(out_name, out_type)

    _, dout = tex("diffuseTex", diffuse, "sRGB", Sdf.ValueTypeNames.Float3, "rgb")
    pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(dout)

    if alpha:
        _, aout = tex("opacityTex", alpha, "raw", Sdf.ValueTypeNames.Float, "r")
    else:
        _, aout = tex("opacityTex", diffuse, "raw", Sdf.ValueTypeNames.Float, "a")
    pbr.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(aout)
    pbr.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.35)

    if rough:
        _, rout = tex("roughTex", rough, "raw", Sdf.ValueTypeNames.Float, "r")
        pbr.GetInput("roughness").ConnectToSource(rout)
    if metal:
        _, mout = tex("metalTex", metal, "raw", Sdf.ValueTypeNames.Float, "r")
        pbr.GetInput("metallic").ConnectToSource(mout)
    if normal:
        nt, nout = tex("normalTex", normal, "raw", Sdf.ValueTypeNames.Normal3f, "rgb")
        nt.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((2, 2, 2, 1))
        nt.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((-1, -1, -1, 0))
        pbr.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(nout)

    return True


def process(asset_path, tex_dir_override=None, force=False, uv_override=None):
    print("=" * 78)
    print(f"ASSET: {asset_path}")
    stage = Usd.Stage.Open(asset_path)
    if stage is None:
        print("  FAIL: cannot open stage")
        return False

    asset_dir = os.path.dirname(os.path.abspath(asset_path))
    tex_dir = find_tex_dir(asset_path, tex_dir_override)
    all_tex = list_textures(tex_dir)
    print(f"  texture dir: {tex_dir}  ({len(all_tex)} images)")
    if not all_tex:
        print("  FAIL: no texture images found — pass --tex-dir")
        return False

    uv_name = uv_override or detect_uv_primvar(stage)
    print(f"  UV primvar : {uv_name}" + ("  (detected)" if not uv_override else "  (forced)"))

    bound = {}
    for prim in stage.Traverse():
        if not UsdGeom.Mesh(prim):
            continue
        mat, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if mat and mat.GetPrim().IsValid():
            bound[mat.GetPath().pathString] = mat
    for prim in stage.Traverse():
        m = UsdShade.Material(prim)
        if m and prim.GetPath().pathString not in bound:
            bound.setdefault(prim.GetPath().pathString, m)

    if not bound:
        print("  FAIL: no materials found in stage")
        return False

    changed = False
    for mpath, mat in sorted(bound.items()):
        name = mat.GetPrim().GetName()
        if material_has_shaders(mat) and not force:
            print(f"  OK   {mpath}: already has shaders, skipping (use --force to rebuild)")
            continue
        print(f"  FIX  {mpath}")
        subset = textures_for_material(all_tex, name)
        if build_network(stage, mat, subset, asset_dir, uv_name):
            changed = True

    mats = list(bound.values())
    if len(mats) == 1:
        only = mats[0]
        for prim in stage.Traverse():
            if UsdGeom.Mesh(prim):
                UsdShade.MaterialBindingAPI(prim).Bind(only)

    if not changed:
        print("  nothing to do")
        return True

    out = os.path.splitext(asset_path)[0] + "_fixed.usda"
    stage.GetRootLayer().Export(out)
    print(f"  wrote {out}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="USD file or directory")
    ap.add_argument("--tex-dir", default=None, help="override texture directory")
    ap.add_argument("--uv-primvar", default=None, help="override UV primvar name (st, UVMap)")
    ap.add_argument("--force", action="store_true", help="rebuild even if shaders exist")
    ap.add_argument("--recursive", action="store_true", help="scan directory recursively")
    args = ap.parse_args()

    targets = []
    if os.path.isdir(args.path):
        walker = os.walk(args.path) if args.recursive else [
            (args.path, [], os.listdir(args.path))]
        for root, _d, names in walker:
            for n in sorted(names):
                if n.lower().endswith(USD_EXT) and not n.endswith("_fixed.usda"):
                    targets.append(os.path.join(root, n))
    else:
        targets.append(args.path)

    if not targets:
        sys.exit("no USD files found")
    for t in targets:
        process(t, args.tex_dir, args.force, args.uv_primvar)


if __name__ == "__main__":
    main()
