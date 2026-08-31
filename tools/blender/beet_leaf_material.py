"""
Blender material setup for sugar beet leaves

Usage:
    blender in.blend --background --python beet_leaf_material.py -- --mode fix
    blender in.blend --background --python beet_leaf_material.py -- --mode replace
"""

import bpy
import sys
import os

TEX_DIR = "/path/to/beet_tex"          
STAGE = "early_leaf"                    

TARGET_BASE_LINEAR = (0.203, 0.313, 0.065, 1.0)   
ROUGHNESS = 0.52          
SPECULAR_LEVEL = 0.30     
SUBSURF_WEIGHT = 0.15     
SUBSURF_RADIUS = (0.09, 0.16, 0.05)   

SATURATION_GAIN = 1.10

VALUE_GAIN = 0.90
HUE_SHIFT = 0.500

MIDRIB_LO = 0.18          
MIDRIB_HI = 0.55          
MIDRIB_PULL = 1.00        

MIDRIB_DARKEN = 0.30

LEAF_KEYWORDS = ("leaf", "leaves", "list", "blad", "foliage",
                 "plant", "beet", "repa")

def find_input(node, *names):
    """Socket names moved between Blender 3.x and 4.x. Try each in turn."""
    for n in names:
        if n in node.inputs:
            return node.inputs[n]
    return None


def is_leaf_material(mat):
    nm = mat.name.lower()
    return any(k in nm for k in LEAF_KEYWORDS)


def get_principled(mat):
    if not mat.use_nodes:
        return None
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            return n
    return None


def set_common(bsdf):
    r = find_input(bsdf, "Roughness")
    if r:
        r.default_value = ROUGHNESS

    s = find_input(bsdf, "Specular IOR Level", "Specular")
    if s:
        s.default_value = SPECULAR_LEVEL

    ss = find_input(bsdf, "Subsurface Weight", "Subsurface")
    if ss:
        ss.default_value = SUBSURF_WEIGHT
    rad = find_input(bsdf, "Subsurface Radius")
    if rad:
        rad.default_value = SUBSURF_RADIUS

    
    m = find_input(bsdf, "Metallic")
    if m:
        m.default_value = 0.0



FIX_TAG = "BEET_MATFIX"


def fix_material(mat):
    bsdf = get_principled(mat)
    if bsdf is None:
        return False
    nt = mat.node_tree
    base = find_input(bsdf, "Base Color")
    if base is None:
        return False

    
    
    for n in nt.nodes:
        if n.label == FIX_TAG:
            print(f"       (already fixed, skipping colour chain)")
            set_common(bsdf)
            return True

    set_common(bsdf)

    if not base.is_linked:
        
        base.default_value = TARGET_BASE_LINEAR
        return True

    src_socket = base.links[0].from_socket
    x, y = bsdf.location.x - 900, bsdf.location.y

    
    
    
    try:
        sep = nt.nodes.new("ShaderNodeSeparateHSV")
    except RuntimeError:
        sep = nt.nodes.new("ShaderNodeSeparateColor")
        sep.mode = 'HSV'
    sep.location = (x, y + 300)
    nt.links.new(src_socket, sep.inputs[0])

    s_out = sep.outputs["S"] if "S" in sep.outputs else sep.outputs[1]
    v_out = sep.outputs["V"] if "V" in sep.outputs else sep.outputs[2]

    inv = nt.nodes.new("ShaderNodeInvert")          
    inv.location = (x + 180, y + 380)
    nt.links.new(s_out, inv.inputs["Color"])

    mul = nt.nodes.new("ShaderNodeMath")            
    mul.operation = 'MULTIPLY'
    mul.location = (x + 360, y + 300)
    nt.links.new(inv.outputs["Color"], mul.inputs[0])
    nt.links.new(v_out, mul.inputs[1])
    
    rng_node = nt.nodes.new("ShaderNodeMapRange")
    rng_node.location = (x + 540, y + 300)
    rng_node.inputs["From Min"].default_value = MIDRIB_LO
    rng_node.inputs["From Max"].default_value = MIDRIB_HI
    rng_node.inputs["To Min"].default_value = 0.0
    rng_node.inputs["To Max"].default_value = 1.0
    rng_node.clamp = True
    nt.links.new(mul.outputs[0], rng_node.inputs["Value"])
    
    gain = nt.nodes.new("ShaderNodeMath")
    gain.operation = 'MULTIPLY'
    gain.inputs[1].default_value = MIDRIB_PULL
    gain.location = (x + 700, y + 300)
    nt.links.new(rng_node.outputs["Result"], gain.inputs[0])
    
    try:
        mix = nt.nodes.new("ShaderNodeMixRGB")
        f_in, a_in, b_in = mix.inputs[0], mix.inputs[1], mix.inputs[2]
        mix_out = mix.outputs[0]
    except RuntimeError:
        mix = nt.nodes.new("ShaderNodeMix")
        mix.data_type = 'RGBA'
        f_in, a_in, b_in = mix.inputs[0], mix.inputs[6], mix.inputs[7]
        mix_out = mix.outputs[2]
    mix.location = (x + 860, y)
    nt.links.new(gain.outputs[0], f_in)
    nt.links.new(src_socket, a_in)
    b_in.default_value = TARGET_BASE_LINEAR
 
    if MIDRIB_DARKEN > 0.0:
        dk_amt = nt.nodes.new("ShaderNodeMath")
        dk_amt.operation = 'MULTIPLY'
        dk_amt.inputs[1].default_value = MIDRIB_DARKEN
        dk_amt.location = (x + 860, y + 300)
        nt.links.new(gain.outputs[0], dk_amt.inputs[0])

        dk_inv = nt.nodes.new("ShaderNodeMath")     
        dk_inv.operation = 'SUBTRACT'
        dk_inv.inputs[0].default_value = 1.0
        dk_inv.location = (x + 1020, y + 300)
        nt.links.new(dk_amt.outputs[0], dk_inv.inputs[1])

        try:
            dk = nt.nodes.new("ShaderNodeMixRGB")
            dk.blend_type = 'MULTIPLY'
            dk_f, dk_a, dk_b = dk.inputs[0], dk.inputs[1], dk.inputs[2]
            dk_out = dk.outputs[0]
        except RuntimeError:
            dk = nt.nodes.new("ShaderNodeMix")
            dk.data_type = 'RGBA'
            dk.blend_type = 'MULTIPLY'
            dk_f, dk_a, dk_b = dk.inputs[0], dk.inputs[6], dk.inputs[7]
            dk_out = dk.outputs[2]
        dk.location = (x + 1180, y)
        dk_f.default_value = 1.0
        nt.links.new(mix_out, dk_a)
        
        grey = nt.nodes.new("ShaderNodeCombineColor") \
            if hasattr(bpy.types, "ShaderNodeCombineColor") else None
        if grey is not None:
            grey.location = (x + 1020, y + 150)
            for k in range(3):
                nt.links.new(dk_inv.outputs[0], grey.inputs[k])
            nt.links.new(grey.outputs[0], dk_b)
        else:
            dk_b.default_value = (1.0, 1.0, 1.0, 1.0)
        mix_out = dk_out

    hsv = nt.nodes.new("ShaderNodeHueSaturation")
    hsv.location = (x + 1040, y)
    hsv.label = FIX_TAG          
    hsv.inputs["Hue"].default_value = HUE_SHIFT
    hsv.inputs["Saturation"].default_value = SATURATION_GAIN
    hsv.inputs["Value"].default_value = VALUE_GAIN
    nt.links.new(mix_out, hsv.inputs["Color"])
    nt.links.new(hsv.outputs["Color"], base)

    return True

def tex_node(nt, path, loc, non_color=False):
    if not os.path.isfile(path):
        print(f"  [WARN] missing texture: {path}")
        return None
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(path, check_existing=True)
    if non_color:
        n.image.colorspace_settings.name = 'Non-Color'
    n.location = loc
    return n

def replace_material(mat):
    bsdf = get_principled(mat)
    if bsdf is None:
        return False
    nt = mat.node_tree
    set_common(bsdf)
    x, y = bsdf.location.x - 800, bsdf.location.y

    alb = tex_node(nt, f"{TEX_DIR}/beet_{STAGE}_albedo.png", (x, y))
    if alb:
        base = find_input(bsdf, "Base Color")
        if base:
            nt.links.new(alb.outputs["Color"], base)

    a = tex_node(nt, f"{TEX_DIR}/beet_{STAGE}_alpha.png", (x, y - 300), True)
    if a:
        al = find_input(bsdf, "Alpha")
        if al:
            nt.links.new(a.outputs["Color"], al)
        mat.blend_method = 'CLIP'

    r = tex_node(nt, f"{TEX_DIR}/beet_{STAGE}_roughness.png", (x, y - 600), True)
    if r:
        rs = find_input(bsdf, "Roughness")
        if rs:
            nt.links.new(r.outputs["Color"], rs)

    nmap = tex_node(nt, f"{TEX_DIR}/beet_{STAGE}_normal.png", (x - 250, y - 900),
                    True)
    if nmap:
        nn = nt.nodes.new("ShaderNodeNormalMap")
        nn.location = (x, y - 900)
        nn.inputs["Strength"].default_value = 0.85
        nt.links.new(nmap.outputs["Color"], nn.inputs["Color"])
        nrm = find_input(bsdf, "Normal")
        if nrm:
            nt.links.new(nn.outputs["Normal"], nrm)

    return True

def bake_materials(mats, out_dir, res=2048, margin=16):
    os.makedirs(out_dir, exist_ok=True)

    scene = bpy.context.scene
    prev_engine = scene.render.engine
    scene.render.engine = 'CYCLES'
    
    if hasattr(scene, "cycles"):
        scene.cycles.samples = 1      
    else:
        print("  [WARN] Cycles properties unavailable;")
    scene.render.bake.use_pass_direct = False
    scene.render.bake.use_pass_indirect = False
    scene.render.bake.margin = margin

    
    users = {}
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        for slot in ob.material_slots:
            if slot.material in mats:
                users.setdefault(slot.material, []).append(ob)

    baked = 0
    view_objs = bpy.context.view_layer.objects
    for mat in mats:
        obs = users.get(mat, [])
        if not obs:
            print(f"  SKIP {mat.name}: no mesh uses it")
            continue
        
        obs = [o for o in obs if o.name in view_objs]
        if not obs:
            print(f"  SKIP {mat.name}: users are not in the view layer "
                  f"(collection excluded?)")
            continue

        ob = obs[0]
        if not ob.data.uv_layers:
            print(f"  SKIP {mat.name}: {ob.name} has no UV map (bake needs one)")
            continue

        nt = mat.node_tree
        img_name = f"{mat.name}_baked"
        img = bpy.data.images.get(img_name)
        if img is None:
            img = bpy.data.images.new(img_name, res, res, alpha=True)
        img.generated_color = (0, 0, 0, 1)

        tex = nt.nodes.new("ShaderNodeTexImage")
        tex.image = img
        tex.location = (0, 600)
        tex.label = "BAKE_TARGET"
        for n in nt.nodes:
            n.select = False
        tex.select = True
        nt.nodes.active = tex

        for o in view_objs:
            try:
                o.select_set(False)
            except RuntimeError:
                pass          

        restore = []
        for o in obs:
            restore.append((o, o.hide_viewport, o.hide_render,
                            o.hide_get() if o.name in view_objs else False))
            o.hide_viewport = False
            o.hide_render = False
            try:
                o.hide_set(False)
            except RuntimeError:
                pass
            o.select_set(True)
        view_objs.active = ob
  
        try:
            if ob.mode != 'OBJECT':
                bpy.ops.object.mode_set(mode='OBJECT')
        except RuntimeError:
            pass

        try:
            bpy.ops.object.bake(type='DIFFUSE', pass_filter={'COLOR'},
                                use_clear=True)
        except Exception as exc:
            print(f"  FAIL {mat.name}: bake error {exc}")
            nt.nodes.remove(tex)
            for o, hv, hr, hg in restore:
                o.hide_viewport, o.hide_render = hv, hr
            continue

        path = os.path.join(out_dir, f"{img_name}.png")
        img.filepath_raw = path
        img.file_format = 'PNG'
        img.save()
        print(f"  BAKED {mat.name} -> {path}  ({len(obs)} object(s))")

        for o, hv, hr, hg in restore:
            o.hide_viewport, o.hide_render = hv, hr

        
        bsdf = get_principled(mat)
        base = find_input(bsdf, "Base Color") if bsdf else None
        if base is not None:
            for lk in list(base.links):
                nt.links.remove(lk)
            nt.links.new(tex.outputs["Color"], base)
            tex.label = FIX_TAG + "_BAKED"
        baked += 1

    scene.render.engine = prev_engine
    print(f"baked {baked}/{len(mats)}")
    if baked:
        print("baking needs non-overlapping UVs")
    return baked


def report_uncovered(mats):
    done = set(mats)
    missed = {}
    for ob in bpy.data.objects:
        if ob.type != 'MESH':
            continue
        for slot in ob.material_slots:
            m = slot.material
            if m is not None and m not in done:
                missed.setdefault(m.name, []).append(ob.name)
    if not missed:
        print("\ncoverage: every mesh material was processed")
        return
    print("\n" + "!" * 70)
    print("UNPROCESSED MATERIALS -- these keep their ORIGINAL appearance:")
    for mname, obnames in sorted(missed.items()):
        print(f"  {mname}   used by: {', '.join(sorted(obnames))}")
    print("If any of those are leaves, re-run with --all-materials,")
    print("or add a matching word to LEAF_KEYWORDS.")
    print("!" * 70)


def verify(mats, mode):
    print("\nverification:")
    for m in mats:
        bsdf = get_principled(m)
        if bsdf is None:
            print(f"  {m.name}: no Principled BSDF")
            continue
        r = find_input(bsdf, "Roughness")
        s = find_input(bsdf, "Specular IOR Level", "Specular")
        ss = find_input(bsdf, "Subsurface Weight", "Subsurface")
        base = find_input(bsdf, "Base Color")
        chain = "none"
        if base is not None and base.is_linked:
            chain = base.links[0].from_node.bl_idname.replace("ShaderNode", "")
        print(f"  {m.name}")
        print(f"      roughness={r.default_value:.2f} " if r else "      roughness=?")
        print(f"      specular={s.default_value:.2f}" if s else "      specular=?")
        print(f"      subsurface={ss.default_value:.2f}" if ss else "      subsurface=?")
        print(f"      Base Color <- {chain}")


def main():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    mode = "fix"
    only_leaf = True
    out_path = None
    inplace = False
    do_bake = False
    bake_dir = None
    for i, a in enumerate(argv):
        if a == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1]
        if a == "--all-materials":
            only_leaf = False
        if a == "--out" and i + 1 < len(argv):
            out_path = argv[i + 1]
        if a == "--inplace":
            inplace = True
        if a == "--bake":
            do_bake = True
        if a == "--bake-dir" and i + 1 < len(argv):
            bake_dir = argv[i + 1]
            do_bake = True

    mats = [m for m in bpy.data.materials
            if m.use_nodes and (not only_leaf or is_leaf_material(m))]

    if not mats:
        print("No matching materials found.")
        print("Materials in file:")
        for m in bpy.data.materials:
            print("   ", m.name)
        print("Either rename leaf materials to contain 'leaf', "
              "or re-run with --all-materials")
        return

    print(f"mode={mode}  materials={len(mats)}")
    ok = 0
    for m in mats:
        done = fix_material(m) if mode == "fix" else replace_material(m)
        print(f"  {'OK  ' if done else 'SKIP'} {m.name}")
        ok += bool(done)
    print(f"done: {ok}/{len(mats)}")
    print(f"roughness={ROUGHNESS}  specular={SPECULAR_LEVEL}  "
          f"subsurface={SUBSURF_WEIGHT}")

    verify(mats, mode)
    report_uncovered(mats)

    print(f"\nrunning Blender {'.'.join(str(v) for v in bpy.app.version)}")
    try:
        eng = bpy.context.scene.render.engine
        avail = [e.bl_idname for e in
                 bpy.types.RenderEngine.__subclasses__()]
        if eng not in ('CYCLES', 'BLENDER_EEVEE', 'BLENDER_EEVEE_NEXT',
                       'BLENDER_WORKBENCH') and eng not in avail:
            print(f"  [WARN] scene render engine '{eng}' is not available in "
                  f"this Blender build -- likely a version mismatch with the "
                  f"file. Re-run with the Blender that authored it.")
    except Exception:
        pass

    if do_bake:
        print("\nbaking graded albedo (required before USD export):")
        try:
            n_baked = bake_materials(mats, bake_dir or os.path.join(
                os.path.dirname(bpy.data.filepath) or ".", "baked_textures"))
        except Exception as exc:
            
            import traceback
            traceback.print_exc()
            print(f"\n[ERROR] baking failed: {exc}")
            print("The material fixes are still intact and will be saved, but")
            print("they will NOT survive USD export until a bake succeeds.")
            n_baked = 0
        if n_baked == 0:
            print("\n[WARN] nothing was baked -- do not export to USD yet.")
    elif mode == "fix":
        print("\n" + "!" * 70)
        print("WARNING: not baked. Blender's USD exporter supports only simple")
        print("node trees (Principled BSDF, Image Texture, UVMap, Separate RGB).")
        print("The colour-repair chain uses SeparateColor/Math/MapRange/Mix/")
        print("HueSaturation, none of which export. Isaac Sim would load the")
        print("ORIGINAL washed-out texture and this whole run would be wasted.")
        print("Re-run with --bake before exporting to USD.")
        print("!" * 70)
    
    src = bpy.data.filepath
    if out_path:
        dest = os.path.abspath(out_path)
    elif inplace:
        dest = src
    else:
        
        root, ext = os.path.splitext(src)
        dest = f"{root}_matfix{ext}"

    if not dest:
        print("\n[ERROR] no filepath to save to. Pass --out /path/file.blend")
        return

    try:
        bpy.ops.wm.save_as_mainfile(filepath=dest)
        print(f"\nSAVED -> {dest}")
        if dest != src:
            print("       (original left untouched; pass --inplace to overwrite)")
    except Exception as exc:
        print(f"\n[ERROR] save failed: {exc}")


if __name__ == "__main__":
    main()
