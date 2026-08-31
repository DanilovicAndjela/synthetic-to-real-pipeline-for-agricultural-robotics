#!/usr/bin/env python3
"""
Usage:
    python 02_convert_cropandweed.py \\
        --labels-dir  data/real/raw/bboxes/ \\
        --images-dir  data/real/images \\
        --out-dir     data/real/annotations \\
        --crop-ids 1 --weed-ids 2 \\
        --test 250 --val 90
"""
import argparse, csv, json, os, random, re
from collections import defaultdict

CATEGORIES = [
    {"id": 1, "name": "crop", "supercategory": "plant"},
    {"id": 2, "name": "weed", "supercategory": "plant"},
]

SESSION_RE = re.compile(r"^([a-zA-Z]+)-(\d+)-")

def session_of(stem):
    m = SESSION_RE.match(stem)
    return f"{m.group(1)}-{m.group(2)}" if m else stem

def image_size(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size

def read_csv_rows(path):
    with open(path, newline="") as f:
        for row in csv.reader(f):
            row = [c.strip() for c in row if c.strip() != ""]
            if len(row) < 5:
                continue
            try:
                yield [float(row[0]), float(row[1]), float(row[2]),
                       float(row[3]), int(float(row[4]))]
            except ValueError:
                continue  

def build(args):
    label_map = {}
    for i in args.crop_ids:
        label_map[i] = 1
    for i in args.weed_ids:
        label_map[i] = 2

    exts = (".png", ".jpg", ".jpeg", ".JPG", ".PNG")
    images, annotations = [], []
    by_session = defaultdict(list)
    img_id, ann_id = 1, 1
    dropped_label, dropped_degen, no_image = 0, 0, 0

    csvs = sorted(f for f in os.listdir(args.labels_dir) if f.lower().endswith(".csv"))
    print(f"Found {len(csvs)} label files")

    for fn in csvs:
        stem = os.path.splitext(fn)[0]
        img_path = None
        for e in exts:
            p = os.path.join(args.images_dir, stem + e)
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            no_image += 1
            continue

        W, H = image_size(img_path)
        rows = list(read_csv_rows(os.path.join(args.labels_dir, fn)))

        kept = []
        for left, top, right, bottom, lid in rows:
            if lid not in label_map:
                dropped_label += 1
                continue
                
            l, t = max(0.0, left), max(0.0, top)
            r, b = min(float(W), right), min(float(H), bottom)
            w, h = r - l, b - t
            if w < args.min_side or h < args.min_side:
                dropped_degen += 1
                continue
            kept.append({
                "id": ann_id, "image_id": img_id,
                "category_id": label_map[lid],
                "bbox": [round(l, 2), round(t, 2), round(w, 2), round(h, 2)],
                "area": round(w * h, 2),
                "iscrowd": 0,
            })
            ann_id += 1

        if not kept and not args.keep_empty:
            continue

        images.append({"id": img_id, "file_name": os.path.basename(img_path),
                       "width": W, "height": H})
        annotations.extend(kept)
        by_session[session_of(stem)].append(img_id)
        img_id += 1

    print(f"images kept={len(images)} instances={len(annotations)}")
    print(f"dropped: unmapped_label={dropped_label} degenerate={dropped_degen} "
          f"no_matching_image={no_image}")
    print(f"sessions: {len(by_session)}")
    return images, annotations, by_session

def split_sessions(by_session, n_test, n_val, seed):
    rng = random.Random(seed)
    sessions = sorted(by_session, key=lambda s: -len(by_session[s]))
    rng.shuffle(sessions)

    test, val, train, acc = [], [], [], 0
    for s in sessions:
        n = len(by_session[s])
        if acc < n_test:
            test.append(s); acc += n
        else:
            break
    rest = [s for s in sessions if s not in set(test)]
    acc = 0
    for s in rest:
        n = len(by_session[s])
        if acc < n_val:
            val.append(s); acc += n
        else:
            train.append(s)
    return train, val, test

def dump(path, images, annotations, ids):
    ids = set(ids)
    ims = [i for i in images if i["id"] in ids]
    ans = [a for a in annotations if a["image_id"] in ids]
    with open(path, "w") as f:
        json.dump({"images": ims, "annotations": ans, "categories": CATEGORIES}, f)
    per = defaultdict(int)
    for a in ans:
        per[a["category_id"]] += 1
    print(f"  {os.path.basename(path):<28} images={len(ims):<5} "
          f"instances={len(ans):<6} crop={per[1]} weed={per[2]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--crop-ids", type=int, nargs="+", required=True,
                    help="LabelIDs meaning sugar beet, from CropAndWeed datasets.py")
    ap.add_argument("--weed-ids", type=int, nargs="+", required=True)
    ap.add_argument("--test", type=int, default=250)
    ap.add_argument("--val", type=int, default=90)
    ap.add_argument("--min-side", type=float, default=4.0)
    ap.add_argument("--keep-empty", action="store_true")
    ap.add_argument("--subsets", type=int, nargs="*", default=[50, 100, 200, 400],
                    help="N-sweep training subset sizes")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images, annotations, by_session = build(args)

    tr_s, va_s, te_s = split_sessions(by_session, args.test, args.val, args.seed)
    tr = [i for s in tr_s for i in by_session[s]]
    va = [i for s in va_s for i in by_session[s]]
    te = [i for s in te_s for i in by_session[s]]

    print(f"\nsession-disjoint split: train={len(tr)} ({len(tr_s)} sessions)  "
          f"val={len(va)} ({len(va_s)}) test={len(te)} ({len(te_s)})")
    dump(os.path.join(args.out_dir, "real_train_full.json"), images, annotations, tr)
    dump(os.path.join(args.out_dir, "real_val.json"), images, annotations, va)
    dump(os.path.join(args.out_dir, "real_test.json"), images, annotations, te)
    dump(os.path.join(args.out_dir, "real_all.json"), images, annotations,
         [i["id"] for i in images])

    rng = random.Random(args.seed)
    shuffled = tr[:]
    rng.shuffle(shuffled)
    for n in sorted(args.subsets):
        if n > len(shuffled):
            print(f"  skipping N={n}: only {len(shuffled)} train images available")
            continue
        dump(os.path.join(args.out_dir, f"real_train_n{n}.json"),
             images, annotations, shuffled[:n])

if __name__ == "__main__":
    main()
