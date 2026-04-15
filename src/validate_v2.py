import os
import json

# ==============================
# CONFIG
# ==============================

DATA_DIR = "data"
ANN_DIR = os.path.join(DATA_DIR, "ann")
CLIP_DIR = os.path.join(DATA_DIR, "event_clips")

REQUIRED_CAM_FILES = ["objects.jsonl", "relations.jsonl", "events.jsonl"]

# ==============================
# HELPERS
# ==============================

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append((i, json.loads(line)))
            except Exception as e:
                print(f"❌ JSON parse error in {path} line {i}: {e}")
    return rows


def check_bbox(bbox):
    """bbox must be [x1,y1,x2,y2]"""
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    return all(isinstance(x, (int, float)) for x in bbox)


# ==============================
# VALIDATION
# ==============================

def validate_camera(cam):

    print("\n" + "=" * 60)
    print(f"📌 VALIDATING CAMERA: {cam}")
    print("=" * 60)

    cam_ann = os.path.join(ANN_DIR, cam)
    cam_clips = os.path.join(CLIP_DIR, cam)

    # --- Check files exist
    for fn in REQUIRED_CAM_FILES:
        fp = os.path.join(cam_ann, fn)
        if not os.path.exists(fp):
            print(f"❌ Missing annotation file: {fp}")
            return

    objects_path = os.path.join(cam_ann, "objects.jsonl")
    relations_path = os.path.join(cam_ann, "relations.jsonl")
    events_path = os.path.join(cam_ann, "events.jsonl")

    # --- Load annotations
    objects = load_jsonl(objects_path)
    relations = load_jsonl(relations_path)
    events = load_jsonl(events_path)

    # --- Build object_id map: clip_id → set(object_id)
    obj_map = {}

    for line_no, row in objects:
        clip_id = row.get("clip_id")
        if not clip_id:
            print(f"[objects line {line_no}] ❌ Missing clip_id")
            continue

        if clip_id not in obj_map:
            obj_map[clip_id] = set()

        for obj in row.get("objects", []):
            oid = obj.get("object_id")
            if not oid:
                print(f"[objects line {line_no}] ❌ Missing object_id")
                continue

            obj_map[clip_id].add(oid)

            # Keyframes check
            kfs = obj.get("keyframes", [])
            if not (1 <= len(kfs) <= 3):
                print(f"[objects line {line_no}] ❌ object {oid} must have 1–3 keyframes")
                continue

            for kf in kfs:
                if "t" not in kf or "bbox" not in kf:
                    print(f"[objects line {line_no}] ❌ keyframe missing t/bbox")
                    continue
                if not check_bbox(kf["bbox"]):
                    print(f"[objects line {line_no}] ❌ invalid bbox format: {kf['bbox']}")

    # --- Validate relations
    for line_no, row in relations:
        clip_id = row.get("clip_id")
        rels = row.get("relations", [])

        for rel in rels:
            subj = rel.get("subject")
            obj = rel.get("object")

            if clip_id not in obj_map:
                print(f"[relations line {line_no}] ❌ clip {clip_id} has no objects.jsonl entry")
                continue

            if subj not in obj_map[clip_id]:
                print(f"[relations line {line_no}] ❌ subject {subj} not found in objects of {clip_id}")

            if obj not in obj_map[clip_id]:
                print(f"[relations line {line_no}] ❌ object {obj} not found in objects of {clip_id}")

    # --- Validate events
    ok = True

    for line_no, row in events:

        clip_id = row.get("clip_id")
        t0 = row.get("t_start")
        t1 = row.get("t_end")

        # Clip file must exist
        clip_path = os.path.join(cam_clips, clip_id + ".mp4")
        if not os.path.exists(clip_path):
            ok = False
            print(f"[events line {line_no}] ❌ Missing clip file: {clip_path}")

        # Time check
        if t0 is None or t1 is None or float(t0) >= float(t1):
            ok = False
            print(f"[events line {line_no}] ❌ Bad time range: {t0} >= {t1}")

        # Query check
        if "query_vi" not in row:
            ok = False
            print(f"[events line {line_no}] ❌ Missing query_vi")

        # English query requirement
        q_en = row.get("queries_en", [])
        if len(q_en) < 1:
            print(f"[events line {line_no}] ⚠️ Need >=3 queries_en (guideline)")

        # Object link check
        obj_ids = row.get("object_ids", [])
        if clip_id in obj_map:
            for oid in obj_ids:
                if oid not in obj_map[clip_id]:
                    ok = False
                    print(f"[events line {line_no}] ❌ object_id {oid} not found in objects.jsonl")

    # --- Summary
    if ok:
        print(f"\n✅ CAMERA {cam}: All core labels OK")
    else:
        print(f"\n❌ CAMERA {cam}: Found errors, fix them!")



# ==============================
# MAIN
# ==============================

def main():
    cams = sorted(os.listdir(ANN_DIR))

    if not cams:
        print("❌ No cameras found in data/ann/")
        return

    for cam in cams:
        validate_camera(cam)


if __name__ == "__main__":
    main()
