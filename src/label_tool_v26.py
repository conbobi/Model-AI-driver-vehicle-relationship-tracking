import cv2
import json
import os
import glob
import re

# =================================================================================
# 1. CẤU HÌNH HỆ THỐNG
# =================================================================================
DATA_DIR = "data"
CLIPS_DIR = os.path.join(DATA_DIR, "event_clips")
ANN_DIR = os.path.join(DATA_DIR, "ann")
CAM_IDS = ["cam01"]
OUTPUT_FILE = "annotations.jsonl"
REVIEW_MODE = False
UNKNOWN_TERMS_LOG = "unknown_terms.log"
CUSTOM_TRANSLATIONS_FILE = "custom_translations.json"

# =================================================================================
# 2. TỪ ĐIỂN LOCAL (ƯU TIÊN TRA CỨU)
# =================================================================================
LOCAL_DICT = {
    # Màu sắc cơ bản
    "trắng": "white", "đen": "black", "đỏ": "red", "xanh": "blue",
    "xanh dương": "blue", "xanh nước biển": "blue", "xanh da trời": "sky blue",
    "xanh lá": "green", "xanh lục": "green", "xanh bộ đội": "army green",
    "xanh rêu": "moss green", "xanh ngọc": "turquoise", "xanh lơ": "cyan",
    "vàng": "yellow", "vàng chanh": "lime yellow", "vàng đồng": "gold", "vàng cát": "beige",
    "bạc": "silver", "ghi": "grey", "xám": "grey", "lông chuột": "dark grey",
    "nâu": "brown", "nâu đất": "earth brown", "cà phê": "coffee",
    "cam": "orange", "hồng": "pink", "tím": "purple", "tím than": "dark violet",
    "kem": "cream", "be": "beige", "sữa": "milky white",
    "nhiều màu": "multicolor", "kẻ ca rô": "plaid", "sọc": "striped",
    "kẻ ca ro": "plaid", "kẻ caro": "plaid",
    # Loại xe
    "xe máy": "motorcycle", "xe số": "motorcycle", "tay ga": "scooter",
    "xe đạp": "bicycle", "xe điện": "electric bike", "xe con": "car",
    "xe hơi": "car", "ô tô": "car", "taxi": "taxi", "xe khách": "coach",
    "xe buýt": "bus", "xe 16 chỗ": "van", "xe 29 chỗ": "minibus",
    "bán tải": "pickup truck", "xe tải": "truck", "xe container": "container truck",
    "xe cẩu": "crane truck", "xe đầu kéo": "tractor head", "xe ben": "dump truck",
    # Trạng thái nón
    "không đội nón bảo hiểm": "no helmet", "không đội mũ bảo hiểm": "no helmet",
    "nón bảo hiểm trắng": "a white helmet", "nón bảo hiểm đỏ": "a red helmet",
    "nón bảo hiểm đen": "a black helmet", "nón bảo hiểm xanh": "a blue helmet",
    "nón bảo hiểm vàng": "a yellow helmet", "nón bảo hiểm hồng": "a pink helmet",
    "nón lá": "conical leaf hat",
    # Mô tả thùng xe
    "thùng kín": "enclosed box", "thùng lửng": "open bed", "thùng rỗng": "empty box",
    "thùng đông lạnh": "refrigerated box", "thùng xốp": "styrofoam box",
    "thùng bạt xanh": "blue tarpaulin box", "thùng bạt đen": "black tarpaulin box",
    "thùng khung": "cage box", "cẩu vàng": "yellow crane", "cẩu đỏ": "red crane",
    "cẩu xanh": "blue crane", "cẩu trắng": "white crane", "cẩu đen": "black crane",
    # Áo
    "áo trắng": "white shirt", "áo đen": "black shirt", "áo đỏ": "red shirt",
    "áo xanh": "blue shirt", "áo caro": "plaid shirt", "áo kẻ caro": "plaid shirt",
    # Khác
    "người": "person", "tài xế": "driver", "chở hàng": "carrying goods",
    "chở hàng cồng kềnh": "carrying bulky goods", "chở 1 người": "carrying a passenger",
    "chở 2 người": "carrying two passengers", "đeo balo": "wearing a backpack",
}

# Map cứng cho các slot nghiệp vụ (không bao giờ dùng translator)
CAMERA_DIR_MAP = {
    "đi từ xa về gần camera": "moving towards the camera",
    "đi từ gần ra xa camera": "moving away from the camera",
}

TRAFFIC_ACTION_MAP = {
    "đi thẳng": "going straight",
    "rẽ trái": "turning left",
    "rẽ phải": "turning right",
    "đi thẳng rồi rẽ trái": "going straight and then turning left",
    "đi thẳng rồi rẽ phải": "going straight and then turning right",
    "quay đầu": "making a u-turn",
}

ROAD_STATUS_MAP = {
    "ngược chiều": "against traffic",
    "trên vỉa hè": "on the sidewalk",
    "đi lên vỉa hè": "onto the sidewalk",
    "vượt đèn đỏ": "running a red light",
}

CARGO_MAP = {
    "chở 1 người": "carrying a passenger",
    "chở 2 người": "carrying two passengers",
    "chở 3 người": "carrying three passengers",
    "chở hàng": "carrying goods",
    "chở hàng cồng kềnh": "carrying bulky goods",
    "đeo balo": "wearing a backpack",
}

# =================================================================================
# 3. QUẢN LÝ TỪ ĐIỂN MỞ RỘNG (CUSTOM TRANSLATIONS)
# =================================================================================
def translate_cargo(vi_text: str) -> str:
    """Dịch cụm 'chở X người' thành 'carrying X passenger(s)', các trường hợp khác dùng CARGO_MAP."""
    if not vi_text:
        return ""
    # Xử lý "chở X người"
    match = re.match(r'^chở\s+(\d+)\s+người$', vi_text.strip())
    if match:
        num = int(match.group(1))
        if num == 1:
            return "carrying 1 passenger"
        else:
            return f"carrying {num} passengers"
    # Các trường hợp còn lại dùng từ điển tĩnh
    return CARGO_MAP.get(vi_text, "")

def load_custom_translations():
    if os.path.exists(CUSTOM_TRANSLATIONS_FILE):
        try:
            with open(CUSTOM_TRANSLATIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_custom_translations(data):
    with open(CUSTOM_TRANSLATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

CUSTOM_DICT = load_custom_translations()

def ask_manual_translation(vi_text: str) -> str:
    print(f"\n⚠️ Không tìm thấy bản dịch cho: '{vi_text}'")
    print("Nhập cụm tiếng Anh thay thế để giữ đúng cấu trúc câu.")
    print("Ví dụ:")
    print("  - xanh tím than -> navy blue")
    print("  - áo phản quang -> reflective shirt")
    print("  - chở 3 người -> carrying three passengers")
    en_text = input(">> English replacement (Enter để bỏ qua): ").strip()
    return en_text

# =================================================================================
# 4. HÀM TIỆN ÍCH (SỬA normalize_vi, THÊM NORMALIZER CHUYÊN DỤNG)
# =================================================================================
def normalize_vi(text: str) -> str:
    """Chuẩn hóa tiếng Việt: lower, strip, thay thế các biến thể thông dụng.
    KHÔNG xóa 'đi' ở đầu."""
    if not text:
        return ""
    t = text.lower().strip()
    t = t.replace("quẹo phải", "rẽ phải").replace("quẹo trái", "rẽ trái")
    t = t.replace("nón bh", "nón bảo hiểm").replace("mũ bh", "mũ bảo hiểm")
    t = t.replace("xe ga", "xe tay ga")
    # Không xóa 'đi ' nữa
    return " ".join(t.split())

def normalize_camera_direction(text: str) -> str:
    """Chuẩn hóa hướng camera về 1 trong 2 giá trị chuẩn (có 'đi')."""
    if not text:
        return ""
    t = normalize_vi(text)
    if t in ["lại gần camera", "tới gần camera", "từ xa lại gần", "từ xa về gần camera"]:
        return "đi từ xa về gần camera"
    if t in ["ra xa camera", "đi ra xa camera"]:
        return "đi từ gần ra xa camera"
    if t in ["đi từ xa về gần camera", "đi từ gần ra xa camera"]:
        return t
    return text

def normalize_traffic_action(text: str) -> str:
    """Chuẩn hóa hành vi giao thông về các giá trị trong TRAFFIC_ACTION_MAP."""
    if not text or text == "không có":
        return ""
    t = normalize_vi(text)
    valid_actions = ["đi thẳng", "rẽ trái", "rẽ phải", "đi thẳng rồi rẽ trái", "đi thẳng rồi rẽ phải", "quay đầu"]
    if t in valid_actions:
        return t
    return text

def normalize_road_status(text: str) -> str:
    """Chuẩn hóa trạng thái đường / vi phạm."""
    if not text or text == "không có":
        return ""
    t = normalize_vi(text)
    valid_status = ["ngược chiều", "trên vỉa hè", "đi lên vỉa hè", "vượt đèn đỏ"]
    if t in valid_status:
        return t
    return text

def translate_slot(text: str) -> str:
    """Dịch 1 slot (cụm từ) sang tiếng Anh.
    Ưu tiên local dict, sau đó custom dict, nếu không có thì hỏi người dùng nhập thay thế."""
    if not text:
        return ""
    normalized = normalize_vi(text)
    if normalized in LOCAL_DICT:
        return LOCAL_DICT[normalized]
    if normalized in CUSTOM_DICT:
        return CUSTOM_DICT[normalized]
    replacement = ask_manual_translation(normalized)
    if replacement:
        CUSTOM_DICT[normalized] = replacement
        save_custom_translations(CUSTOM_DICT)
        with open(UNKNOWN_TERMS_LOG, "a", encoding="utf-8") as f:
            f.write(f"MANUAL: {normalized} -> {replacement}\n")
        return replacement
    with open(UNKNOWN_TERMS_LOG, "a", encoding="utf-8") as f:
        f.write(f"UNKNOWN: {normalized}\n")
    return ""

def remove_punctuation(text):
    cleaned = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE)
    cleaned = cleaned.replace('_', '')
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def append_jsonl(filepath, data):
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def get_done_clips(filepath):
    if not os.path.exists(filepath):
        return set()
    done = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                done.add(d['clip_id'])
            except:
                pass
    return done

def choose_or_type(prompt, options):
    print(prompt)
    for i, opt in enumerate(options, 1):
        print(f"  {i}. {opt}")
    print(f"  {len(options)+1}. Nhập tay (mô tả tự do)")
    while True:
        try:
            choice = input(">> Chọn (số): ").strip()
            if not choice:
                continue
            idx = int(choice)
            if 1 <= idx <= len(options):
                return options[idx-1]
            elif idx == len(options)+1:
                return input("   Nhập mô tả: ").strip()
            else:
                print("Số không hợp lệ, chọn lại.")
        except ValueError:
            return choice

def build_action_from_fields(camera_dir, traffic_action, road_status, cargo_passenger):
    """Ghép các thành phần thành câu hành động hoàn chỉnh (tiếng Việt)"""
    parts = []
    # camera_dir đã chuẩn hóa (vd: "đi từ xa về gần camera")
    if camera_dir:
        parts.append(camera_dir)
    if traffic_action and traffic_action != "đi thẳng" and "rồi" not in traffic_action:
        # Nếu có cả camera_dir và traffic_action, thêm "và"
        if camera_dir:
            parts.append(f"và {traffic_action}")
        else:
            parts.append(traffic_action)
    elif traffic_action == "đi thẳng" or "rồi" in traffic_action:
        # "đi thẳng" hoặc "đi thẳng rồi rẽ..." thì không cần "và"
        parts.append(traffic_action)
    if road_status:
        parts.append(road_status)
    if cargo_passenger:
        parts.append(cargo_passenger)
    return " ".join(parts)

# =================================================================================
# 5. CÁC HÀM RENDER TEMPLATE TIẾNG ANH (KHÔNG DÙNG TRANSLATOR)
# =================================================================================
def render_motorbike_caption(attrs):
    """Tạo câu tiếng Anh cho xe máy theo đúng template"""
    shirt = attrs.get('shirt', '')
    helmet = attrs.get('helmet', '')
    bike_type = attrs.get('bike_type', 'xe máy')
    bike_color = attrs.get('bike_color', '')
    camera_dir = attrs.get('camera_direction', '')
    traffic_action = attrs.get('traffic_action', '')
    road_status = attrs.get('road_status', '')
    cargo = attrs.get('cargo_passenger', '')

    shirt_en = translate_slot(shirt)
    helmet_en = translate_slot(helmet)
    bike_type_en = translate_slot(bike_type)
    bike_color_en = translate_slot(bike_color)

    # Dùng trực tiếp giá trị đã chuẩn hóa
    camera_en = CAMERA_DIR_MAP.get(camera_dir, '')
    traffic_en = TRAFFIC_ACTION_MAP.get(traffic_action, '')
    road_en = ROAD_STATUS_MAP.get(road_status, '')
    cargo_en = translate_cargo(cargo)

    # Mô tả người
    wear_parts = []
    if shirt_en:
        if shirt_en.endswith("shirt"):
            wear_parts.append(shirt_en)
        else:
            wear_parts.append(f"{shirt_en} shirt")
    if helmet_en:
        if helmet_en == "no helmet":
            wear_parts.append("no helmet")
        elif helmet_en.startswith("a ") or helmet_en.startswith("an "):
            wear_parts.append(helmet_en)
        else:
            wear_parts.append(f"a {helmet_en}")

    if wear_parts:
        if "no helmet" in wear_parts and len(wear_parts) == 1:
            subject = "A person without a helmet"
        elif "no helmet" in wear_parts:
            other_parts = [p for p in wear_parts if p != "no helmet"]
            subject = f"A person wearing {' and '.join(other_parts)} and without a helmet"
        else:
            subject = f"A person wearing {' and '.join(wear_parts)}"
    else:
        subject = "A person"

    # Mô tả xe
    if bike_color_en and bike_type_en:
        vehicle = f"a {bike_color_en} {bike_type_en}"
    elif bike_color_en:
        vehicle = f"a {bike_color_en} motorcycle"
    elif bike_type_en:
        vehicle = f"a {bike_type_en}"
    else:
        vehicle = "a motorcycle"

    sentence = f"{subject} is riding {vehicle}"
    if road_en:
        sentence += f" {road_en}"
    extras = [x for x in [camera_en, traffic_en, cargo_en] if x]
    if extras:
        sentence += ", " + " and ".join(extras)
    return sentence + "."

def render_car_caption(attrs):
    """Tạo câu tiếng Anh cho xe con / khách"""
    ctype = attrs.get('type', 'xe con')
    color = attrs.get('color', '')
    camera_dir = attrs.get('camera_direction', '')
    traffic_action = attrs.get('traffic_action', '')
    road_status = attrs.get('road_status', '')

    ctype_en = translate_slot(ctype)
    color_en = translate_slot(color)
    camera_en = CAMERA_DIR_MAP.get(camera_dir, '')
    traffic_en = TRAFFIC_ACTION_MAP.get(traffic_action, '')
    road_en = ROAD_STATUS_MAP.get(road_status, '')

    if not ctype_en:
        ctype_en = "car"
    if color_en:
        vehicle_desc = f"A {color_en} {ctype_en}"
    else:
        vehicle_desc = f"A {ctype_en}"

    sentence = f"{vehicle_desc} is driving"
    if road_en:
        sentence += f" {road_en}"
    extras = [x for x in [camera_en, traffic_en] if x]
    if extras:
        sentence += ", " + " and ".join(extras)
    return sentence + "."

def render_truck_caption(attrs):
    """Tạo câu tiếng Anh cho xe tải / container (noun phrase, không "is")"""
    ctype = attrs.get('type', 'xe tải')
    head_col = attrs.get('head_color', '')
    box_val = attrs.get('box_val', '')
    camera_dir = attrs.get('camera_direction', '')
    traffic_action = attrs.get('traffic_action', '')
    road_status = attrs.get('road_status', '')

    ctype_en = translate_slot(ctype)
    head_en = translate_slot(head_col)
    box_en = translate_slot(box_val)
    camera_en = CAMERA_DIR_MAP.get(camera_dir, '')
    traffic_en = TRAFFIC_ACTION_MAP.get(traffic_action, '')
    road_en = ROAD_STATUS_MAP.get(road_status, '')

    if not ctype_en:
        ctype_en = "truck"

    noun_parts = []
    if head_en:
        noun_parts.append(f"{head_en} cabin {ctype_en}")
    else:
        noun_parts.append(ctype_en)
    if box_en:
        if "crane" in box_en:
            noun_parts.append(f"with {box_en}")
        else:
            if "box" not in box_en and "tarpaulin" not in box_en and "bed" not in box_en:
                noun_parts.append(f"with {box_en} box")
            else:
                noun_parts.append(f"with {box_en}")
    caption = " ".join(noun_parts)

    actions = [x for x in [camera_en, traffic_en, road_en] if x]
    if actions:
        caption += " " + " and ".join(actions)

    caption = caption.strip()
    if caption:
        caption = caption[0].upper() + caption[1:]
    if not caption.endswith('.'):
        caption += '.'
    return caption
def generate_queries_from_fields(group, attrs):
    """Tạo cặp câu tiếng Việt và tiếng Anh theo template, không dùng translator"""
    camera_dir = attrs.get('camera_direction', '')
    traffic_action = attrs.get('traffic_action', '')
    road_status = attrs.get('road_status', '')
    cargo = attrs.get('cargo_passenger', '')
    action_vi = attrs.get('action', '')

    # Xây câu tiếng Việt
    if group == "1":   # xe máy
        shirt = attrs.get('shirt', '')
        helmet = attrs.get('helmet', '')
        bike_color = attrs.get('bike_color', '')
        bike_type = attrs.get('bike_type', 'xe máy')
        vi_parts = []
        if shirt:
            vi_parts.append(f"Người mặc áo {shirt}")
        else:
            vi_parts.append("Người")
        if helmet:
            if helmet:
                if "không" in helmet.lower():
                    vi_parts.append("không đội nón bảo hiểm")
                else:
                    # Chỉ thêm "nón bảo hiểm" nếu helmet chưa có từ "nón" hoặc "mũ"
                    if "nón" not in helmet.lower() and "mũ" not in helmet.lower():
                        helmet = f"nón bảo hiểm {helmet}"
                    vi_parts.append(f"đội {helmet}")
        veh_str = f"chạy {bike_type}" if bike_type.lower().startswith("xe") else f"chạy xe {bike_type}"
        if bike_color:
            veh_str += f" màu {bike_color}"
        vi_parts.append(veh_str)
        if action_vi:
            vi_parts.append(action_vi)
        q_vi = " ".join(vi_parts)
        q_vi = remove_punctuation(q_vi)
        q_vi = " ".join(q_vi.split())
        q_en = render_motorbike_caption(attrs)

    elif group == "2":   # xe con / khách
        ctype = attrs.get('type', 'xe con')
        color = attrs.get('color', '')
        if ctype.startswith("xe "):
            vi_parts = [ctype.capitalize()]
        else:
            vi_parts = [f"Xe {ctype}"]
        if color:
            vi_parts.append(f"màu {color}")
        if action_vi:
            vi_parts.append(action_vi)
        q_vi = " ".join(vi_parts)
        q_vi = remove_punctuation(q_vi)
        q_vi = " ".join(q_vi.split())
        q_en = render_car_caption(attrs)

    else:   # group == "3" - xe tải / container / cẩu
        ctype = attrs.get('type', 'xe tải')
        head_col = attrs.get('head_color', '')
        box_val = attrs.get('box_val', '')
        # Xử lý thêm/xóa từ "xe" (giống nhóm 2)
        if ctype.startswith("xe "):
            vi_parts = [ctype.capitalize()]
        else:
            vi_parts = [f"Xe {ctype}"]
        if head_col:
            prefix = "đầu màu"
            if any(w in head_col.lower() for w in ["đầu", "cabin"]):
                prefix = ""
            elif "màu" in head_col.lower():
                prefix = "đầu"
            vi_parts.append(f"{prefix} {head_col}".strip())
        if box_val:
            prefix = "thùng xe"
            box_lower = box_val.lower()
            if "thùng" in box_lower:
                prefix = ""
            elif "cẩu" in box_lower or "cẩu" in ctype.lower():
                prefix = "có"
            elif any(w in box_lower for w in ["khung", "rào", "kín", "lửng", "rỗng", "xốp"]):
                prefix = "thùng"
            elif not any(w in box_lower for w in ["bạt", "màu", "đông lạnh"]):
                prefix = "thùng xe màu"
            vi_parts.append(f"{prefix} {box_val}".strip())
        if action_vi:
            vi_parts.append(action_vi)
        q_vi = " ".join(vi_parts)
        q_vi = remove_punctuation(q_vi)
        q_vi = " ".join(q_vi.split())
        q_en = render_truck_caption(attrs)

    return q_vi, q_en

# =================================================================================
# 6. GÁN NHÃN CHÍNH (MENU SỐ) – BỎ TRANSLATOR
# =================================================================================
def label_clip(cam_id, clip_path, output_path):
    filename = os.path.basename(clip_path)
    clip_id = filename.split(".")[0]
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = round(frame_count / fps, 2) if fps > 0 else 0.0

    print(f"\n🎥 ĐANG XỬ LÝ: {filename} (Time: {total_duration}s)")
    print("="*65)
    print(" CÁC PHÍM TẮT ĐIỀU KHIỂN:")
    print(" [SPACE]: Dừng/Phát    [A]: Gán nhãn đối tượng")
    print(" [N]: Bỏ qua clip      [R]: Lùi lại clip trước    [ESC]: Thoát")
    print("="*65)

    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            display_frame = cv2.resize(frame, (960, 540))

        if paused:
            cv2.putText(display_frame, "PAUSED - PRESS 'A' TO LABEL", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Label Tool - Manual EN Replacement", display_frame)
        key = cv2.waitKey(30) & 0xFF

        if key == 27:
            return "EXIT"
        elif key == ord('n'):
            break
        elif key == ord('r'):
            return "PREV"
        elif key == ord(' '):
            paused = not paused
        elif key == ord('a') and paused:
            print("\n" + "*"*40)
            print("1. XE MÁY, XE ĐẠP (2-Wheelers)")
            print("2. XE CON / KHÁCH")
            print("3. XE TẢI / CÔNG / CẨU")
            print("*"*40)

            while True:
                choice = input(">> Chọn (1/2/3): ").strip()
                if choice in ['1', '2', '3']:
                    break

            print(">> Vẽ Box. ENTER khi xong.")
            bbox = cv2.selectROI("Label Tool - Manual EN Replacement", display_frame, fromCenter=False, showCrosshair=True)
            h_orig, w_orig = frame.shape[:2]
            real_bbox = [int(bbox[0]*(w_orig/960)), int(bbox[1]*(h_orig/540)),
                         int(bbox[2]*(w_orig/960)), int(bbox[3]*(h_orig/540))]

            attrs = {}
            cls_name = "vehicle"

            # ========= NHÓM 1: XE MÁY =========
            if choice == "1":
                cls_name = "motorcyclist"
                shirt_opts = ["trắng", "đen", "caro (kẻ ca rô)", "xanh", "đỏ", "vàng", "khác"]
                attrs['shirt'] = choose_or_type("Màu áo:", shirt_opts)
                helmet_opts = ["không đội nón bảo hiểm", "nón bảo hiểm trắng", "nón bảo hiểm đỏ", "nón bảo hiểm đen", "nón bảo hiểm xanh","nón lá", "nón bảo hiểm khác"]
                attrs['helmet'] = choose_or_type("Nón bảo hiểm:", helmet_opts)
                color_opts = ["đen", "trắng", "đỏ", "xanh", "vàng", "bạc", "khác"]
                attrs['bike_color'] = choose_or_type("Màu xe:", color_opts)
                type_opts = ["xe máy", "xe số", "tay ga", "xe đạp", "xe điện", "khác"]
                attrs['bike_type'] = choose_or_type("Loại xe:", type_opts)

                cam_dir_opts = ["đi từ xa về gần camera", "đi từ gần ra xa camera"]
                cam_dir_raw = choose_or_type("Hướng di chuyển theo camera:", cam_dir_opts)
                attrs['camera_direction'] = normalize_camera_direction(cam_dir_raw)

                action_opts = ["đi thẳng", "rẽ trái", "rẽ phải", "quay đầu", "đi thẳng rồi rẽ trái", "đi thẳng rồi rẽ phải"]
                traffic_action_raw = choose_or_type("Hành vi điều hướng:", action_opts)
                attrs['traffic_action'] = normalize_traffic_action(traffic_action_raw)

                road_opts = ["không có", "ngược chiều", "trên vỉa hè", "đi lên vỉa hè", "vượt đèn đỏ"]
                road_status_raw = choose_or_type("Trạng thái đường/vi phạm:", road_opts)
                attrs['road_status'] = normalize_road_status(road_status_raw)

                cargo_opts = ["không", "chở hàng", "chở hàng cồng kềnh", "chở 1 người", "chở 2 người", "đeo balo","chở ... người (nhập số)"]
                cargo_raw = choose_or_type("Chở hàng / người / phụ kiện:", cargo_opts)
                if cargo_raw == "chở ... người (nhập số)":
                    while True:
                        try:
                            num = int(input("   Nhập số người (ví dụ 3, 4, 5...): "))
                            if num > 0:
                                cargo_raw = f"chở {num} người"
                                break
                            else:
                                print("Số phải lớn hơn 0.")
                        except ValueError:
                            print("Vui lòng nhập số nguyên.")
                else:
                    # giữ nguyên nếu chọn "không" hoặc các option khác
                    if cargo_raw == "không":
                        cargo_raw = ""
                attrs['cargo_passenger'] = cargo_raw if cargo_raw != "không" else ""

                attrs['action'] = build_action_from_fields(
                    attrs.get('camera_direction', ''),
                    attrs.get('traffic_action', ''),
                    attrs.get('road_status', ''),
                    attrs.get('cargo_passenger', '')
                )

            # ========= NHÓM 2: XE CON =========
            elif choice == "2":
                cls_name = "car"
                type_opts = ["con", "khách", "buýt", "tải con", "taxi", "khác"]
                attrs['type'] = choose_or_type("Loại xe:", type_opts)
                color_opts = ["trắng", "đen", "đỏ", "xanh", "bạc", "vàng", "khác"]
                attrs['color'] = choose_or_type("Màu xe:", color_opts)

                cam_dir_opts = ["đi từ xa về gần camera", "đi từ gần ra xa camera"]
                cam_dir_raw = choose_or_type("Hướng di chuyển theo camera:", cam_dir_opts)
                attrs['camera_direction'] = normalize_camera_direction(cam_dir_raw)

                action_opts = ["đi thẳng", "rẽ trái", "rẽ phải", "quay đầu", "đi thẳng rồi rẽ trái", "đi thẳng rồi rẽ phải"]
                traffic_action_raw = choose_or_type("Hành vi điều hướng:", action_opts)
                attrs['traffic_action'] = normalize_traffic_action(traffic_action_raw)

                road_opts = ["không có", "ngược chiều", "trên vỉa hè", "đi lên vỉa hè", "vượt đèn đỏ"]
                road_status_raw = choose_or_type("Trạng thái đường/vi phạm:", road_opts)
                attrs['road_status'] = normalize_road_status(road_status_raw)

                attrs['action'] = build_action_from_fields(
                    attrs.get('camera_direction', ''),
                    attrs.get('traffic_action', ''),
                    attrs.get('road_status', ''),
                    ''
                )

            # ========= NHÓM 3: XE TẢI / CONTAINER / CẨU =========
            elif choice == "3":
                cls_name = "truck"
                type_opts = ["xe tải", "xe container", "xe cẩu", "xe đầu kéo", "xe ben", "khác"]
                attrs['type'] = choose_or_type("Loại xe:", type_opts)
                head_opts = ["trắng", "xanh", "đỏ", "vàng", "đen", "không rõ", "khác"]
                attrs['head_color'] = choose_or_type("Màu đầu xe (cabin):", head_opts)

                if "cẩu" in attrs['type'].lower():
                    crane_opts = ["vàng", "đỏ", "xanh", "trắng", "đen", "khác"]
                    crane_color = choose_or_type("Màu cần cẩu:", crane_opts)
                    attrs['box_val'] = f"cẩu {crane_color}" if crane_color != "khác" else input("Nhập mô tả cẩu: ")
                else:
                    box_opts = ["thùng kín", "thùng bạt xanh", "thùng bạt đen", "thùng lửng", "thùng khung", "khác"]
                    attrs['box_val'] = choose_or_type("Mô tả thùng xe:", box_opts)

                cam_dir_opts = ["đi từ xa về gần camera", "đi từ gần ra xa camera"]
                cam_dir_raw = choose_or_type("Hướng di chuyển theo camera:", cam_dir_opts)
                attrs['camera_direction'] = normalize_camera_direction(cam_dir_raw)

                action_opts = ["đi thẳng", "rẽ trái", "rẽ phải", "quay đầu", "đi thẳng rồi rẽ trái", "đi thẳng rồi rẽ phải"]
                traffic_action_raw = choose_or_type("Hành vi điều hướng:", action_opts)
                attrs['traffic_action'] = normalize_traffic_action(traffic_action_raw)

                road_opts = ["không có", "ngược chiều", "trên vỉa hè", "đi lên vỉa hè", "vượt đèn đỏ"]
                road_status_raw = choose_or_type("Trạng thái đường/vi phạm:", road_opts)
                attrs['road_status'] = normalize_road_status(road_status_raw)

                attrs['action'] = build_action_from_fields(
                    attrs.get('camera_direction', ''),
                    attrs.get('traffic_action', ''),
                    attrs.get('road_status', ''),
                    ''
                )

            # Sinh câu mô tả (không translator)
            q_vi, q_en = generate_queries_from_fields(choice, attrs)
            print("-" * 50)
            print(f"🇻🇳 VI: {q_vi}")
            print(f"🇬🇧 EN: {q_en}")
            print("-" * 50)

            confirm = input(">> ENTER để lưu (no để hủy): ").strip().lower()
            if confirm != 'no':
                record = {
                    "clip_id": clip_id,
                    "image_path": filename,
                    "timestamp": round(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0, 2),
                    "duration": total_duration,
                    "segment": [0.0, total_duration],
                    "bbox": real_bbox,
                    "class_name": cls_name,
                    "attributes": attrs,
                    "query_vi": q_vi,
                    "query_en": q_en
                }
                append_jsonl(output_path, record)
                print("✅ ĐÃ LƯU!")

            cont = input("\n>> Tiếp tục clip này? (Enter=Có / n=Qua clip khác): ").strip().lower()
            if cont in ['n', 'no', 'k']:
                break

    cap.release()
    cv2.destroyAllWindows()
    return "NEXT"

# =================================================================================
# 7. MAIN
# =================================================================================
def main():
    global CUSTOM_DICT
    print("=== TOOL GÁN NHÃN - MANUAL ENGLISH REPLACEMENT (HỎI NGƯỜI DÙNG KHI THIẾU TỪ) ===")
    CUSTOM_DICT = load_custom_translations()
    for cam_id in CAM_IDS:
        cam_clip_dir = os.path.join(CLIPS_DIR, cam_id)
        cam_ann_dir = os.path.join(ANN_DIR, cam_id)
        ensure_dir(cam_ann_dir)
        output_path = os.path.join(cam_ann_dir, OUTPUT_FILE)
        done_clips = get_done_clips(output_path)
        clips = sorted(glob.glob(os.path.join(cam_clip_dir, "*.mp4")))

        idx = 0
        force_open_review = False
        while idx < len(clips):
            clip_path = clips[idx]
            clip_id = os.path.basename(clip_path).split(".")[0]
            if not REVIEW_MODE and not force_open_review and clip_id in done_clips:
                idx += 1
                continue
            force_open_review = False
            status = label_clip(cam_id, clip_path, output_path)
            if status == "EXIT":
                return
            elif status == "NEXT":
                idx += 1
            elif status == "PREV":
                idx = max(0, idx - 1)
                force_open_review = True

if __name__ == "__main__":
    main()