import cv2
import os
import json
import numpy as np
import subprocess  # Thư viện chạy lệnh FFmpeg ổn định

# ==============================================================================
# 1. CẤU HÌNH (SỬA LẠI ĐƯỜNG DẪN CỦA BẠN CHO ĐÚNG)
# ==============================================================================

CAMERA_ID = "cam01"  # Tên camera

# Đường dẫn file video gốc (Video dài mà bạn đã chuẩn hóa)
VIDEO_PATH = "data/raw/633_NguyenChiThanh/Ngay11-03-2026/cam01/00000000216000000_norm.mp4"

# Đường dẫn file ROI (Vùng quan tâm)
ROI_FILE = "data/meta/roi.json"

# Thư mục xuất clip cắt ra (Sẽ tự tạo nếu chưa có)
OUT_DIR = f"data/event_clips/{CAMERA_ID}"
MAP_FILE = f"data/event_clips/{CAMERA_ID}/clip_map.jsonl"

# THAM SỐ CẮT CLIP
FPS_SAMPLE = 5          # Chỉ kiểm tra 5 khung hình/giây (để chạy nhanh)
MOTION_THRESH = 0.015   # Ngưỡng chuyển động (1.5% diện tích ROI)

MIN_EVENT_SEC = 2.0     # Clip ngắn hơn 2s thì bỏ qua
MAX_EVENT_SEC = 20.0    # Clip dài tối đa 20s (đủ để thấy hết sự kiện)

PADDING_BEFORE = 1.5    # Lấy dư 1.5 giây TRƯỚC khi xe xuất hiện
PADDING_AFTER = 1.5     # Lấy dư 1.5 giây SAU khi xe đi qua
COOLDOWN_SEC = 2.0      # Nếu xe dừng 2s rồi đi tiếp -> Vẫn tính là 1 clip (không cắt vụn)

# ==============================================================================
# 2. CÁC HÀM XỬ LÝ
# ==============================================================================

def ensure_dir(p):
    """Tạo thư mục nếu chưa tồn tại"""
    os.makedirs(p, exist_ok=True)

def load_roi(camera_id):
    """Đọc file ROI json"""
    if not os.path.exists(ROI_FILE):
        print(f"⚠️ Không tìm thấy file ROI: {ROI_FILE}. Sẽ quét toàn bộ khung hình.")
        return None
    
    with open(ROI_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if camera_id not in data:
        print(f"⚠️ Không có ROI cho {camera_id}. Sẽ quét toàn bộ khung hình.")
        return None

    return data[camera_id]["roi_polygon"]

def polygon_mask(frame, polygon):
    """Tạo mặt nạ từ đa giác ROI"""
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if polygon is None:
        mask.fill(255) # Nếu không có ROI thì lấy hết
    else:
        pts = np.array(polygon, np.int32)
        cv2.fillPoly(mask, [pts], 255)
    return mask

def motion_score(prev_gray, gray, mask):
    """Tính điểm chuyển động"""
    # 1. Trừ nền
    diff = cv2.absdiff(prev_gray, gray)
    diff = cv2.bitwise_and(diff, diff, mask=mask)
    
    # 2. Nhị phân hóa
    _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    
    # 3. Lọc nhiễu (Morphology)
    kernel = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    motion_pixels = np.sum(th > 0)
    roi_pixels = np.sum(mask > 0)

    if roi_pixels == 0: return 0.0
    return motion_pixels / roi_pixels

def cut_clip_ffmpeg(src_video, t0, t1, out_path):
    """
    Cắt clip CHÍNH XÁC + RESET TIMESTAMP + XÓA METADATA GỐC
    """
    cmd = (
        f'ffmpeg -y -ss {t0:.3f} -to {t1:.3f} -i "{src_video}" '
        f'-c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p '  # Re-encode hình ảnh
        f'-c:a copy '
        f'-avoid_negative_ts make_zero '  # Ép thời gian bắt đầu về 0
        f'-map_metadata -1 '              # <--- QUAN TRỌNG: Xóa sạch thông tin file gốc
        f'"{out_path}"'
    )
    
    # Chạy lệnh ngầm
    subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ==============================================================================
# 3. CHƯƠNG TRÌNH CHÍNH
# ==============================================================================

def main():
    ensure_dir(OUT_DIR)
    
    # Xóa file log cũ
    if os.path.exists(MAP_FILE): os.remove(MAP_FILE)

    roi_poly = load_roi(CAMERA_ID)
    if roi_poly: print("✅ Đã load vùng ROI:", roi_poly)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("❌ Không mở được video:", VIDEO_PATH)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"🎥 Video Info: FPS={fps:.2f} | Tổng Frame={total_frames}")

    sample_step = int(fps / FPS_SAMPLE) if FPS_SAMPLE > 0 else 1

    ret, frame = cap.read()
    if not ret: return

    mask = polygon_mask(frame, roi_poly)
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    event_on = False
    event_start = 0.0
    last_motion_time = -10.0
    clip_count = 0

    map_f = open(MAP_FILE, "a", encoding="utf-8")

    print("\n🚀 Đang quét motion và cắt clip (vui lòng đợi)...")

    for i in range(1, total_frames):
        if i % sample_step != 0:
            ret = cap.grab() 
            continue
        
        ret, frame = cap.read()
        if not ret: break

        t = i / fps
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = motion_score(prev_gray, gray, mask)

        # --- LOGIC PHÁT HIỆN SỰ KIỆN ---
        
        if score > MOTION_THRESH:
            last_motion_time = t
            if not event_on:
                event_on = True
                event_start = t
                print(f"   🚗 [Bắt đầu] Motion tại {t:.2f}s (score={score:.4f})")

        elif event_on and (t - last_motion_time > COOLDOWN_SEC):
            event_end = last_motion_time
            dur = event_end - event_start

            if dur >= MIN_EVENT_SEC:
                if dur > MAX_EVENT_SEC: event_end = event_start + MAX_EVENT_SEC
                
                t0 = max(0, event_start - PADDING_BEFORE)
                t1 = min(total_frames/fps, event_end + PADDING_AFTER)

                clip_count += 1
                clip_id = f"e{clip_count:06d}"
                out_path = os.path.join(OUT_DIR, clip_id + ".mp4")

                print(f"   ✂ [CẮT CLIP] {clip_id}: {t0:.2f}s -> {t1:.2f}s (Dài: {t1-t0:.2f}s)")
                
                cut_clip_ffmpeg(VIDEO_PATH, t0, t1, out_path)

                rec = {
                    "camera_id": CAMERA_ID,
                    "clip_id": clip_id,
                    "src_video": VIDEO_PATH.replace("\\", "/"),
                    "src_t0": round(t0, 3),
                    "src_t1": round(t1, 3),
                    "duration": round(t1-t0, 3)
                }
                map_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                map_f.flush()

            else:
                print(f"   ⚠️ [Bỏ qua] Event quá ngắn ({dur:.2f}s)")

            event_on = False

        prev_gray = gray

    map_f.close()
    cap.release()

    print("\n===================================")
    print(f"✅ HOÀN TẤT! Đã cắt {clip_count} clips.")
    print(f"📂 Kiểm tra folder: {OUT_DIR}")
    print("===================================")

if __name__ == "__main__":
    main()