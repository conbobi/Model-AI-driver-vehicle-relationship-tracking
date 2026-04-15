import cv2
import json
import os
import numpy as np  # <--- THÊM DÒNG NÀY

# ==============================
# CONFIG
# ==============================

# Dùng r"..." để tránh lỗi gạch chéo trong Windows
# Hãy sửa lại đúng tên file video bạn đang có trong máy
VIDEO_PATH = "data/raw/633_NguyenChiThanh/Ngay11-03-2026/cam01/00000000202000000_norm.mp4"
CAMERA_ID = "cam01"

OUT_FILE = "data/meta/roi.json"

# ==============================
# UTILS
# ==============================

points = []

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_roi(camera_id, frame_w, frame_h, polygon):
    """Save ROI polygon into roi.json"""
    
    ensure_dir("data/meta")

    # Load existing roi.json if exists
    if os.path.exists(OUT_FILE):
        with open(OUT_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    else:
        data = {}

    # Update camera ROI
    data[camera_id] = {
        "frame_w": frame_w,
        "frame_h": frame_h,
        "roi_polygon": polygon
    }

    # Write back
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    print(f"\n✅ ROI saved for {camera_id} into {OUT_FILE}")

# ==============================
# MOUSE CALLBACK
# ==============================

def mouse_click(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point added: ({x},{y})")

    # (Tùy chọn) Chuột phải để xóa điểm cuối nếu lỡ tay
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            print("Removed last point")

# ==============================
# MAIN
# ==============================

def main():
    global points

    print(f"Đang đọc video từ: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print("❌ Cannot open video. Kiểm tra lại đường dẫn!")
        return

    # Read first frame
    ret, frame = cap.read()
    cap.release() # Đóng video ngay sau khi lấy được 1 frame

    if not ret:
        print("❌ Cannot read frame")
        return

    h, w, _ = frame.shape

    print("\n===================================")
    print("📌 ROI DRAW TOOL")
    print("===================================")
    print("Left click  : Thêm điểm")
    print("Right click : Xóa điểm vừa vẽ")
    print("S           : Lưu ROI")
    print("R           : Reset vẽ lại từ đầu")
    print("Q           : Thoát")
    print("===================================\n")

    cv2.namedWindow("Draw ROI")
    cv2.setMouseCallback("Draw ROI", mouse_click)

    while True:
        temp = frame.copy()

        # Vẽ các điểm (chấm tròn)
        for p in points:
            cv2.circle(temp, tuple(p), 4, (0, 0, 255), -1)

        # Vẽ đường nối các điểm
        if len(points) > 1:
            # Chuyển points sang numpy array để vẽ
            pts_array = np.array(points, np.int32)
            pts_array = pts_array.reshape((-1, 1, 2))
            
            # isClosed = True để tự nối điểm cuối về điểm đầu tạo thành vòng kín
            cv2.polylines(temp, [pts_array], isClosed=True, color=(0, 255, 255), thickness=2)

        # Show
        cv2.imshow("Draw ROI", temp)

        key = cv2.waitKey(20) & 0xFF

        # Quit
        if key == ord("q"):
            print("Exit.")
            break
        
        # Reset
        elif key == ord("r"):
            points = []
            print("ROI reset.")

        # Save
        elif key == ord("s"):
            if len(points) < 3:
                print("❌ Cần ít nhất 3 điểm để tạo thành đa giác!")
                continue
            
            save_roi(CAMERA_ID, w, h, points)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()