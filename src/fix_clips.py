import os
import subprocess
import glob

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN (Sửa lại cho đúng máy bạn)
# ==============================================================================
# Thư mục chứa các clip bị lỗi (clip đã cắt)
INPUT_DIR = r"data/event_clips/cam02" 

# Thư mục chứa clip đã sửa xong (Sẽ tạo mới)
OUTPUT_DIR = r"data/event_clips_fixed/cam02" 
# ==============================================================================

def fix_all_clips():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Lấy danh sách tất cả file mp4
    clips = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    print(f"📂 Tìm thấy {len(clips)} clip trong: {INPUT_DIR}")
    print(f"🚀 Bắt đầu sửa lỗi và lưu vào: {OUTPUT_DIR}\n")

    count = 0
    for clip_path in clips:
        filename = os.path.basename(clip_path)
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        # Lệnh FFmpeg quan trọng:
        # -c:v libx264: Nén lại hình ảnh (tạo Keyframe mới từ giây 0)
        # -preset ultrafast: Làm cho nhanh (vì clip ngắn)
        # -crf 23: Giữ nguyên chất lượng
        # -pix_fmt yuv420p: Định dạng màu chuẩn nhất cho OpenCV
        cmd = f'ffmpeg -y -i "{clip_path}" -c:v libx264 -preset ultrafast -crf 23 -pix_fmt yuv420p -c:a copy "{save_path}"'
        
        # Chạy lệnh ẩn (không hiện cửa sổ đen)
        result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if result.returncode == 0:
            print(f"✅ [OK] {filename}")
            count += 1
        else:
            print(f"❌ [LỖI] {filename}")

    print(f"\n🎉 ĐÃ HOÀN TẤT! Đã sửa {count}/{len(clips)} clip.")
    print(f"👉 Hãy đổi đường dẫn trong Tool Label sang thư mục mới: {OUTPUT_DIR}")

if __name__ == "__main__":
    fix_all_clips()