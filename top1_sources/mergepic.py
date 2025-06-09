import os
import cv2
import numpy as np
from collections import defaultdict

input_dir = "./"
output_dir = "top1_merged"
os.makedirs(output_dir, exist_ok=True)

# 收集所有檔案並依照 frame 分組
groups = defaultdict(list)
for fname in sorted(os.listdir(input_dir)):
    if fname.endswith(".png") and "frame" in fname:
        parts = fname.split("frame")
        if len(parts) != 2:
            continue
        frame_id = parts[1][:3]  # 取 '015'
        groups[frame_id].append(os.path.join(input_dir, fname))

# 對每個 frame 群組進行拼接
for frame_id, file_list in groups.items():
    imgs = [cv2.imread(f) for f in file_list]
    imgs = [img for img in imgs if img is not None]
    if len(imgs) == 0:
        continue

    # 確保所有圖片大小一致
    h, w = imgs[0].shape[:2]
    imgs = [cv2.resize(img, (w, h)) for img in imgs]

    # 分成兩排
    half = (len(imgs) + 1) // 2  # 上排分多一張時不影響
    top_row = np.hstack(imgs[:half])
    bottom_row = np.hstack(imgs[half:]) if len(imgs[half:]) > 0 else np.zeros_like(top_row)

    # 對齊寬度（若上下排寬度不同）
    max_width = max(top_row.shape[1], bottom_row.shape[1])
    if top_row.shape[1] < max_width:
        pad_w = max_width - top_row.shape[1]
        top_row = cv2.copyMakeBorder(top_row, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    if bottom_row.shape[1] < max_width:
        pad_w = max_width - bottom_row.shape[1]
        bottom_row = cv2.copyMakeBorder(bottom_row, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=0)

    # 合併兩排
    merged = np.vstack([top_row, bottom_row])
    out_path = os.path.join(output_dir, f"merged_frame{frame_id}.png")
    cv2.imwrite(out_path, merged)
    print(f"Saved: {out_path}")