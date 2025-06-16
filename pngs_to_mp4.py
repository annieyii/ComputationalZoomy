import cv2
import os
from pathlib import Path

image_folder = 'zoom_depth'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort(key=lambda x: float(x.split('_')[1].split('.png')[0]))  # 依檔名數字排序

# 確保 output_video 資料夾存在
output_dir = Path('output_video')
output_dir.mkdir(exist_ok=True)

video_name = output_dir / f'output_{len(images)}.mp4'

# 讀取第一張圖片以取得尺寸
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(str(video_name), fourcc, 1, (width, height))

for image in images:
    img_path = os.path.join(image_folder, image)
    frame = cv2.imread(img_path)
    video.write(frame)

video.release()
print(f"影片已儲存為 {video_name}")