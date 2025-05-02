import numpy as np
import cv2
import os
from tqdm import tqdm

def get_intrinsic(focal_mm, w, h, sensor_width_mm=36.0):
    # 計算相機內部參數矩陣 K，將世界座標轉換為影像座標
    # fx = w / sensor_width_mm * focal_mm
    fx = w / sensor_width_mm * focal_mm  # 計算水平方向 focal length (pixels)
    fy = h / sensor_width_mm * focal_mm  # 垂直方向 focal length (pixels)
    cx = w / 2  # 主點 x 座標（影像中心）
    cy = h / 2  # 主點 y 座標
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0, 0, 1]], dtype=np.float32)

def backproject(depth, K):
    # 轉換成世界座標系
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    x = (i - cx) * depth / fx
    y = (j - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1)

def project(points, K):
    # 將3D點投影到2D影像平面
    x, y, z = points[...,0], points[...,1], points[...,2]
    fx, fy = K[0,0], K[1,1]
    cx, cy = K[0,2], K[1,2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v

def warp_background(img, depth, K, scale, dolly_plane_depth):
    # 將遠景進行縮放
    h, w = depth.shape
    points = backproject(depth, K)
    mask_far = points[...,2] > dolly_plane_depth # 後景區域

    points_warped = points.copy()
    # points_warped[mask_far] *= scale  # only scale background
    points_warped[mask_far, 0] /= scale 
    points_warped[mask_far, 1] /= scale

    u, v = project(points_warped, K)
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped

def create_zoom_animation(img_path, depth_path, dolly_plane_depth=15, output_dir="zoom_frames", video_name="zoom.mp4"):
    img = cv2.imread(img_path)[:, :, ::-1]
    depth = np.load(depth_path)["depth"]
    h, w = depth.shape
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)

    K = get_intrinsic(focal_mm=16, w=w, h=h)

    os.makedirs(output_dir, exist_ok=True)
    n_frames = 30
    scales = np.linspace(1.0, 1.5, n_frames) # 放大倍率

    for idx, scale in tqdm(enumerate(scales), total=n_frames):
        warped = warp_background(img, depth, K, scale=scale, dolly_plane_depth=dolly_plane_depth)
        out_path = os.path.join(output_dir, f"frame_{idx:03d}.png")
        cv2.imwrite(out_path, cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))

    # create video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_name, fourcc, 10, (w, h))
    for idx in range(n_frames):
        frame = cv2.imread(os.path.join(output_dir, f"frame_{idx:03d}.png"))
        video.write(frame)
    video.release()
    print(f"✅ 動畫已完成，儲存於 {video_name}")

if __name__ == "__main__":
    create_zoom_animation("pic1.png", "pic1.npz", dolly_plane_depth=15)