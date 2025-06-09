"""cz_ulr_pipeline.py
-------------------------------------------------
*Computational Zoomy × Unstructured-Lumigraph* –
最小可執行示範：

1. 先用 **ULRRenderer** (modernGL) 載入　dataset
2. CZ 端 `warp_image()`　產生的 foreground 仍保留
3. 對遠景洞洞　`hole_mask`　→ 呼叫 `ulr.render()` 取得同視角完整圖
4. 只把洞覆蓋回去 → 完全無霧面

> ⚠️ 這是​ *MVP skeleton*. 你必須先完成 `ulr_renderer.py` 裡
> `upload_images()` 與 `upload_cameras()` (依你自己的 COLMAP　檔案)
> 然後改 `DATASET_DIR` 為你的資料夾。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2
from tqdm import tqdm

from ulr_renderer import ULRRenderer, Camera  # ← 來自 canvas 那支檔案

# ----------------------------------------------------------------------------
#  ↓↓↓↓↓  你的 ULR / COLMAP scene  資料夾  ↓↓↓↓↓
# ----------------------------------------------------------------------------
DATASET_DIR = Path('./ulr/sunglassgirl')      # TODO: 改成你的資料夾
SHADER_DIR  = Path('ulr') 
IMG_EXT     = '.jpg'                    # 若是 .png 就改
PROXY_OBJ   = None                      # 或 Path('proxy.obj')

# ----------------------------------------------------------------------------
#  (A) —— 讀兩張 CZ 原始圖片 + 深度
# ----------------------------------------------------------------------------
IMG1_PATH = Path('./img/raw/pic1.png')
DEP1_PATH = Path('./img/output/pic1.npz')
IMG2_PATH = Path('./img/raw/pic2.png')
DEP2_PATH = Path('./img/output/pic2.npz')

# ----------------------------------------------------------------------------
#  (B) —— ULR 初始化
# ----------------------------------------------------------------------------
print('[INFO] initialising ULRRenderer …')
ulr = ULRRenderer(dataset_dir=DATASET_DIR,
                  shader_dir =SHADER_DIR,
                  width=1024, height=768)

# ---- (B1) 讀取多張影像 -------------------------------------------------------
image_paths: List[Path] = sorted(DATASET_DIR.glob(f'images/*{IMG_EXT}'))
ulr.upload_images(image_paths)
print(f'[INFO] loaded {len(image_paths)} images into 2D-array texture')

# ---- (B2) 讀取 cameras.txt / images.txt → Camera list -----------------------
# 這裡示範「假資料」把位置設原點、MVP=I；請依照你的 COLMAP 解析替換！
print('[WARN] using dummy Camera list – replace with real COLMAP parsing!')
dummy_cam = Camera(position=(0, 0, 0), mvp=np.eye(4, dtype=np.float32))
ulr.upload_cameras([dummy_cam])

# ----------------------------------------------------------------------------
#  (C) —— CZ warp 與洞洞修補
# ----------------------------------------------------------------------------

def get_intrinsic(focal_mm: float, w: int, h: int, sensor_w_mm: float = 36.0):
    fx = w / sensor_w_mm * focal_mm
    fy = h / sensor_w_mm * focal_mm
    cx, cy = w / 2, h / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float32)

def backproject(depth: np.ndarray, K: np.ndarray):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (i - cx) * depth / fx
    y = (j - cy) * depth / fy
    return np.stack([x, y, depth], axis=-1)

def project(points: np.ndarray, K: np.ndarray):
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    return fx * x / z + cx, fy * y / z + cy

def warp_image(img: np.ndarray, depth: np.ndarray, K: np.ndarray, scale: float, d_plane: float):
    h, w = depth.shape
    img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
    pts = backproject(depth, K)
    mask_far = pts[..., 2] > d_plane
    pts[mask_far, 0] /= scale
    pts[mask_far, 1] /= scale
    u, v = project(pts, K)
    warped = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
    return warped, mask_far

def fill_holes_with_ulr(ulr_: ULRRenderer, warped: np.ndarray, hole_mask: np.ndarray,
                        quat: Tuple[float, float, float, float],
                        pos: Tuple[float, float, float], fov: float):
    repaired = ulr_.render(quat, pos, fov)  # (H,W,3)
    out = warped.copy()
    out[hole_mask] = repaired[hole_mask]
    return out

# ---- load raw imgs / depth --------------------------------------------------
img1 = cv2.imread(str(IMG1_PATH))[:, :, ::-1]
img2 = cv2.imread(str(IMG2_PATH))[:, :, ::-1]

depth1 = np.load(DEP1_PATH)['depth']
depth2 = np.load(DEP2_PATH)['depth']
H, W = depth1.shape
img1 = cv2.resize(img1, (W, H))
img2 = cv2.resize(img2, (W, H))
depth2 = cv2.resize(depth2, (W, H))

K = get_intrinsic(16, W, H)

dolly_z = 20.0
n_frames = 30
scales = np.linspace(1.0, 1.5, n_frames)

os.makedirs('zoom_frames', exist_ok=True)
print('[INFO] start rendering frames …')
for idx, sc in tqdm(list(enumerate(scales)), desc='frames'):
    warped1, holes = warp_image(img1, depth1, K, sc, dolly_z)

    # TODO: 替換成真正對應的姿態；這裡先固定為 (I,0)
    quat = (0, 0, 0, 1)
    pos  = (0, 0, 0)
    fov  = 45.0 / sc

    repaired = fill_holes_with_ulr(ulr, warped1, holes, quat, pos, fov)
    cv2.imwrite(f'zoom_frames/frame_{idx:03}.png', cv2.cvtColor(repaired, cv2.COLOR_RGB2BGR))

# ---- export mp4 -------------------------------------------------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
vid    = cv2.VideoWriter('zoom.mp4', fourcc, 10, (W, H))
for idx in range(n_frames):
    frame = cv2.imread(f'zoom_frames/frame_{idx:03}.png')
    vid.write(frame)
vid.release()
print('zoom.mp4 完成！')
