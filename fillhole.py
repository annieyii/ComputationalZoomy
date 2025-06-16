# 使用 COLMAP 的 text 輸出 (cameras.txt / images.txt)
# 前景 warped1 不更動，背景 warped2 需要補洞，只補補有洞的部份

from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import List
import cv2, numpy as np
from tqdm import tqdm
from scipy.ndimage import binary_dilation

# −−−−−−−−−−−−−−−−−−− Dolly Zoom 相關函數 −−−−−−−−−−−−−−−−−−

def get_intrinsic(focal_mm, w, h, sensor_width_mm=36.0):
    fx = w / sensor_width_mm * focal_mm
    fy = h / sensor_width_mm * focal_mm
    cx, cy = w / 2, h / 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

# 執行復置，將 depth 變為 3D point cloud
def backproject(depth, K):
    h, w = depth.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (i - cx) * depth / fx
    y = (j - cy) * depth / fy
    z = depth
    return np.stack([x, y, z], axis=-1)  # (H,W,3)

# 將 3D points 投影回像素座標 (u,v)
def project(points, K):
    x, y, z = points[..., 0], points[..., 1], points[..., 2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v

# 將圖片根據深度與錯視角 scale 進行曲面 warp
# 只移動 dolly_plane 之後的傷影區域
# 返回被 warp 後的圖片和 mask

def warp_image(img, depth, K, scale, dolly_plane_depth, use_soft_mask=False):
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    
    # 將每個像素根據深度反投影為 3D 空間點 (x, y, z)
    x = (i - cx) * depth / fx
    y = (j - cy) * depth / fy
    z = depth
    points = np.stack([x, y, z], axis=-1)

    # 建立遮罩：只有 dolly_plane 之後（背景）需要變形
    mask = depth > dolly_plane_depth

    # 將 3D 點投影回原始影像空間取得像素座標 (x_img, y_img)
    x_img = fx * points[..., 0] / points[..., 2] + cx
    y_img = fy * points[..., 1] / points[..., 2] + cy

    # 定義影像中心點（用於後續的縮放）
    x_center = w / 2
    y_center = h / 2

    # 對背景區域做 Dolly Zoom 視角變形（以中心為基準縮放）
    x_img[mask] = (x_img[mask] - x_center) / scale + x_center
    y_img[mask] = (y_img[mask] - y_center) / scale + y_center

    # 將變形後的影像座標投影回 3D 空間（保持深度不變）
    points[..., 0] = (x_img - cx) * points[..., 2] / fx
    points[..., 1] = (y_img - cy) * points[..., 2] / fy

    # 再次將 3D 點投影成像素座標 (u, v)
    u = fx * points[..., 0] / points[..., 2] + cx
    v = fy * points[..., 1] / points[..., 2] + cy

    # 建立 remap 所需的座標對應圖
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)

    # 使用 OpenCV 對影像做重新取樣，產生扭曲後的圖像
    warped = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return warped, mask

# --- Camera intrinsics loader (from COLMAP text) ---
def load_colmap_text(scene_dir: Path):
    cam_file = scene_dir / "cameras.txt"
    img_file = scene_dir / "images.txt"
    if not cam_file.exists() or not img_file.exists():
        raise FileNotFoundError("cameras.txt / images.txt not found in", scene_dir)

    with cam_file.open() as f:
        for ln in f:
            if ln.startswith('#') or not ln.strip(): continue
            _, _, w, h, fx, *rest = ln.split()
            w, h, fx = int(w), int(h), float(fx)
            cx = float(rest[0]) if rest else w / 2
            cy = float(rest[1]) if len(rest) > 1 else h / 2
            break
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], np.float32)

    quats, poss, names = [], [], []
    with img_file.open() as f:
        while True:
            hdr = f.readline()
            if not hdr: break
            if hdr.startswith('#') or not hdr.strip(): continue
            parts = hdr.split()
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            img_name = parts[-1]
            quats.append(np.array([qw, qx, qy, qz], np.float32))
            poss.append(np.array([tx, ty, tz], np.float32))
            f.readline()
            names.append(img_name)
    return K, quats, poss, names, w, h

# --- Geometry helpers --｀
def quat_to_rot(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),     1-2*(x**2+z**2), 2*(y*z-x*w)],
        [2*(x*z-y*w),     2*(y*z+x*w),   1-2*(x**2+y**2)]
    ], dtype=np.float32)

def build_proj(K, R, t):
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    P = np.eye(4, dtype=np.float32)
    P[:3, :4] = K @ Rt[:3, :4]
    return P

# --- Weight functions ---
def angle_weight(normal: np.ndarray, view_dir: np.ndarray):
    cos_theta = (normal * view_dir).sum(-1)
    return np.clip(cos_theta, 0, 1)  # shape (H,W)

def distance_weight(depth: np.ndarray, z_plane: float, beta=0.15):
    return np.exp(-beta * np.abs(depth - z_plane))

def jacobian_weight(jacobian: np.ndarray, gamma=1.0):
    return np.exp(-gamma * np.abs(jacobian - 1))  # favor 1 (no stretch)

# --- Normal map estimation ---
def compute_normals(depth: np.ndarray, fx, fy):
    dzdx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    nx = -dzdx / fx
    ny = -dzdy / fy
    nz = np.ones_like(depth)
    n = np.stack([nx, ny, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    return n  # shape (H,W,3)

# def select_best_background(stack_imgs, stack_depths, mvps, K, scale, dolly_plane_depth, W, H):
def select_best_background(stack_imgs, stack_depths, mvps, K, scale, dolly_plane_depth, W, H, frame_index):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    i_grid, j_grid = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([(i_grid - cx) / fx, (j_grid - cy) / fy, np.ones_like(i_grid)], axis=-1)
    view_dir = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)

    best_score = -np.inf
    best_img = None
    best_depth = None
    best_index = -1

    # 建立儲存資料夾
    warped_dir = Path("selected_warped")
    warped_dir.mkdir(exist_ok=True)

    for i, (img, dep, P) in enumerate(zip(stack_imgs, stack_depths, mvps)):
        # warp 後背景
        warped_img, _ = warp_image(img, dep, K, scale, dolly_plane_depth)
        warped_depth, _ = warp_image(dep, dep, K, scale, dolly_plane_depth)

        # normal 與角度一致性分數
        ref_normal = compute_normals(dep, fx, fy)
        angle_sim = angle_weight(ref_normal, view_dir)

        # 只取背景區域來計分
        bg_mask = (dep > dolly_plane_depth)
        score = angle_sim[bg_mask].mean()

        if score > best_score:
            best_score = score
            best_img = warped_img
            best_depth = warped_depth
            best_index = i

    # 儲存所有 warped image（可選擇只存最佳者）
    save_path = warped_dir / f"frame{frame_index:03d}_cam{i:02d}.png"
    cv2.imwrite(str(save_path), cv2.cvtColor(best_img, cv2.COLOR_RGB2BGR))

    return best_img, best_depth, best_index
# --- Main weighted fill ---
# def fill_holes_weighted(target_rgb: np.ndarray, hole_mask: np.ndarray,
#                         ref_depth: np.ndarray, ref_normal: np.ndarray,
#                         mvps: List[np.ndarray], K: np.ndarray,
#                         stack_imgs: np.ndarray, stack_depths: np.ndarray, z_plane: float,
#                         top_k: int = 4 , frame_index=0):
#     H, W = ref_depth.shape
#     fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
#     i, j = np.meshgrid(np.arange(W), np.arange(H))
#     dirs = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], axis=-1)

#     # --- 新增：洞口周圍特徵參考 ---
#     kernel = np.ones((5, 5), np.uint8)
#     dilated = binary_dilation(hole_mask, structure=kernel)
#     rim_mask = dilated & (~hole_mask)
#     rim_color = target_rgb[rim_mask].mean(axis=0) if rim_mask.any() else np.array([0, 0, 0])
#     rim_normal = ref_normal[rim_mask].mean(axis=0) if rim_mask.any() else np.array([0, 0, 1])
#     rim_depth = ref_depth[rim_mask].mean() if rim_mask.any() else z_plane

#     scores = []
#     samples = []
#     with open("weight_log.txt", "a") as log_file:
#         for i, (img, dep, P) in enumerate(tqdm(zip(stack_imgs, stack_depths, mvps), total=len(stack_imgs), desc='weight blend')):
#             uvw = np.einsum('hwc,dc->hwd', np.dstack([dirs, np.ones_like(ref_depth)]), P.T)
#             z = uvw[..., 2]
#             u = (uvw[..., 0] / z).astype(np.float32)
#             v = (uvw[..., 1] / z).astype(np.float32)
#             proj_sample = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

#             view_dir = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
#             mean_ref_normal = ref_normal[hole_mask].mean(axis=0)
#             mean_view_dir = view_dir[hole_mask].mean(axis=0)
#             angle_cosine = np.dot(mean_ref_normal, mean_view_dir) / (
#                 np.linalg.norm(mean_ref_normal) * np.linalg.norm(mean_view_dir) + 1e-6
#             )
#             log_file.write(f"[frame {frame_index:03d}][camera {i:02d}] mean cos(angle): {angle_cosine:.4f} (dot of means)\n")

#             w1 = angle_weight(ref_normal, view_dir)
#             w2 = distance_weight(ref_depth, z_plane)

#             rim_w_color = np.exp(-np.linalg.norm(proj_sample - rim_color, axis=-1) / 20.0)
#             rim_w_normal = np.clip((ref_normal @ rim_normal) / (np.linalg.norm(ref_normal, axis=-1) * np.linalg.norm(rim_normal) + 1e-6), 0, 1)
#             rim_w_depth = np.exp(-np.abs(ref_depth - rim_depth) / 10.0)

#             rim_weight = rim_w_color * rim_w_normal * rim_w_depth
#             total = w1 * w2 * rim_weight

#             scores.append(total)
#             samples.append(proj_sample)

#         scores = np.stack(scores)
#         samples = np.stack(samples)

#         topk_idx = np.argsort(-scores, axis=0)[:top_k]
#         topk_w = np.take_along_axis(scores, topk_idx, axis=0)
#         topk_img = np.take_along_axis(samples, topk_idx[..., None], axis=0)

#         topk_w = np.clip(topk_w, 1e-6, None)
#         topk_w /= np.sum(topk_w, axis=0, keepdims=True)

#         top1_idx = topk_idx[0]
#         out = target_rgb.copy()
#         for idx in np.unique(top1_idx[hole_mask]):
#             mask_i = (top1_idx == idx) & hole_mask
#             out[mask_i] = stack_imgs[idx][mask_i]
#         # 回復 debug 可視化：將 top1 index 所用影像標記出來
#         top1_img = np.zeros_like(target_rgb)
#         for idx in np.unique(top1_idx):
#             mask_i = (top1_idx == idx) & hole_mask  # 只處理洞區域
#             top1_img[mask_i] = stack_imgs[idx][mask_i]
#         debug_dir = Path("top1_sources")
#         debug_dir.mkdir(exist_ok=True)

#         # 找出每張圖片有貢獻的像素位置，畫出黃點
#         unique_sources = np.unique(top1_idx[hole_mask])
#         for idx in unique_sources:
#             positions = np.argwhere((top1_idx == idx) & hole_mask)
#             if positions.size == 0:
#                 continue
#             source_img = stack_imgs[idx].copy()
#             for y, x in positions:
#                 cv2.circle(source_img, (x, y), radius=2, color=(0, 255, 255), thickness=-1)
#             save_path = debug_dir / f"source_{idx:02d}_frame{frame_index:03d}.png"
#             cv2.imwrite(str(save_path), cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR))

#         return out.astype(np.uint8), out.astype(np.uint8), samples, scores

   
# 主程式：轉成動畫、遞擺前景不更動，背景補洞後 composite

def create_composited_zoom(img1_path: str, depth1_path: str,
                           stack_dir: Path, n_frames: int = 30,
                           dolly_plane_depth: float = 20.0,
                           video_name: str = 'zoom_filled.mp4'):

    # ---------- 載入前景、後景、深度 ----------
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    depth1 = np.load(depth1_path)['depth']
    H, W = depth1.shape
    img1 = cv2.resize(img1, (W, H))

    # ---------- 載入 COLMAP 場景 ----------
    K, quats, poss, names, _, _ = load_colmap_text(stack_dir)
    fx = K[0,0]; fy = K[1,1]
    stack_imgs = []
    stack_depths = []
    for nm in names:
        img_path = stack_dir / 'images' / nm
        basename = Path(nm).stem
        depth_path = stack_dir / 'depth_outputs' / f"{basename}.npy" / f"{basename}.npz"

        img = cv2.resize(cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB), (W, H))
        stack_imgs.append(img)

        if depth_path.exists():
            depth = np.load(str(depth_path))['depth']
            depth = cv2.resize(depth, (W, H))
            stack_depths.append(depth)
        else:
            print(f"找不到對應的深度檔案: {depth_path}")
            stack_depths.append(np.zeros((H, W), dtype=np.float32))

    stack_imgs = np.stack(stack_imgs)
    stack_depths = np.stack(stack_depths)
    mvps = [build_proj(K, quat_to_rot(q), t) for q, t in zip(quats, poss)]

    # ---------- 建立影片與暫存資料夾 ----------
    # tmp_dir = Path('zoom_frames')
    tmp_dir = Path('zoom_depth')
    tmp_dir.mkdir(exist_ok=True)
    # filled_dir = Path('filled_results')
    # filled_dir.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(str(video_name), fourcc, 10, (W, H))


    # ---------- 處理每一幀 ----------
    depth = np.linspace(2.0, dolly_plane_depth, n_frames)
    open("weight_log.txt", "w").close()
    for fi, dp in enumerate(tqdm(depth, desc='frames')):
        warped1 = img1.copy()
        hole_mask_raw = (depth1 <= dp)
        warped_hole_mask, _ = warp_image(hole_mask_raw.astype(np.float32), depth1, K, 1, dp)

        warped2, depth2_alt, ref_idx = select_best_background(
            stack_imgs, stack_depths, mvps, K, 1, dp, W, H, fi)
        
        foreground_mask = (depth1 <= dp)
        composed = warped2.copy()
        composed[foreground_mask] = warped1[foreground_mask]

        # 補洞區域：原本是前景，但在變形背景中露出來了
        # hole_mask = (warped_hole_mask > 0.5) & (depth2_alt > dolly_plane_depth)
        # composed[hole_mask]  = [255, 0, 0]  # 將洞區塗成紅色

        frame_path = tmp_dir / f'depth_{dp}.png'
        cv2.imwrite(str(frame_path), cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))
        vw.write(cv2.cvtColor(composed, cv2.COLOR_RGB2BGR))

    vw.release()
    print(f'\n輸出完成：{video_name}（補洞後動畫')
    print(f'{dolly_plane_depth}')
