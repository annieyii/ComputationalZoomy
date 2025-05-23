# hole_filler_weighted.py – CPU 版 ULR-style 補洞器
# ==============================================
# ※ 該檔使用 COLMAP 的 text 輸出 (cameras.txt / images.txt)
# ※ 讓行過 CZ warped 後的前景 warped1 不更動，背景 warped2 需要補洞，只補補有洞的部份

from __future__ import annotations
import argparse, os
from pathlib import Path
from typing import List
import cv2, numpy as np
from tqdm import tqdm

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

def warp_image(img, depth, K, scale, dolly_plane_depth):
    h, w = depth.shape
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    
    x = (i - cx) * depth / fx
    y = (j - cy) * depth / fy
    z = depth

    points = np.stack([x, y, z], axis=-1)
    mask = depth > dolly_plane_depth
    # points[mask, :2] /= scale  # 傷影區域做縮放
    x_img = fx * points[..., 0] / points[..., 2] + cx
    y_img = fy * points[..., 1] / points[..., 2] + cy

    x_center = w / 2
    y_center = h / 2

    x_img[mask] = (x_img[mask] - x_center) / scale + x_center
    y_img[mask] = (y_img[mask] - y_center) / scale + y_center

    # 投影回 3D 空間
    points[..., 0] = (x_img - cx) * points[..., 2] / fx
    points[..., 1] = (y_img - cy) * points[..., 2] / fy

    u = fx * points[..., 0] / points[..., 2] + cx
    v = fy * points[..., 1] / points[..., 2] + cy
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
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

# --- Geometry helpers ---
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

# --- Main weighted fill ---
def fill_holes_weighted(target_rgb: np.ndarray, hole_mask: np.ndarray,
                        ref_depth: np.ndarray, ref_normal: np.ndarray,
                        mvps: List[np.ndarray], K: np.ndarray,
                        stack_imgs: np.ndarray, z_plane: float,
                        top_k: int = 4 , frame_index=0):
    H, W = ref_depth.shape
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]
    i, j = np.meshgrid(np.arange(W), np.arange(H))
    dirs = np.stack([(i - cx) / fx, (j - cy) / fy, np.ones_like(i)], axis=-1)  # shape (H,W,3)

    scores = []
    samples = []
    with open("weight_log.txt", "a") as log_file:
        for i, (img, P) in enumerate(tqdm(zip(stack_imgs, mvps), total=len(stack_imgs), desc='weight blend')):
            uvw = np.einsum('hwc,dc->hwd', np.dstack([dirs, np.ones_like(ref_depth)]), P.T)
            z = uvw[..., 2]
            u = (uvw[..., 0] / z).astype(np.float32)
            v = (uvw[..., 1] / z).astype(np.float32)
            proj_sample = cv2.remap(img, u, v, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

            # weights
            view_dir = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
            
            # mean_ref_normal = ref_normal.mean(axis=(0, 1))  # (3,)
            mean_ref_normal = ref_normal[hole_mask].mean(axis=0)
            # mean_view_dir = view_dir.mean(axis=(0, 1))      # (3,)
            mean_view_dir = view_dir[hole_mask].mean(axis=0)
            angle_cosine = np.dot(mean_ref_normal, mean_view_dir) / (
                np.linalg.norm(mean_ref_normal) * np.linalg.norm(mean_view_dir) + 1e-6
            )
            log_file.write(f"[frame {frame_index:03d}][camera {i:02d}] mean cos(angle): {angle_cosine:.4f} (dot of means)\n")
            
            # 角度權重
            w1 = angle_weight(ref_normal, view_dir)
            # 深度權重
            w2 = distance_weight(ref_depth, z_plane)
            # 理論上這裡要是texture 
            w3 = np.ones_like(w1)

            total = w1 * w2 * w3

            mean_w1 = w1[hole_mask].mean() if hole_mask.any() else 0
            mean_w2 = w2[hole_mask].mean() if hole_mask.any() else 0
            mean_total = total[hole_mask].mean() if hole_mask.any() else 0
            log_file.write(f"[frame {frame_index:03d}][camera {i:02d}] angle_w={mean_w1:.4f}, dist_w={mean_w2:.4f}, total={mean_total:.4f}\n")

            scores.append(total)
            samples.append(proj_sample)

        scores = np.stack(scores)  # (N,H,W)
        samples = np.stack(samples)  # (N,H,W,3)

        topk_idx = np.argsort(-scores, axis=0)[:top_k]  # (K,H,W)
        topk_w = np.take_along_axis(scores, topk_idx, axis=0)
        topk_img = np.take_along_axis(samples, topk_idx[..., None], axis=0)  # (K,H,W,3)

        topk_w = np.clip(topk_w, 1e-6, None)
        topk_w /= np.sum(topk_w, axis=0, keepdims=True)
        debug_filled = target_rgb.copy()
        debug_filled[hole_mask] = [0, 255, 0]  # 改成綠色確認哪裡被更新
        cv2.imwrite(f"debug_output/filled_check_{frame_index:03d}.png", cv2.cvtColor(debug_filled, cv2.COLOR_RGB2BGR))
        # weighted = np.sum(topk_w[..., None] * topk_img, axis=0)
        # out = target_rgb.copy()
        # out[hole_mask] = weighted[hole_mask]
        # topk_idx shape = (K, H, W)
        top1_idx = topk_idx[0]  # shape (H, W)，每個 pixel 選的 image stack index
        # 建立 output
        out = target_rgb.copy()

        # 逐張 image stack 照著 top1_idx 對應填入
        for idx in np.unique(top1_idx[hole_mask]):
            mask_i = (top1_idx == idx) & hole_mask
            out[mask_i] = stack_imgs[idx][mask_i]

        # log_file.write(f"weighted min/max (whole): {weighted.min():.4f}, {weighted.max():.4f}\n")
        # log_file.write(f"weighted min/max (hole): {weighted[hole_mask].min():.4f}, {weighted[hole_mask].max():.4f}\n")
        # log_file.write(f"hole pixels count: {np.sum(hole_mask)}\n")
        top1_idx = topk_idx[0]  # (H, W)
        top1_img = np.zeros_like(target_rgb)
        for idx in np.unique(top1_idx):
            mask_i = (top1_idx == idx) & hole_mask  # 只處理洞區域
            top1_img[mask_i] = stack_imgs[idx][mask_i]
        debug_dir = Path("top1_sources")
        debug_dir.mkdir(exist_ok=True)

        # 找出每張圖片有貢獻的像素位置
        unique_sources = np.unique(top1_idx[hole_mask])
        for idx in unique_sources:
            positions = np.argwhere((top1_idx == idx) & hole_mask)
            if positions.size == 0:
                continue
            source_img = stack_imgs[idx].copy()
            for y, x in positions:
                cv2.circle(source_img, (x, y), radius=2, color=(0, 255, 255), thickness=-1)
            save_path = debug_dir / f"source_{idx:02d}_frame{frame_index:03d}.png"
            cv2.imwrite(str(save_path), cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR))
        log_file.write(f"[frame {frame_index:03d}] Top-1 image indices in hole: {np.unique(top1_idx[hole_mask])}\n")
    # return out.astype(np.uint8), weighted.astype(np.uint8), samples, scores
    return out.astype(np.uint8), out.astype(np.uint8), samples, scores
def save_debug_visualizations(samples, scores, hole_mask, out_dir: Path):
    """
    儲存每一張 stack 圖像的投影樣貌與權重 heatmap，方便 debug。
    """
    os.makedirs(out_dir / "proj_samples", exist_ok=True)
    os.makedirs(out_dir / "weights", exist_ok=True)

    n = len(samples)
    for i in range(n):
        img = samples[i]
        weight = scores[i]

        # Apply hole mask only for clarity
        masked_img = img.copy()
        masked_img[~hole_mask] = 0

        weight_vis = (255 * (weight - weight.min()) / (weight.max() - weight.min() + 1e-6)).astype(np.uint8)
        weight_vis_colored = cv2.applyColorMap(weight_vis, cv2.COLORMAP_JET)

        cv2.imwrite(str(out_dir / "proj_samples" / f"proj_{i:02d}.png"), cv2.cvtColor(masked_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / "weights" / f"weight_{i:02d}.png"), weight_vis_colored)

   
# 主程式：轉成動畫、遞擺前景不更動，背景補洞後 composite

def create_composited_zoom(img1_path: str, depth1_path: str,
                           img2_path: str, depth2_path: str,
                           stack_dir: Path, n_frames: int = 30,
                           dolly_plane_depth: float = 20.0,
                           video_name: str = 'zoom_filled.mp4'):

    # ---------- 載入前景、後景、深度 ----------
    img1 = cv2.cvtColor(cv2.imread(img1_path), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(img2_path), cv2.COLOR_BGR2RGB)
    depth1 = np.load(depth1_path)['depth']
    depth2 = np.load(depth2_path)['depth']
    H, W = depth1.shape
    img1 = cv2.resize(img1, (W, H))
    img2 = cv2.resize(img2, (W, H))
    depth2 = cv2.resize(depth2, (W, H))

    # ---------- 載入 COLMAP 場景 ----------
    K, quats, poss, names, _, _ = load_colmap_text(stack_dir)
    fx = K[0,0]; fy = K[1,1]
    # stack_imgs = np.stack([
    #     cv2.cvtColor(cv2.imread(str(stack_dir / 'images' / nm)), cv2.COLOR_BGR2RGB)
    #     for nm in names
    # ])
    stack_imgs = np.stack([
        cv2.resize(cv2.cvtColor(cv2.imread(str(stack_dir / 'images' / nm)), cv2.COLOR_BGR2RGB), (W, H))
        for nm in names
    ])
    mvps = [build_proj(K, quat_to_rot(q), t) for q, t in zip(quats, poss)]

    # ---------- 建立影片與暫存資料夾 ----------
    tmp_dir = Path('zoom_frames')
    tmp_dir.mkdir(exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw_filled = cv2.VideoWriter(str(video_name), fourcc, 10, (W, H))
    vw_raw    = cv2.VideoWriter("zoom_raw.mp4", fourcc, 10, (W, H))

    # ---------- 處理每一幀 ----------
    scales = np.linspace(1.0, 1.5, n_frames)
    open("weight_log.txt", "w").close()
    for fi, sc in enumerate(tqdm(scales, desc='frames')):
        # 1. 前景固定（不變）
        warped1 = img1.copy()
        foreground_mask = depth1 <= dolly_plane_depth

        # 2. 後景 warping + 找破洞
        warped_fg_mask, _ = warp_image(foreground_mask.astype(np.float32), depth1, K, sc, dolly_plane_depth)
        warped2, _ = warp_image(img2, depth2, K, sc, dolly_plane_depth)
        # 建立原始空間下的破洞區域：前景擋住背景的地方
        hole_mask_raw = (depth1 <= dolly_plane_depth)  # 前景所在區域

        # warp 到目標空間以取得破洞位置
        warped_hole_mask, _ = warp_image(hole_mask_raw.astype(np.float32), depth1, K, sc, dolly_plane_depth)
        hole_mask = (warped_hole_mask > 0.5) & (depth2 > dolly_plane_depth)

        hole_mask_vis_dir = Path("hole_masks")
        hole_mask_vis_dir.mkdir(exist_ok=True)

        # 2b. 儲存著色後的 hole mask 可視化圖（紅色顯示洞）
        hole_vis = warped2.copy()
        hole_vis[hole_mask] = [255, 0, 0]  # 標紅

        cv2.imwrite(str(Path("hole_masks") / f"hole_{fi:03d}.png"), cv2.cvtColor(hole_vis, cv2.COLOR_RGB2BGR))

        # 使用原始 warped2 圖當底，將 hole 處塗成紅色
        hole_vis = warped2.copy()
        hole_vis[hole_mask] = [255, 0, 0]  # 紅色

        cv2.imwrite(str(hole_mask_vis_dir / f"hole_{fi:03d}.png"), cv2.cvtColor(hole_vis, cv2.COLOR_RGB2BGR))

        # 3. 填補破洞   
        patched_dir = Path('patched_chunks')
        patched_dir.mkdir(exist_ok=True)
        ref_normal = compute_normals(depth2, fx, fy)
        filled, patched, samples, scores = fill_holes_weighted(warped2, hole_mask, depth2, ref_normal,
                             mvps, K, stack_imgs, dolly_plane_depth, frame_index=fi)

        save_debug_visualizations(samples, scores, hole_mask, Path("debug_output"))
        patch = np.zeros_like(patched)
        patch[hole_mask] = patched[hole_mask]

        patch_path = patched_dir / f'patch_{fi:03d}.png'
        cv2.imwrite(str(patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        # 4a. 尚未補洞的合成版本（可見破洞）
        composed_raw = warped2.copy()
        composed_raw[depth1 <= dolly_plane_depth] = warped1[depth1 <= dolly_plane_depth]

        # 4b. 補洞後的合成版本（完整）
        composed_filled = filled.copy()
        composed_filled[depth1 <= dolly_plane_depth] = warped1[depth1 <= dolly_plane_depth]

        # 5. 寫入影片與暫存圖片
        frame_path = tmp_dir / f'frame_{fi:03d}.png'
        cv2.imwrite(str(frame_path), cv2.cvtColor(composed_filled, cv2.COLOR_RGB2BGR))
        vw_filled.write(cv2.cvtColor(composed_filled, cv2.COLOR_RGB2BGR))
        vw_raw.write(cv2.cvtColor(composed_raw, cv2.COLOR_RGB2BGR))
    vw_filled.release()
    vw_raw.release()
    print(f'\n✅ 輸出影片完成：{video_name}（補洞後）、zoom_raw.mp4（補洞前）')

# 启動 CLI
# if __name__ == '__main__':
#     ap = argparse.ArgumentParser()
#     ap.add_argument('--img1', required=True); ap.add_argument('--depth1', required=True)
#     ap.add_argument('--img2', required=True); ap.add_argument('--depth2', required=True)
#     ap.add_argument('--stack_dir', required=True, type=Path)
#     ap.add_argument('--out', default='repaired.png'); ap.add_argument('--frames', type=int, default=30)
#     ap.add_argument('--dolly_z', type=float, default=20.0)
#     args = ap.parse_args()
#     create_composited_zoom(args.img1, args.depth1, args.img2, args.depth2,
#                            args.stack_dir, args.frames, args.dolly_z, args.out)