#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch-rename and resize all images in a folder.
Dependencies: Pillow (pip install pillow)
Author: ChatGPT (2025-05-20)
"""

import os
from pathlib import Path
from PIL import Image

# ========= 參數區（改這裡就好） ========= #
SRC_DIR      = Path("./images")    # 你原始圖片資料夾
DST_DIR      = Path("./changes_images")  # 重新命名＋調整大小後要放哪裡
NAME_PATTERN = "img_{:05d}.jpg"        # 00001, 00002, …；改成你想要的格式
TARGET_W     = 2048                    # 目標寬
TARGET_H     = 1536                    # 目標高
KEEP_RATIO   = False                   # True => 等比縮放並填黑邊；False => 直接拉伸/裁切
OVERWRITE    = False                   # 已存在同名檔案時是否覆蓋
# ====================================== #

def resize_image(im: Image.Image) -> Image.Image:
    if KEEP_RATIO:
        # 等比縮放，然後補邊 (letterbox)
        im.thumbnail((TARGET_W, TARGET_H), Image.LANCZOS)
        new_im = Image.new("RGB", (TARGET_W, TARGET_H), (0, 0, 0))
        offset = ((TARGET_W - im.width) // 2, (TARGET_H - im.height) // 2)
        new_im.paste(im, offset)
        return new_im
    else:
        # 直接 resize 成固定大小（可能會變形）
        return im.resize((TARGET_W, TARGET_H), Image.LANCZOS)

def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    files = sorted([p for p in SRC_DIR.iterdir() if p.suffix.lower() in exts])
    if not files:
        print(f"No images found in {SRC_DIR}")
        return

    for idx, src in enumerate(files, 1):
        dst_name = NAME_PATTERN.format(idx)
        dst_path = DST_DIR / dst_name
        if dst_path.exists() and not OVERWRITE:
            print(f"Skip (exists): {dst_path}")
            continue

        try:
            with Image.open(src) as im:
                im = im.convert("RGB")      # 保證輸出為 8-bit JPG
                im = resize_image(im)
                im.save(dst_path, quality=95)
            print(f"{src.name:>30}  →  {dst_name}")
        except Exception as e:
            print(f"Error processing {src}: {e}")

    print("\nDone!  Renamed & resized {} images → {}".format(len(files), DST_DIR))

if __name__ == "__main__":
    main()