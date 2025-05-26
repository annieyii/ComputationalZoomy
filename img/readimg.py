import cv2

img = cv2.imread("./raw/pic1.png")  # 讀取圖片（BGR 格式）
h, w = img.shape[:2]  # 取得高度與寬度

print(f"圖片寬度：{w} 像素")
print(f"圖片高度：{h} 像素")