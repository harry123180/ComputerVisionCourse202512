import cv2
import numpy as np

# --- 讀取圖片 ---

img = cv2.imread("coins.jpg")
output = img.copy()

# --- 灰階 + 模糊 ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 2)

# --- Hough 圓形偵測 ---
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=80,
    param1=100,
    param2=30,
    minRadius=20,
    maxRadius=100
)

coin_count = {
    "1 元": 0,
    "5 元": 0,
    "10 元": 0,
    "未知": 0
}

# --- 根據半徑分類 ---
def classify_coin(r):
    if r < 45:
        return "1 元"
    elif r < 55:
        return "5 元"
    elif r < 70:
        return "10 元"
    else:
        return "未知"

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        coin_type = classify_coin(r)
        coin_count[coin_type] += 1

        # 畫圓
        cv2.circle(output, (x, y), r, (0, 255, 0), 3)
        cv2.putText(output, coin_type, (x - 40, y - r - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# --- 印出結果 ---
print("辨識結果：")
for k, v in coin_count.items():
    print(f"{k}：{v} 枚")

# --- 顯示結果 ---
cv2.imshow("Detected Coins", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

