"""
ROI (Region of Interest) 範例程式
展示如何使用 OpenCV 進行 ROI Mask 和 Inverse Mask 操作

使用方式:
python roi_example.py --image path/to/image.jpg --x 100 --y 100 --w 200 --h 150

或不帶參數執行，使用預設值進行示範
"""

import cv2
import numpy as np
import argparse
import os

# ============== 取得腳本所在目錄 ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_roi_mask(image, x, y, w, h):
    """
    建立 ROI Mask

    Args:
        image: 輸入圖片 (BGR)
        x, y: ROI 左上角座標
        w, h: ROI 寬度和高度

    Returns:
        mask: 二值化遮罩 (ROI 區域為白色)
        roi_result: 只保留 ROI 區域的圖片
    """
    img_h, img_w = image.shape[:2]

    # 建立全黑遮罩
    mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # 在 ROI 區域填入白色
    mask[y:y+h, x:x+w] = 255

    # 套用遮罩 (只保留 ROI 區域)
    roi_result = cv2.bitwise_and(image, image, mask=mask)

    return mask, roi_result


def create_inverse_roi_mask(image, x, y, w, h):
    """
    建立 Inverse ROI Mask (反向遮罩)

    Args:
        image: 輸入圖片 (BGR)
        x, y: ROI 左上角座標
        w, h: ROI 寬度和高度

    Returns:
        inverse_mask: 反向二值化遮罩 (ROI 區域為黑色)
        inverse_result: 遮蔽 ROI 區域的圖片
    """
    img_h, img_w = image.shape[:2]

    # 建立全白遮罩
    inverse_mask = np.ones((img_h, img_w), dtype=np.uint8) * 255

    # 在 ROI 區域填入黑色
    inverse_mask[y:y+h, x:x+w] = 0

    # 套用遮罩 (遮蔽 ROI 區域)
    inverse_result = cv2.bitwise_and(image, image, mask=inverse_mask)

    return inverse_mask, inverse_result


def crop_roi(image, x, y, w, h):
    """
    裁切 ROI 區域

    Args:
        image: 輸入圖片 (BGR)
        x, y: ROI 左上角座標
        w, h: ROI 寬度和高度

    Returns:
        cropped: 裁切後的圖片
    """
    return image[y:y+h, x:x+w].copy()


def draw_roi_rectangle(image, x, y, w, h, color=(0, 255, 0), thickness=2):
    """
    在圖片上繪製 ROI 矩形框

    Args:
        image: 輸入圖片 (會被修改)
        x, y: ROI 左上角座標
        w, h: ROI 寬度和高度
        color: 框線顏色 (BGR)
        thickness: 框線粗細

    Returns:
        image: 繪製後的圖片
    """
    result = image.copy()
    cv2.rectangle(result, (x, y), (x+w, y+h), color, thickness)
    return result


def create_demo_image():
    """建立示範用圖片"""
    # 建立一個彩色漸層圖片
    img = np.zeros((400, 600, 3), dtype=np.uint8)

    # 繪製彩色背景
    for i in range(400):
        for j in range(600):
            img[i, j] = [
                int(255 * j / 600),      # B
                int(255 * i / 400),      # G
                int(255 * (1 - j / 600)) # R
            ]

    # 加入一些形狀
    cv2.circle(img, (150, 200), 80, (255, 255, 255), -1)
    cv2.rectangle(img, (350, 100), (500, 300), (255, 255, 0), -1)
    cv2.putText(img, "ROI Demo", (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    return img


def main():
    parser = argparse.ArgumentParser(description='ROI Mask 範例程式')
    parser.add_argument('--image', '-i', type=str, help='輸入圖片路徑')
    parser.add_argument('--x', type=int, default=100, help='ROI X 座標')
    parser.add_argument('--y', type=int, default=100, help='ROI Y 座標')
    parser.add_argument('--w', type=int, default=200, help='ROI 寬度')
    parser.add_argument('--h', type=int, default=150, help='ROI 高度')
    parser.add_argument('--save', '-s', action='store_true', help='儲存結果圖片')

    args = parser.parse_args()

    print("=" * 50)
    print("ROI (Region of Interest) 範例")
    print("=" * 50)
    print()

    # 載入或建立圖片
    if args.image:
        if not os.path.exists(args.image):
            print(f"錯誤: 找不到圖片 {args.image}")
            return
        image = cv2.imread(args.image)
        print(f"載入圖片: {args.image}")
    else:
        print("使用示範圖片")
        image = create_demo_image()

    img_h, img_w = image.shape[:2]
    print(f"圖片大小: {img_w} x {img_h}")
    print()

    # ROI 參數
    x, y, w, h = args.x, args.y, args.w, args.h

    # 檢查範圍
    if x + w > img_w:
        w = img_w - x
        print(f"警告: ROI 寬度超出範圍，調整為 {w}")
    if y + h > img_h:
        h = img_h - y
        print(f"警告: ROI 高度超出範圍，調整為 {h}")

    print(f"ROI 參數: X={x}, Y={y}, W={w}, H={h}")
    print()

    # ========== 1. 建立 ROI Mask ==========
    print("[1] 建立 ROI Mask...")
    mask, roi_result = create_roi_mask(image, x, y, w, h)
    print("    - 只保留 ROI 區域內的像素")

    # ========== 2. 建立 Inverse ROI Mask ==========
    print("[2] 建立 Inverse ROI Mask...")
    inverse_mask, inverse_result = create_inverse_roi_mask(image, x, y, w, h)
    print("    - 遮蔽 ROI 區域，保留其他區域")

    # ========== 3. 裁切 ROI ==========
    print("[3] 裁切 ROI 區域...")
    cropped = crop_roi(image, x, y, w, h)
    print(f"    - 裁切後大小: {cropped.shape[1]} x {cropped.shape[0]}")

    # ========== 4. 繪製 ROI 框 ==========
    print("[4] 繪製 ROI 矩形框...")
    image_with_roi = draw_roi_rectangle(image, x, y, w, h)

    print()
    print("-" * 50)
    print("顯示結果視窗 (按任意鍵關閉)")
    print("-" * 50)

    # 顯示結果
    # 調整視窗大小以便顯示
    def resize_for_display(img, max_width=600, max_height=500):
        h, w = img.shape[:2]
        scale = min(max_width/w, max_height/h, 1.0)
        return cv2.resize(img, (int(w*scale), int(h*scale)))

    # 建立顯示視窗
    cv2.imshow("Original + ROI Box", resize_for_display(image_with_roi))
    cv2.imshow("ROI Mask Result", resize_for_display(roi_result))
    cv2.imshow("Inverse Mask Result", resize_for_display(inverse_result))
    cv2.imshow("Cropped ROI", resize_for_display(cropped))

    # 移動視窗位置
    cv2.moveWindow("Original + ROI Box", 50, 50)
    cv2.moveWindow("ROI Mask Result", 700, 50)
    cv2.moveWindow("Inverse Mask Result", 50, 400)
    cv2.moveWindow("Cropped ROI", 700, 400)

    # 儲存結果
    if args.save:
        print()
        print("儲存結果...")
        cv2.imwrite("roi_original_with_box.png", image_with_roi)
        cv2.imwrite("roi_mask_result.png", roi_result)
        cv2.imwrite("roi_inverse_result.png", inverse_result)
        cv2.imwrite("roi_cropped.png", cropped)
        print("已儲存: roi_original_with_box.png")
        print("已儲存: roi_mask_result.png")
        print("已儲存: roi_inverse_result.png")
        print("已儲存: roi_cropped.png")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print()
    print("=" * 50)
    print("完成!")
    print("=" * 50)


if __name__ == "__main__":
    main()
