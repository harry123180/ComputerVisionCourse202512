"""
手勢辨識 GUI 輔助函數與常數
"""

import os
import urllib.request
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 音效播放 (使用 pygame 支援音量控制)
import pygame

# ============== 路徑設定 ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 模型路徑
MODEL_PATH = os.path.join(SCRIPT_DIR, "gesture_recognizer.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"

# 音效路徑
SOUND_FIST = os.path.join(PROJECT_ROOT, "Hand Sign_2.wav")      # 握拳音效
SOUND_TAB = os.path.join(PROJECT_ROOT, "Hand Sign.wav")         # 切換 Tab 音效


# ============== 尺寸設定 ==============
CIRCLE_SIZE = 70          # 圓形按鈕直徑
EXPANDED_SIZE = 500       # 展開後的正方形邊長


# ============== 手勢對照表 ==============
GESTURE_MAP = {
    "None": {"full": "無手勢", "short": "---"},
    "Closed_Fist": {"full": "握拳 👊", "short": "👊"},
    "Open_Palm": {"full": "張開手掌 🖐️", "short": "🖐️"},
    "Pointing_Up": {"full": "指向上 ☝️", "short": "☝️"},
    "Thumb_Down": {"full": "拇指向下 👎", "short": "👎"},
    "Thumb_Up": {"full": "拇指向上 👍", "short": "👍"},
    "Victory": {"full": "勝利 ✌️", "short": "✌️"},
    "ILoveYou": {"full": "我愛你 🤟", "short": "🤟"},
}


# ============== 動作事件對照表 ==============
ACTION_MAP = {
    "alt_tab_start": {"full": "🔄 Alt+Tab 啟動", "color": (0, 255, 255)},
    "prev_window": {"full": "👍 上一個視窗", "color": (0, 255, 0)},
    "next_window": {"full": "👎 下一個視窗", "color": (255, 128, 0)},
    "confirm_select": {"full": "👊 確認選擇", "color": (0, 128, 255)},
}


# ============== 音效系統 ==============
# 初始化 pygame mixer
pygame.mixer.init()

# 預載入音效
sound_fist = None
sound_tab = None

if os.path.exists(SOUND_FIST):
    sound_fist = pygame.mixer.Sound(SOUND_FIST)
    print(f"已載入音效: {SOUND_FIST}")
if os.path.exists(SOUND_TAB):
    sound_tab = pygame.mixer.Sound(SOUND_TAB)
    print(f"已載入音效: {SOUND_TAB}")

# 全域音量 (0.0 ~ 1.0)
current_volume = 0.5


def set_volume(volume):
    """設定音量 (0.0 ~ 1.0)"""
    global current_volume
    current_volume = max(0.0, min(1.0, volume))


def get_volume():
    """取得目前音量"""
    return current_volume


def play_sound_async(sound_type):
    """
    異步播放音效 (不阻塞主執行緒)

    Args:
        sound_type: 'fist' 或 'tab'
    """
    global current_volume
    try:
        if sound_type == 'fist' and sound_fist:
            sound_fist.set_volume(current_volume)
            sound_fist.play()
        elif sound_type == 'tab' and sound_tab:
            sound_tab.set_volume(current_volume)
            sound_tab.play()
    except Exception as e:
        print(f"音效播放失敗: {e}")


def cleanup_sound():
    """清理音效系統"""
    pygame.mixer.quit()


# ============== 模型下載 ==============
def download_model():
    """下載手勢辨識模型"""
    if not os.path.exists(MODEL_PATH):
        print(f"正在下載手勢辨識模型...")
        print(f"URL: {MODEL_URL}")
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"模型已下載至: {MODEL_PATH}")
        except Exception as e:
            print(f"下載失敗: {e}")
            print("請手動下載模型並放到 DAY4 資料夾")
            return False
    return True


# ============== 時間格式化 ==============
def get_time_str():
    """取得目前時間字串"""
    return time.strftime("%H:%M:%S")


# ============== 中文文字繪製 ==============
# 嘗試載入中文字體
FONT_PATH = "C:/Windows/Fonts/msjh.ttc"  # 微軟正黑體
if not os.path.exists(FONT_PATH):
    FONT_PATH = "C:/Windows/Fonts/simsun.ttc"  # 新細明體備選
if not os.path.exists(FONT_PATH):
    FONT_PATH = None


def put_chinese_text(img, text, position, font_size=24, color=(0, 255, 0)):
    """
    在 OpenCV 圖像上繪製中文文字

    Args:
        img: OpenCV 圖像 (BGR)
        text: 要繪製的文字
        position: (x, y) 位置
        font_size: 字體大小
        color: BGR 顏色

    Returns:
        繪製後的圖像
    """
    if FONT_PATH is None:
        # 沒有中文字體，使用 OpenCV 預設 (會顯示???)
        import cv2
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return img

    # 轉換為 PIL Image
    img_pil = Image.fromarray(img[..., ::-1])  # BGR to RGB
    draw = ImageDraw.Draw(img_pil)

    # 載入字體
    try:
        font = ImageFont.truetype(FONT_PATH, font_size)
    except Exception:
        font = ImageFont.load_default()

    # 繪製文字 (PIL 使用 RGB 顏色)
    rgb_color = (color[2], color[1], color[0])
    draw.text(position, text, font=font, fill=rgb_color)

    # 轉回 OpenCV 格式
    return np.array(img_pil)[..., ::-1]  # RGB to BGR
