"""
浮動圓形按鈕 GUI + MediaPipe 手勢辨識
- 預設為右下角的小圓形按鈕，顯示手勢簡稱
- 點擊展開為正方形設定視窗，顯示 webcam 畫面
- 始終置頂

手勢動作 (Alt+Tab 視窗切換):
- 握拳 👊 = 啟動 Alt+Tab 視窗切換
- 無手勢 = 動作間的斷點 (準備下一個動作)
- 拇指向上 👍 = 上一個視窗 (Shift+Tab)
- 拇指向下 👎 = 下一個視窗 (Tab)
- 再次握拳 👊 = 確認選擇並關閉 Alt+Tab

安裝套件:
    pip install pyautogui mediapipe customtkinter opencv-python pillow pygame

需要先下載手勢辨識模型:
https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
放到 DAY4 資料夾 (會自動下載)
"""

import customtkinter as ctk
from PIL import Image, ImageTk
import cv2
import threading
import os
import time
from tkinter import messagebox

# MediaPipe
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 鍵盤控制
import pyautogui

# 匯入自定義函數和常數
from functions import (
    CIRCLE_SIZE, EXPANDED_SIZE,
    GESTURE_MAP, ACTION_MAP,
    MODEL_PATH, MODEL_URL,
    play_sound_async, set_volume, get_volume,
    download_model, cleanup_sound, put_chinese_text
)

# 設定外觀
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# pyautogui 設定
pyautogui.FAILSAFE = False


class FloatingBubble(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 狀態
        self.is_expanded = False
        self.webcam_running = False
        self.cap = None
        self.current_gesture = "None"
        self.prev_gesture = "None"
        self.gesture_recognizer = None

        # 動作檢測狀態
        self.current_action = None
        self.action_display_time = 0
        self.action_display_duration = 1.0

        # Alt+Tab 狀態機
        self.alt_tab_active = False
        self.ready_for_action = False

        # 移除標題欄
        self.overrideredirect(True)

        # 置頂
        self.attributes('-topmost', True)

        # 設定背景
        self.configure(fg_color='#1a1a2e')

        # 取得螢幕尺寸
        self.screen_width = self.winfo_screenwidth()
        self.screen_height = self.winfo_screenheight()

        # 初始化手勢辨識器
        self.init_gesture_recognizer()

        # 初始化為圓形模式
        self.setup_circle_mode()

        # 拖曳功能
        self._drag_x = 0
        self._drag_y = 0

        # 啟動 webcam
        self.start_webcam()

    def init_gesture_recognizer(self):
        """初始化手勢辨識器"""
        if not os.path.exists(MODEL_PATH):
            print(f"找不到模型: {MODEL_PATH}")
            return

        try:
            base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
            options = vision.GestureRecognizerOptions(
                base_options=base_options,
                num_hands=2
            )
            self.gesture_recognizer = vision.GestureRecognizer.create_from_options(options)
            print("手勢辨識器初始化成功")
        except Exception as e:
            print(f"手勢辨識器初始化失敗: {e}")

    def process_gesture_state_machine(self, current_time):
        """
        手勢狀態機處理 Alt+Tab 視窗切換

        流程:
        1. 握拳 (Closed_Fist) → 啟動 Alt+Tab
        2. 無手勢 (None) → 準備接收下一個動作
        3. 拇指向上 (Thumb_Up) → 上一個視窗
        4. 拇指向下 (Thumb_Down) → 下一個視窗
        5. 再次握拳 (Closed_Fist) → 確認選擇，關閉 Alt+Tab
        """
        gesture = self.current_gesture
        prev = self.prev_gesture

        gesture_changed = (gesture != prev)

        if not gesture_changed:
            return

        # 狀態 1: Alt+Tab 未啟動
        if not self.alt_tab_active:
            if gesture == "Closed_Fist" and prev == "None":
                self.start_alt_tab(current_time)
                self.ready_for_action = False

        # 狀態 2: Alt+Tab 已啟動
        else:
            if gesture == "None":
                self.ready_for_action = True
                print("準備接收下一個動作...")

            elif self.ready_for_action and prev == "None":
                if gesture == "Thumb_Up":
                    self.switch_prev_window(current_time)
                    self.ready_for_action = False

                elif gesture == "Thumb_Down":
                    self.switch_next_window(current_time)
                    self.ready_for_action = False

                elif gesture == "Closed_Fist":
                    self.confirm_selection(current_time)
                    self.ready_for_action = False

    def start_alt_tab(self, current_time):
        """啟動 Alt+Tab"""
        print("啟動 Alt+Tab")
        self.alt_tab_active = True

        play_sound_async('fist')

        pyautogui.keyDown('alt')
        pyautogui.press('tab')

        self.trigger_action("alt_tab_start", current_time)

    def switch_prev_window(self, current_time):
        """切換到上一個視窗 (Shift+Tab)"""
        if not self.alt_tab_active:
            return

        print("上一個視窗")
        play_sound_async('tab')

        pyautogui.hotkey('shift', 'tab')
        self.trigger_action("prev_window", current_time)

    def switch_next_window(self, current_time):
        """切換到下一個視窗 (Tab)"""
        if not self.alt_tab_active:
            return

        print("下一個視窗")
        play_sound_async('tab')

        pyautogui.press('tab')
        self.trigger_action("next_window", current_time)

    def confirm_selection(self, current_time):
        """確認選擇並關閉 Alt+Tab"""
        print("確認選擇")
        play_sound_async('fist')

        pyautogui.keyUp('alt')
        self.alt_tab_active = False

        self.trigger_action("confirm_select", current_time)

    def trigger_action(self, action_type, current_time):
        """觸發動作事件"""
        self.current_action = action_type
        self.action_display_time = current_time

        action_info = ACTION_MAP.get(action_type, {})
        event_text = f"{action_info.get('full', action_type)}"
        print(f"觸發動作: {event_text}")

    def setup_circle_mode(self):
        """設定圓形按鈕模式"""
        self.is_expanded = False

        for widget in self.winfo_children():
            widget.destroy()

        # 右下角位置
        margin = 20
        x = self.screen_width - CIRCLE_SIZE - margin
        y = self.screen_height - CIRCLE_SIZE - margin - 40
        self.geometry(f"{CIRCLE_SIZE}x{CIRCLE_SIZE}+{x}+{y}")

        self.circle_frame = ctk.CTkFrame(
            self,
            width=CIRCLE_SIZE,
            height=CIRCLE_SIZE,
            corner_radius=CIRCLE_SIZE // 2,
            fg_color='#4a90d9'
        )
        self.circle_frame.pack(expand=True, fill='both')
        self.circle_frame.pack_propagate(False)

        self.circle_label = ctk.CTkLabel(
            self.circle_frame,
            text="---",
            font=ctk.CTkFont(size=28),
            text_color='white'
        )
        self.circle_label.place(relx=0.5, rely=0.5, anchor='center')

        # 綁定事件
        self.circle_frame.bind('<Button-1>', self.on_click)
        self.circle_label.bind('<Button-1>', self.on_click)
        self.circle_frame.bind('<ButtonPress-1>', self.start_drag)
        self.circle_frame.bind('<B1-Motion>', self.on_drag)
        self.circle_label.bind('<ButtonPress-1>', self.start_drag)
        self.circle_label.bind('<B1-Motion>', self.on_drag)

        # Hover 效果
        self.circle_frame.bind('<Enter>', lambda e: self.circle_frame.configure(fg_color='#5ba0e9'))
        self.circle_frame.bind('<Leave>', lambda e: self.circle_frame.configure(fg_color='#4a90d9'))

    def setup_expanded_mode(self):
        """設定展開後的正方形視窗模式 (置中)"""
        self.is_expanded = True

        for widget in self.winfo_children():
            widget.destroy()

        # 置中位置
        x = (self.screen_width - EXPANDED_SIZE) // 2
        y = (self.screen_height - EXPANDED_SIZE) // 2
        self.geometry(f"{EXPANDED_SIZE}x{EXPANDED_SIZE}+{x}+{y}")

        # 主容器
        self.main_frame = ctk.CTkFrame(
            self,
            corner_radius=15,
            fg_color='#1a1a2e'
        )
        self.main_frame.pack(expand=True, fill='both', padx=2, pady=2)

        # 標題欄
        self.title_bar = ctk.CTkFrame(
            self.main_frame,
            height=40,
            corner_radius=0,
            fg_color='#2d2d44'
        )
        self.title_bar.pack(fill='x', padx=10, pady=(10, 5))
        self.title_bar.pack_propagate(False)

        self.title_label = ctk.CTkLabel(
            self.title_bar,
            text="🖐️ 手勢辨識 (Alt+Tab)",
            font=ctk.CTkFont(size=14, weight='bold')
        )
        self.title_label.pack(side='left', padx=10, pady=5)

        self.close_btn = ctk.CTkButton(
            self.title_bar,
            text="✕",
            width=30,
            height=30,
            corner_radius=15,
            fg_color='transparent',
            hover_color='#ff6b6b',
            command=self.collapse
        )
        self.close_btn.pack(side='right', padx=5, pady=5)

        # 內容區域
        self.content_frame = ctk.CTkFrame(
            self.main_frame,
            fg_color='transparent'
        )
        self.content_frame.pack(expand=True, fill='both', padx=10, pady=5)

        # Webcam 顯示區域
        self.video_label = ctk.CTkLabel(
            self.content_frame,
            text="Webcam Loading...",
            width=EXPANDED_SIZE - 40,
            height=200
        )
        self.video_label.pack(pady=5)

        # 手勢顯示
        self.gesture_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color='#2d2d44',
            corner_radius=10
        )
        self.gesture_frame.pack(fill='x', pady=5)

        self.gesture_label = ctk.CTkLabel(
            self.gesture_frame,
            text="手勢: ---",
            font=ctk.CTkFont(size=18, weight='bold'),
            text_color='#00ff7f'
        )
        self.gesture_label.pack(pady=10)

        # 動作顯示區域
        self.action_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color='#3d3d5c',
            corner_radius=10
        )
        self.action_frame.pack(fill='x', pady=5)

        self.action_label = ctk.CTkLabel(
            self.action_frame,
            text="動作: ---",
            font=ctk.CTkFont(size=16, weight='bold'),
            text_color='#ffff00'
        )
        self.action_label.pack(pady=8)

        # 音量控制區域
        volume_frame = ctk.CTkFrame(
            self.content_frame,
            fg_color='#2d2d44',
            corner_radius=10
        )
        volume_frame.pack(fill='x', pady=5)

        volume_label = ctk.CTkLabel(
            volume_frame,
            text="🔊 音量",
            font=ctk.CTkFont(size=12),
            text_color='#aaaaaa'
        )
        volume_label.pack(side='left', padx=10, pady=8)

        self.volume_value_label = ctk.CTkLabel(
            volume_frame,
            text=f"{int(get_volume() * 100)}%",
            font=ctk.CTkFont(size=12),
            text_color='#00ff7f',
            width=40
        )
        self.volume_value_label.pack(side='right', padx=10, pady=8)

        self.volume_slider = ctk.CTkSlider(
            volume_frame,
            from_=0,
            to=100,
            number_of_steps=20,
            command=self.on_volume_change
        )
        self.volume_slider.set(get_volume() * 100)
        self.volume_slider.pack(side='right', padx=10, pady=8, fill='x', expand=True)

        # 控制區域
        control_frame = ctk.CTkFrame(self.content_frame, fg_color='transparent')
        control_frame.pack(fill='x', pady=5)

        self.webcam_var = ctk.BooleanVar(value=self.webcam_running)
        self.webcam_switch = ctk.CTkSwitch(
            control_frame,
            text="Webcam",
            variable=self.webcam_var,
            command=self.toggle_webcam
        )
        self.webcam_switch.pack(side='left', padx=10)

        self.topmost_var = ctk.BooleanVar(value=True)
        self.topmost_switch = ctk.CTkSwitch(
            control_frame,
            text="置頂",
            variable=self.topmost_var,
            command=self.toggle_topmost
        )
        self.topmost_switch.pack(side='right', padx=10)

        # 綁定標題欄拖曳
        self.title_bar.bind('<ButtonPress-1>', self.start_drag)
        self.title_bar.bind('<B1-Motion>', self.on_drag)
        self.title_label.bind('<ButtonPress-1>', self.start_drag)
        self.title_label.bind('<B1-Motion>', self.on_drag)

    def start_webcam(self):
        """啟動 webcam"""
        if self.webcam_running:
            return

        try:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                raise Exception("無法開啟攝影機")

            # 測試讀取一幀
            ret, _ = self.cap.read()
            if not ret:
                self.cap.release()
                raise Exception("攝影機無法讀取畫面")

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.webcam_running = True
            print("Webcam 已啟動")

            self.update_thread = threading.Thread(target=self.webcam_loop, daemon=True)
            self.update_thread.start()

        except Exception as e:
            error_msg = f"攝影機錯誤: {str(e)}\n\n請確認:\n1. 攝影機已連接\n2. 沒有其他程式佔用攝影機\n3. 攝影機驅動已安裝"
            print(error_msg)
            self.cap = None
            self.webcam_running = False

            # 顯示錯誤訊息
            self.after(100, lambda: messagebox.showerror("攝影機錯誤", error_msg))

    def stop_webcam(self):
        """停止 webcam"""
        self.webcam_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("Webcam 已停止")

    def toggle_webcam(self):
        """切換 webcam 開關"""
        if self.webcam_var.get():
            self.start_webcam()
        else:
            self.stop_webcam()
            self.current_gesture = "None"
            self.update_gesture_display()

    def webcam_loop(self):
        """Webcam 處理迴圈"""
        while self.webcam_running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                continue

            current_time = time.time()

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if self.gesture_recognizer is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                try:
                    result = self.gesture_recognizer.recognize(mp_image)

                    if result.gestures and len(result.gestures) > 0:
                        gesture = result.gestures[0][0].category_name
                        self.current_gesture = gesture
                    else:
                        self.current_gesture = "None"
                except Exception as e:
                    pass

            self.process_gesture_state_machine(current_time)
            self.prev_gesture = self.current_gesture

            if self.current_action and (current_time - self.action_display_time) > self.action_display_duration:
                self.current_action = None

            self.after(0, self.update_gesture_display)
            self.after(0, self.update_action_display)

            if self.is_expanded:
                self.after(0, lambda f=frame.copy(): self.update_video_display(f))

            cv2.waitKey(30)

    def update_gesture_display(self):
        """更新手勢顯示"""
        gesture_info = GESTURE_MAP.get(self.current_gesture, GESTURE_MAP["None"])

        if hasattr(self, 'circle_label') and self.circle_label.winfo_exists():
            if self.current_action:
                short_action = {
                    "alt_tab_start": "🔄",
                    "prev_window": "👍",
                    "next_window": "👎",
                    "confirm_select": "✅"
                }.get(self.current_action, "---")
                self.circle_label.configure(text=short_action)
            elif self.alt_tab_active:
                self.circle_label.configure(text="🔄")
            else:
                self.circle_label.configure(text=gesture_info["short"])

        if hasattr(self, 'circle_frame') and self.circle_frame.winfo_exists():
            if self.alt_tab_active:
                self.circle_frame.configure(fg_color='#e67e22')
            else:
                self.circle_frame.configure(fg_color='#4a90d9')

        if hasattr(self, 'gesture_label') and self.gesture_label.winfo_exists():
            status = " [Alt+Tab 啟動中]" if self.alt_tab_active else ""
            self.gesture_label.configure(text=f"手勢: {gesture_info['full']}{status}")

    def update_action_display(self):
        """更新動作顯示"""
        if hasattr(self, 'action_label') and self.action_label.winfo_exists():
            if self.current_action:
                action_info = ACTION_MAP.get(self.current_action, {})
                self.action_label.configure(
                    text=f"動作: {action_info.get('full', '---')}",
                    text_color='#00ff00'
                )
            else:
                self.action_label.configure(
                    text="動作: ---",
                    text_color='#ffff00'
                )

    def update_video_display(self, frame):
        """更新影像顯示"""
        if not hasattr(self, 'video_label') or not self.video_label.winfo_exists():
            return

        h, w = frame.shape[:2]

        # 繪製手勢文字 (使用中文字體)
        gesture_info = GESTURE_MAP.get(self.current_gesture, GESTURE_MAP["None"])
        frame = put_chinese_text(frame, gesture_info["full"], (10, 5), font_size=24, color=(0, 255, 0))

        if self.alt_tab_active:
            frame = put_chinese_text(frame, "[Alt+Tab 啟動中]", (10, 35), font_size=20, color=(0, 255, 255))

            tips = "👍上一個 | 👎下一個 | 👊確認"
            frame = put_chinese_text(frame, tips, (10, h - 30), font_size=16, color=(200, 200, 200))

        if self.current_action:
            action_info = ACTION_MAP.get(self.current_action, {})
            action_text = action_info.get("full", "")
            action_color = action_info.get("color", (255, 255, 255))

            # 繪製半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (w//4, h//3), (3*w//4, 2*h//3), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

            # 繪製動作文字 (置中)
            text_x = w // 4 + 20
            text_y = h // 2 - 15
            frame = put_chinese_text(frame, action_text, (text_x, text_y), font_size=28, color=action_color)

        frame = cv2.resize(frame, (EXPANDED_SIZE - 40, 200))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(frame_rgb)
        photo = ctk.CTkImage(light_image=image, dark_image=image, size=(EXPANDED_SIZE - 40, 200))

        self.video_label.configure(image=photo, text="")
        self.video_label.image = photo

    def on_click(self, event):
        """點擊圓形按鈕時展開"""
        if not hasattr(self, '_click_x'):
            self.expand()
            return

        dx = abs(event.x_root - self._click_x)
        dy = abs(event.y_root - self._click_y)

        if dx < 5 and dy < 5:
            self.expand()

    def expand(self):
        """展開視窗"""
        if not self.is_expanded:
            self.setup_expanded_mode()
            if hasattr(self, 'webcam_var'):
                self.webcam_var.set(self.webcam_running)

    def collapse(self):
        """收合視窗"""
        if self.is_expanded:
            self.setup_circle_mode()

    def start_drag(self, event):
        """開始拖曳"""
        self._drag_x = event.x
        self._drag_y = event.y
        self._click_x = event.x_root
        self._click_y = event.y_root

    def on_drag(self, event):
        """拖曳中"""
        x = self.winfo_x() + event.x - self._drag_x
        y = self.winfo_y() + event.y - self._drag_y
        self.geometry(f"+{x}+{y}")

    def toggle_topmost(self):
        """切換置頂狀態"""
        is_top = self.topmost_var.get()
        self.attributes('-topmost', is_top)

    def on_volume_change(self, value):
        """音量滑桿變更"""
        set_volume(value / 100.0)

        if hasattr(self, 'volume_value_label') and self.volume_value_label.winfo_exists():
            self.volume_value_label.configure(text=f"{int(value)}%")

    def on_closing(self):
        """關閉視窗"""
        if self.alt_tab_active:
            pyautogui.keyUp('alt')
            self.alt_tab_active = False
        self.stop_webcam()
        cleanup_sound()
        self.destroy()


def main():
    """主程式入口"""
    # 下載模型
    if not download_model():
        print("請手動下載模型後再執行")
        print(f"下載網址: {MODEL_URL}")

    app = FloatingBubble()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == "__main__":
    main()
