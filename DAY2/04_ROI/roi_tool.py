"""
ROI (Region of Interest) 工具
可以指定輸入圖片，設定 ROI 區域 (x, y, w, h)，產生 Mask 和 Inverse Mask

功能:
1. 載入圖片
2. 輸入 ROI 參數 (x, y, width, height)
3. 顯示 ROI Mask (保留 ROI 區域)
4. 顯示 Inverse ROI Mask (遮蔽 ROI 區域)
5. 滑鼠拖曳選取 ROI

操作說明:
- 載入圖片後，可以手動輸入 x, y, w, h
- 或在圖片上用滑鼠拖曳選取 ROI 區域
- 點擊「套用 ROI」查看結果
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os

# ============== 取得腳本所在目錄 ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class ROIToolGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ROI 工具 - Region of Interest")
        self.root.geometry("1200x700")

        # 圖片相關
        self.original_image = None
        self.display_image = None
        self.image_path = None

        # ROI 相關
        self.roi_start = None
        self.roi_end = None
        self.is_drawing = False

        # 顯示縮放比例
        self.scale = 1.0

        self._setup_ui()

    def _setup_ui(self):
        """建立使用者介面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # ========== 左側：控制面板 ==========
        left_frame = ttk.Frame(main_frame, width=250)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        left_frame.pack_propagate(False)

        # 載入圖片按鈕
        ttk.Button(left_frame, text="載入圖片", command=self._load_image).pack(fill=tk.X, pady=5)

        # 分隔線
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # ROI 參數輸入
        ttk.Label(left_frame, text="ROI 參數", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        ttk.Label(left_frame, text="(可手動輸入或滑鼠拖曳選取)", font=("Arial", 9)).pack(anchor=tk.W)

        params_frame = ttk.Frame(left_frame)
        params_frame.pack(fill=tk.X, pady=10)

        # X
        ttk.Label(params_frame, text="X:").grid(row=0, column=0, sticky=tk.E, padx=2)
        self.entry_x = ttk.Entry(params_frame, width=8)
        self.entry_x.grid(row=0, column=1, padx=2)
        self.entry_x.insert(0, "0")

        # Y
        ttk.Label(params_frame, text="Y:").grid(row=0, column=2, sticky=tk.E, padx=2)
        self.entry_y = ttk.Entry(params_frame, width=8)
        self.entry_y.grid(row=0, column=3, padx=2)
        self.entry_y.insert(0, "0")

        # Width
        ttk.Label(params_frame, text="W:").grid(row=1, column=0, sticky=tk.E, padx=2, pady=5)
        self.entry_w = ttk.Entry(params_frame, width=8)
        self.entry_w.grid(row=1, column=1, padx=2, pady=5)
        self.entry_w.insert(0, "100")

        # Height
        ttk.Label(params_frame, text="H:").grid(row=1, column=2, sticky=tk.E, padx=2, pady=5)
        self.entry_h = ttk.Entry(params_frame, width=8)
        self.entry_h.grid(row=1, column=3, padx=2, pady=5)
        self.entry_h.insert(0, "100")

        # 套用按鈕
        ttk.Button(left_frame, text="套用 ROI", command=self._apply_roi).pack(fill=tk.X, pady=5)

        # 分隔線
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 圖片資訊
        ttk.Label(left_frame, text="圖片資訊", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        self.lbl_image_info = ttk.Label(left_frame, text="尚未載入圖片")
        self.lbl_image_info.pack(anchor=tk.W, pady=5)

        # 分隔線
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 儲存按鈕
        ttk.Label(left_frame, text="儲存結果", font=("Arial", 11, "bold")).pack(anchor=tk.W)

        ttk.Button(left_frame, text="儲存 ROI Mask 結果",
                   command=lambda: self._save_result("roi")).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="儲存 Inverse Mask 結果",
                   command=lambda: self._save_result("inverse")).pack(fill=tk.X, pady=2)
        ttk.Button(left_frame, text="儲存 ROI 裁切區域",
                   command=lambda: self._save_result("crop")).pack(fill=tk.X, pady=2)

        # 分隔線
        ttk.Separator(left_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # 操作說明
        ttk.Label(left_frame, text="操作說明", font=("Arial", 11, "bold")).pack(anchor=tk.W)
        instructions = [
            "1. 載入圖片",
            "2. 在圖片上拖曳滑鼠選取 ROI",
            "   或手動輸入 X, Y, W, H",
            "3. 點擊「套用 ROI」",
            "4. 查看 Mask 和 Inverse 結果",
        ]
        for inst in instructions:
            ttk.Label(left_frame, text=inst, font=("Arial", 9)).pack(anchor=tk.W)

        # ========== 右側：圖片顯示區 ==========
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 使用 Notebook 分頁顯示
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 原始圖片分頁
        self.tab_original = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_original, text="原始圖片")

        self.canvas_original = tk.Canvas(self.tab_original, bg="gray30", cursor="cross")
        self.canvas_original.pack(fill=tk.BOTH, expand=True)

        # 綁定滑鼠事件
        self.canvas_original.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas_original.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas_original.bind("<ButtonRelease-1>", self._on_mouse_up)

        # ROI Mask 分頁
        self.tab_roi = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_roi, text="ROI Mask")

        self.canvas_roi = tk.Canvas(self.tab_roi, bg="gray30")
        self.canvas_roi.pack(fill=tk.BOTH, expand=True)

        # Inverse Mask 分頁
        self.tab_inverse = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_inverse, text="Inverse Mask")

        self.canvas_inverse = tk.Canvas(self.tab_inverse, bg="gray30")
        self.canvas_inverse.pack(fill=tk.BOTH, expand=True)

        # 狀態列
        self.status_var = tk.StringVar(value="請載入圖片")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _load_image(self):
        """載入圖片"""
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[
                ("Image Files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"),
                ("All Files", "*.*")
            ]
        )

        if not file_path:
            return

        self.image_path = file_path
        self.original_image = cv2.imread(file_path)

        if self.original_image is None:
            messagebox.showerror("錯誤", "無法載入圖片")
            return

        h, w = self.original_image.shape[:2]
        self.lbl_image_info.config(text=f"大小: {w} x {h}\n{os.path.basename(file_path)}")

        # 顯示圖片
        self._display_original()
        self.status_var.set(f"已載入: {os.path.basename(file_path)}")

    def _display_original(self):
        """顯示原始圖片"""
        if self.original_image is None:
            return

        # 計算縮放比例
        canvas_w = self.canvas_original.winfo_width()
        canvas_h = self.canvas_original.winfo_height()

        if canvas_w < 10:
            canvas_w = 800
        if canvas_h < 10:
            canvas_h = 600

        h, w = self.original_image.shape[:2]
        scale_w = canvas_w / w
        scale_h = canvas_h / h
        self.scale = min(scale_w, scale_h, 1.0)

        new_w = int(w * self.scale)
        new_h = int(h * self.scale)

        # 縮放圖片
        display = cv2.resize(self.original_image, (new_w, new_h))
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        self.display_image = display

        # 轉換為 Tkinter 格式
        img = Image.fromarray(display_rgb)
        self.photo_original = ImageTk.PhotoImage(img)

        self.canvas_original.delete("all")
        self.canvas_original.create_image(0, 0, image=self.photo_original, anchor=tk.NW)

    def _on_mouse_down(self, event):
        """滑鼠按下事件"""
        if self.original_image is None:
            return

        self.is_drawing = True
        self.roi_start = (event.x, event.y)
        self.roi_end = (event.x, event.y)

    def _on_mouse_drag(self, event):
        """滑鼠拖曳事件"""
        if not self.is_drawing or self.original_image is None:
            return

        self.roi_end = (event.x, event.y)

        # 重新繪製
        self._display_original()

        # 繪製選取框
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end

        self.canvas_original.create_rectangle(
            x1, y1, x2, y2,
            outline="lime", width=2, dash=(4, 4)
        )

    def _on_mouse_up(self, event):
        """滑鼠放開事件"""
        if not self.is_drawing or self.original_image is None:
            return

        self.is_drawing = False
        self.roi_end = (event.x, event.y)

        # 計算 ROI 參數 (轉換回原始圖片座標)
        x1, y1 = self.roi_start
        x2, y2 = self.roi_end

        # 確保 x1 < x2, y1 < y2
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        # 轉換為原始圖片座標
        orig_x = int(x1 / self.scale)
        orig_y = int(y1 / self.scale)
        orig_w = int((x2 - x1) / self.scale)
        orig_h = int((y2 - y1) / self.scale)

        # 更新輸入框
        self.entry_x.delete(0, tk.END)
        self.entry_x.insert(0, str(orig_x))

        self.entry_y.delete(0, tk.END)
        self.entry_y.insert(0, str(orig_y))

        self.entry_w.delete(0, tk.END)
        self.entry_w.insert(0, str(orig_w))

        self.entry_h.delete(0, tk.END)
        self.entry_h.insert(0, str(orig_h))

        self.status_var.set(f"已選取 ROI: X={orig_x}, Y={orig_y}, W={orig_w}, H={orig_h}")

    def _apply_roi(self):
        """套用 ROI"""
        if self.original_image is None:
            messagebox.showwarning("警告", "請先載入圖片")
            return

        try:
            x = int(self.entry_x.get())
            y = int(self.entry_y.get())
            w = int(self.entry_w.get())
            h = int(self.entry_h.get())
        except ValueError:
            messagebox.showerror("錯誤", "請輸入有效的數字")
            return

        img_h, img_w = self.original_image.shape[:2]

        # 檢查範圍
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            messagebox.showerror("錯誤", "ROI 參數必須為正數")
            return

        if x + w > img_w or y + h > img_h:
            messagebox.showwarning("警告", "ROI 超出圖片範圍，將自動調整")
            w = min(w, img_w - x)
            h = min(h, img_h - y)

        # 建立 Mask
        mask = np.zeros((img_h, img_w), dtype=np.uint8)
        mask[y:y+h, x:x+w] = 255

        # ROI Mask 結果 (保留 ROI 區域)
        roi_result = cv2.bitwise_and(self.original_image, self.original_image, mask=mask)

        # Inverse Mask 結果 (遮蔽 ROI 區域)
        inverse_mask = cv2.bitwise_not(mask)
        inverse_result = cv2.bitwise_and(self.original_image, self.original_image, mask=inverse_mask)

        # 儲存結果供儲存使用
        self.roi_result = roi_result
        self.inverse_result = inverse_result
        self.roi_params = (x, y, w, h)

        # 顯示結果
        self._display_result(self.canvas_roi, roi_result, "ROI Mask")
        self._display_result(self.canvas_inverse, inverse_result, "Inverse Mask")

        # 更新原始圖片顯示 (加上 ROI 框)
        self._display_original()
        x1 = int(x * self.scale)
        y1 = int(y * self.scale)
        x2 = int((x + w) * self.scale)
        y2 = int((y + h) * self.scale)

        self.canvas_original.create_rectangle(
            x1, y1, x2, y2,
            outline="lime", width=2
        )

        self.status_var.set(f"已套用 ROI: X={x}, Y={y}, W={w}, H={h}")

        # 切換到 ROI Mask 分頁
        self.notebook.select(self.tab_roi)

    def _display_result(self, canvas, image, title):
        """顯示結果圖片"""
        canvas_w = canvas.winfo_width()
        canvas_h = canvas.winfo_height()

        if canvas_w < 10:
            canvas_w = 800
        if canvas_h < 10:
            canvas_h = 600

        h, w = image.shape[:2]
        scale_w = canvas_w / w
        scale_h = canvas_h / h
        scale = min(scale_w, scale_h, 1.0)

        new_w = int(w * scale)
        new_h = int(h * scale)

        display = cv2.resize(image, (new_w, new_h))
        display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

        img = Image.fromarray(display_rgb)
        photo = ImageTk.PhotoImage(img)

        canvas.delete("all")
        canvas.create_image(0, 0, image=photo, anchor=tk.NW)

        # 保持參考避免被垃圾回收
        canvas.photo = photo

    def _save_result(self, result_type):
        """儲存結果"""
        if self.original_image is None:
            messagebox.showwarning("警告", "請先載入圖片")
            return

        if not hasattr(self, 'roi_result'):
            messagebox.showwarning("警告", "請先套用 ROI")
            return

        # 選擇儲存路徑
        default_name = os.path.splitext(os.path.basename(self.image_path))[0]

        if result_type == "roi":
            save_image = self.roi_result
            default_name += "_roi_mask.png"
            title = "儲存 ROI Mask 結果"
        elif result_type == "inverse":
            save_image = self.inverse_result
            default_name += "_inverse_mask.png"
            title = "儲存 Inverse Mask 結果"
        elif result_type == "crop":
            x, y, w, h = self.roi_params
            save_image = self.original_image[y:y+h, x:x+w]
            default_name += "_crop.png"
            title = "儲存 ROI 裁切區域"

        file_path = filedialog.asksaveasfilename(
            title=title,
            defaultextension=".png",
            initialfile=default_name,
            filetypes=[
                ("PNG", "*.png"),
                ("JPEG", "*.jpg"),
                ("BMP", "*.bmp"),
            ]
        )

        if file_path:
            cv2.imwrite(file_path, save_image)
            self.status_var.set(f"已儲存: {os.path.basename(file_path)}")
            messagebox.showinfo("成功", f"已儲存至:\n{file_path}")


def main():
    root = tk.Tk()
    app = ROIToolGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
