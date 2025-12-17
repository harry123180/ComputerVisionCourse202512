import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import numpy as np
import json
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torchvision import transforms

# ============== Model Definition (Copied from predict_coin.py) ==============
class CoinCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CoinCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ============== Preprocessing (Copied from predict_coin.py) ==============
def get_transform(image_size=224): # IMAGE_SIZE from predict_coin.py
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

# ============== Prediction (Adapted from predict_coin.py) ==============
def predict_coin(model, image_tensor, class_names, device):
    """進行硬幣分類預測"""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = class_names[predicted.item()]
    confidence_value = confidence.item()

    return predicted_class, confidence_value

class ImageProcessorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("OpenCV Interactive Circle Detector")
        self.geometry("1200x800")
        self.minsize(1100, 700)

        self.image_references = []
        self.original_image_cv = None
        self.blurred_image = None
        self.circles_data = None # Raw circles data from HoughCircles
        self.processed_circles_data = [] # Stores circle data with assigned class for drawing
        self.total_weight_var = tk.StringVar(value="Total Weight: N/A")
        self.heads_count_var = tk.StringVar(value="正面 (Heads): N/A")
        self.tails_count_var = tk.StringVar(value="反面 (Tails): N/A")
        
        # Classifier Model Setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier_model = None
        self.classifier_class_names = None
        self.classifier_transform = None
        self.load_classifier_model()


        # Updated default parameters for Hough
        self.hough_params = OrderedDict([
            ('dp', '1.2'), ('minDist', '40'), ('param1', '100'),
            ('param2', '80'),  # Changed to 80
            ('minRadius', '100'), # Changed to 100
            ('maxRadius', '300') # Changed to 300
        ])
        self.hough_param_entries = {}

        self.setup_ui()

    def setup_ui(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        top_frame = ttk.Frame(self, padding="10")
        top_frame.pack(side=tk.TOP, fill=tk.X)
        
        select_button = ttk.Button(top_frame, text="Select Image", command=self.select_image)
        select_button.pack(side=tk.LEFT)
        
        self.image_path_label = ttk.Label(top_frame, text="No image selected", anchor="w")
        self.image_path_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        self.image_frame = ttk.LabelFrame(main_frame, text="Detection Result", padding=10)
        self.image_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        right_pane = ttk.Frame(main_frame)
        right_pane.grid(row=0, column=1, sticky="nsew", padx=(5, 0))
        right_pane.grid_rowconfigure(0, weight=0) # Hough params - fixed size
        right_pane.grid_rowconfigure(1, weight=1) # Results - resizable

        # --- Hough Parameter Frame ---
        hough_frame = ttk.LabelFrame(right_pane, text="Hough Circle Parameters", padding=10)
        hough_frame.grid(row=0, column=0, sticky="new")
        
        for i, (key, value) in enumerate(self.hough_params.items()):
            ttk.Label(hough_frame, text=f"{key}:").grid(row=i, column=0, sticky="w", padx=5, pady=2)
            entry = ttk.Entry(hough_frame, width=10)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            entry.insert(0, value)
            self.hough_param_entries[key] = entry
        
        reprocess_button = ttk.Button(hough_frame, text="Re-process with Parameters", command=self.re_process_hough)
        reprocess_button.grid(row=len(self.hough_params), column=0, columnspan=2, pady=10)

        # --- Classification Results Frame ---
        results_frame = ttk.LabelFrame(right_pane, text="分類結算 (Classification Results)", padding=10)
        results_frame.grid(row=1, column=0, sticky="nsew", pady=5)
        results_frame.grid_rowconfigure(0, weight=1) # Treeview
        results_frame.grid_columnconfigure(0, weight=1)

        results_tree_cols = ('Avg. Dia. (px)', 'Count', 'Weight')
        self.results_tree = ttk.Treeview(results_frame, columns=results_tree_cols, show='headings')
        self.results_tree.heading('Avg. Dia. (px)', text='平均直徑 (px)')
        self.results_tree.heading('Count', text='數量')
        self.results_tree.heading('Weight', text='權重')
        for col in results_tree_cols: self.results_tree.column(col, width=60, anchor='center')
        
        results_tree_scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscroll=results_tree_scrollbar.set)
        self.results_tree.grid(row=0, column=0, sticky="nsew", padx=(0,5))
        results_tree_scrollbar.grid(row=0, column=1, sticky="ns")

        total_weight_label = ttk.Label(results_frame, textvariable=self.total_weight_var, anchor="w", font=('Arial', 12, 'bold'))
        total_weight_label.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10,0))
        
        heads_count_label = ttk.Label(results_frame, textvariable=self.heads_count_var, anchor="w", font=('Arial', 10))
        heads_count_label.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5,0))

        tails_count_label = ttk.Label(results_frame, textvariable=self.tails_count_var, anchor="w", font=('Arial', 10))
        tails_count_label.grid(row=3, column=0, columnspan=2, sticky="ew")


        self.placeholder = ttk.Label(self.image_frame, text="Load an image to begin", anchor="center")
        self.placeholder.pack(expand=True, fill="both")
    
    def load_classifier_model(self):
        """載入訓練好的硬幣分類模型"""
        # SCRIPT_DIR for GUI.py
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "coin_classifier.pth")
        
        if not os.path.exists(model_path):
            messagebox.showwarning(
                "模型錯誤",
                f"找不到硬幣分類模型檔案: {model_path}\n"
                "請確認模型檔案是否存在，或先執行 train_coin.py 訓練模型。"
            )
            return

        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.classifier_class_names = checkpoint['class_names']
            image_size = checkpoint.get('image_size', 224) # Default IMAGE_SIZE for predict_coin.py

            self.classifier_model = CoinCNN(num_classes=len(self.classifier_class_names)).to(self.device)
            self.classifier_model.load_state_dict(checkpoint['model_state_dict'])
            self.classifier_model.eval()
            self.classifier_transform = get_transform(image_size)

            print(f"硬幣分類模型已從 {model_path} 載入")
            print(f"分類類別: {self.classifier_class_names}")
        except Exception as e:
            messagebox.showerror("模型載入錯誤", f"載入硬幣分類模型時發生錯誤: {e}")
            self.classifier_model = None # Ensure model is None on error


    def _get_cjk_font_path(self):
        """搜尋常見的 Windows CJK 字體並回傳第一個找到的路徑"""
        font_paths = [
            "C:/Windows/Fonts/msjh.ttc",  # Microsoft JhengHei (Traditional)
            "C:/Windows/Fonts/simhei.ttf", # SimHei (Simplified)
            "C:/Windows/Fonts/kaiu.ttf",   # DFKai-SB (Traditional)
            "C:/Windows/Fonts/mingliu.ttc" # MingLiU (Traditional)
        ]
        for font_path in font_paths:
            if os.path.exists(font_path):
                return font_path
        return None

    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")])
        if file_path:
            self.image_path_label.config(text=f"Selected: {os.path.basename(file_path)}")
            try:
                with open(file_path, 'rb') as f: file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                self.original_image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if self.original_image_cv is None: raise ValueError("Image data could not be decoded.")
            except Exception as e:
                self.image_path_label.config(text=f"Error: Failed to load {os.path.basename(file_path)}"); return
            
            for key, entry in self.hough_param_entries.items():
                entry.delete(0, tk.END)
                entry.insert(0, self.hough_params[key])

            self.re_process_hough()

    def re_process_hough(self):
        if self.original_image_cv is None:
            messagebox.showinfo("No Image", "Please select an image first."); return
        
        self.image_references = []; [widget.destroy() for widget in self.image_frame.winfo_children()]
        
        try:
            hough_params = {key: float(entry.get()) for key, entry in self.hough_param_entries.items()}
            for key in ['minRadius', 'maxRadius', 'minDist', 'param1', 'param2']:
                hough_params[key] = int(hough_params[key])
        except (ValueError, TypeError):
            messagebox.showerror("Parameter Error", "Hough Circle parameters must be valid numbers."); return

        self.blurred_image = self.apply_blur(self.original_image_cv.copy(), {'ksize': 9})
        
        circles = self.apply_hough_circles(self.blurred_image, hough_params) # Get raw circles
        self.circles_data = circles # Store raw circles data
        
        # Process and display results (classification, weighting, image drawing)
        self.process_and_display_results(circles)

    def process_and_display_results(self, circles_raw):
        # Clear previous results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        self.total_weight_var.set("Total Weight: N/A")
        self.heads_count_var.set("正面 (Heads): 0")
        self.tails_count_var.set("反面 (Tails): 0")
        self.processed_circles_data = [] # Reset processed data

        output_image = self.original_image_cv.copy() # Start with fresh image for drawing

        if circles_raw is None or len(circles_raw[0]) == 0:
            self.add_image_to_widget(self.image_frame, output_image) # Display original image if no circles
            return

        circles = np.uint16(np.around(circles_raw))
        # Convert to list of (x, y, r) tuples for easier processing
        circle_list = [{'center': (c[0], c[1]), 'radius': c[2], 'diameter': c[2] * 2, 'original_idx': i} for i, c in enumerate(circles[0])]

        # --- Classification ---
        # Sort circles by diameter to facilitate grouping
        circle_list.sort(key=lambda x: x['diameter'])

        classes = [] # Each class will be {'diameters': [], 'avg_diameter': 0, 'count': 0, 'weight': 0}
        tolerance = 20 # Pixels

        # Group circles into classes
        for circle_info in circle_list:
            assigned_to_class = False
            for cls in classes:
                if abs(circle_info['diameter'] - cls['avg_diameter']) <= tolerance:
                    cls['diameters'].append(circle_info['diameter'])
                    cls['avg_diameter'] = np.mean(cls['diameters'])
                    cls['count'] += 1
                    assigned_to_class = True
                    break
            if not assigned_to_class:
                new_class = {
                    'diameters': [circle_info['diameter']],
                    'avg_diameter': circle_info['diameter'],
                    'count': 1,
                    'weight': 0
                }
                classes.append(new_class)
        
        # Now, map each circle to its class for drawing purposes
        final_processed_circles_data = []
        for circle_info in circle_list:
            for cls_idx, cls in enumerate(classes):
                if abs(circle_info['diameter'] - cls['avg_diameter']) <= tolerance:
                    final_processed_circles_data.append({
                        'original_circle': circles_raw[0, circle_info['original_idx']],
                        'class_idx': cls_idx,
                        'diameter': circle_info['diameter']
                    })
                    break
        self.processed_circles_data = final_processed_circles_data

        # Sort classes by average diameter
        classes.sort(key=lambda x: x['avg_diameter'])
        
        # --- Weight Assignment ---
        num_classes = len(classes)
        if num_classes > 0:
            avg_diameters = [c['avg_diameter'] for c in classes]
            median_diameter = np.median(avg_diameters)
            
            for cls in classes:
                if cls['avg_diameter'] < median_diameter:
                    cls['weight'] = 1
                elif cls['avg_diameter'] > median_diameter:
                    cls['weight'] = 10
                else:
                    cls['weight'] = 5

        # --- Total Weight & Heads/Tails Calculation ---
        total_calculated_weight = 0
        heads_count = 0
        tails_count = 0
        padding = 10

        for p_circle in self.processed_circles_data:
            assigned_weight = 0
            for cls in classes:
                if abs(p_circle['diameter'] - cls['avg_diameter']) <= tolerance:
                    assigned_weight = cls['weight']
                    break
            total_calculated_weight += assigned_weight
            
            pred_class, conf_value = None, 0.0
            if self.classifier_model and self.classifier_transform:
                circle_data_raw = p_circle['original_circle']
                center_int = (int(circle_data_raw[0]), int(circle_data_raw[1]))
                radius_int = int(circle_data_raw[2])
                
                start_x = max(0, int(center_int[0] - radius_int - padding))
                start_y = max(0, int(center_int[1] - radius_int - padding))
                end_x = min(output_image.shape[1], int(center_int[0] + radius_int + padding))
                end_y = min(output_image.shape[0], int(center_int[1] + radius_int + padding))
                roi_to_predict = self.original_image_cv[start_y:end_y, start_x:end_x]

                if roi_to_predict.size > 0:
                    try:
                        roi_pil = Image.fromarray(cv2.cvtColor(roi_to_predict, cv2.COLOR_BGR2RGB))
                        image_tensor = self.classifier_transform(roi_pil).unsqueeze(0)
                        pred_class, conf_value = predict_coin(self.classifier_model, image_tensor, self.classifier_class_names, self.device)
                        if pred_class == 'heads':
                            heads_count += 1
                        elif pred_class == 'tails':
                            tails_count += 1
                    except Exception as e:
                        print(f"Error during prediction for counting: {e}")
                        pred_class = "Error"
            p_circle['pred_class'] = pred_class
            p_circle['conf_value'] = conf_value
        
        self.total_weight_var.set(f"Total Weight: {total_calculated_weight}")
        self.heads_count_var.set(f"正面 (Heads): {heads_count}")
        self.tails_count_var.set(f"反面 (Tails): {tails_count}")

        for cls in classes:
            self.results_tree.insert('', 'end', values=(f"{cls['avg_diameter']:.1f}", cls['count'], cls['weight']))

        font_thickness = 2
        for p_circle in self.processed_circles_data:
            circle_data_raw = p_circle['original_circle']
            center_int = (int(circle_data_raw[0]), int(circle_data_raw[1]))
            radius_int = int(circle_data_raw[2])
            cv2.circle(output_image, center_int, radius_int, (255, 0, 255), font_thickness)
            cv2.circle(output_image, center_int, 2, (0, 255, 0), font_thickness + 1)
            start_x = max(0, int(center_int[0] - radius_int - padding))
            start_y = max(0, int(center_int[1] - radius_int - padding))
            end_x = min(output_image.shape[1], int(center_int[0] + radius_int + padding))
            end_y = min(output_image.shape[0], int(center_int[1] + radius_int + padding))
            cv2.rectangle(output_image, (start_x, start_y), (end_x, end_y), (0, 255, 0), font_thickness)
        
        output_pil = Image.fromarray(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(output_pil)
        
        font_path = self._get_cjk_font_path()
        use_cjk_font = font_path is not None

        if use_cjk_font:
            font_dia = ImageFont.truetype(font_path, 40)
            font_coin = ImageFont.truetype(font_path, 30)
        else:
            print("警告: 找不到任何支援中文的字體，將使用預設字體並以英文顯示。")
            font_dia = ImageFont.load_default(size=25)
            font_coin = ImageFont.load_default(size=18)

        for p_circle in self.processed_circles_data:
            circle_data_raw = p_circle['original_circle']
            center_int = (int(circle_data_raw[0]), int(circle_data_raw[1]))
            radius_int = int(circle_data_raw[2])
            diameter_float = p_circle['diameter']
            pred_class = p_circle['pred_class']
            conf_value = p_circle['conf_value']
            
            label_dia = f"D: {diameter_float:.0f}"
            bbox_dia = draw.textbbox((0, 0), label_dia, font=font_dia)
            text_w_dia, text_h_dia = bbox_dia[2] - bbox_dia[0], bbox_dia[3] - bbox_dia[1]
            text_pos_dia = (center_int[0] - text_w_dia // 2, center_int[1] - radius_int - text_h_dia - 15)
            draw.text(text_pos_dia, label_dia, font=font_dia, fill=(255, 0, 0))

            coin_display_label = "N/A"
            if pred_class:
                if use_cjk_font:
                    coin_display_label = "正面" if pred_class == "heads" else "反面"
                else:
                    coin_display_label = pred_class.capitalize()

            label_coin = f"{coin_display_label} ({conf_value*100:.0f}%)" if pred_class and pred_class != "Error" else coin_display_label
            bbox_coin = draw.textbbox((0, 0), label_coin, font=font_coin)
            text_w_coin = bbox_coin[2] - bbox_coin[0]
            text_pos_coin = (center_int[0] - text_w_coin // 2, center_int[1] + radius_int + 20)
            draw.text(text_pos_coin, label_coin, font=font_coin, fill=(0, 0, 255))
        
        output_image = cv2.cvtColor(np.array(output_pil), cv2.COLOR_RGB2BGR)

        self.add_image_to_widget(self.image_frame, output_image)


    def add_image_to_widget(self, parent_widget, image_cv):
        parent_widget.update_idletasks()
        max_w = parent_widget.winfo_width() - 20; max_h = parent_widget.winfo_height() - 20
        if max_w < 1 or max_h < 1: max_w, max_h = 800, 600
        display_image = self.resize_for_display(image_cv, max_width=max_w, max_height=max_h)
        img_pil = Image.fromarray(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.image_references.append(img_tk)
        image_label = ttk.Label(parent_widget, image=img_tk); image_label.pack(expand=True, fill="both")

    def apply_blur(self, image, params):
        ksize = int(params.get("ksize", 9))
        if ksize % 2 == 0: ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def apply_threshold(self, image, params):
        thresh_val = int(params.get("thresholdValue", 127)); max_val = int(params.get("maxValue", 255))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        _, binary = cv2.threshold(gray, thresh_val, max_val, cv2.THRESH_BINARY)
        return binary
    
    def apply_hough_circles(self, image, params):
        # This method now only performs detection and returns raw circles data.
        # Image drawing and further processing are handled by process_and_display_results.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                                   dp=params.get('dp', 1.2), minDist=params.get('minDist', 50),
                                   param1=params.get('param1', 100), param2=params.get('param2', 30),
                                   minRadius=params.get('minRadius', 10), maxRadius=params.get('maxRadius', 100))
        return circles

    def resize_for_display(self, image, max_width, max_height):
        h, w = image.shape[:2]
        if w > max_width or h > max_height:
            ratio = min(max_width / w, max_height / h)
            if ratio <= 0: return image
            return cv2.resize(image, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_AREA)
        return image
if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()