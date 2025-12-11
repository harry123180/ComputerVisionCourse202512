# DAY2 - 深度學習圖片分類器 (PyTorch)

## 課程目標

本課程將學習如何使用 PyTorch 建立深度學習圖片分類器：
- 神經網路 (NN) 與卷積神經網路 (CNN) 基礎
- MNIST 手寫數字辨識
- 貓狗二分類器
- 遷移學習 (Transfer Learning)
- 自訂義分類器 (預留)

---

## 環境設置

### 1. 安裝 Python

確保已安裝 Python 3.8 或以上版本。

### 2. 安裝 PyTorch

#### Windows / Linux (有 NVIDIA GPU)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Windows / Linux (無 GPU，使用 CPU)
```bash
pip install torch torchvision
```

#### Mac
```bash
pip install torch torchvision
```

### 3. 安裝其他相依套件
```bash
cd DAY2
pip install -r requirements.txt
```

### 4. 驗證安裝
```python
import torch
print(f"PyTorch 版本: {torch.__version__}")
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 目錄結構

```
DAY2/
├── requirements.txt         # Python 相依套件
├── README.md               # 本說明文件
├── models/                  # 模型儲存目錄 (gitignore)
│   ├── mnist_cnn.pth       # MNIST 模型
│   ├── catdog_model.pth    # 貓狗分類模型
│   └── coin_classifier.pth # 硬幣分類模型
├── 01_MNIST/               # MNIST 手寫數字辨識
│   ├── train.py           # 訓練腳本
│   ├── predict.py         # 預測腳本
│   ├── realtime_webcam.py # WebCam 即時辨識
│   └── draw_predict.py    # 滑鼠手寫辨識
├── 02_CatDog/              # 貓狗分類器
│   ├── train.py           # 訓練腳本
│   ├── predict.py         # 預測腳本
│   └── download_sample_data.py  # 資料準備工具
├── 03_Custom/              # 自訂義分類器
│   ├── train_coin.py       # 硬幣正反面訓練腳本
│   ├── predict_coin.py     # 硬幣預測腳本
│   ├── capture_tool.py     # WebCam 資料蒐集工具
│   └── dataset/            # 資料集目錄 (gitignore)
│       ├── heads/          # 硬幣正面圖片
│       └── tails/          # 硬幣反面圖片
└── 04_ROI/                 # ROI 工具
    ├── roi_tool.py         # ROI GUI 工具
    └── roi_example.py      # ROI 命令列範例
```

**注意**: `models/` 和 `dataset/` 目錄已加入 `.gitignore`，不會上傳至 GitHub。

---

## 範例一：MNIST 手寫數字辨識

### 簡介

MNIST 是機器學習領域最經典的入門資料集，包含 60,000 張訓練圖片和 10,000 張測試圖片，每張圖片是 28x28 像素的灰階手寫數字 (0-9)。

### 操作步驟

#### 步驟 1：進入目錄
```bash
cd DAY2/01_MNIST
```

#### 步驟 2：執行訓練
```bash
python train.py
```

程式會自動：
1. 下載 MNIST 資料集（首次執行時）
2. 建立 CNN 模型
3. 訓練 10 個 epoch
4. 顯示訓練曲線
5. 儲存模型至 `mnist_cnn.pth`

#### 步驟 3：使用模型預測
```bash
python predict.py --image your_image.png
```

### 模型架構

```
SimpleCNN
├── Conv2d(1, 32, 3x3) + BatchNorm + ReLU + MaxPool
├── Conv2d(32, 64, 3x3) + BatchNorm + ReLU + MaxPool
├── Flatten
├── Linear(64*7*7, 128) + ReLU + Dropout
└── Linear(128, 10)
```

### 預期結果

- 訓練時間：約 5-10 分鐘 (CPU) / 1-2 分鐘 (GPU)
- 測試準確率：約 99%

### 輸出檔案

| 檔案 | 說明 |
|------|------|
| `mnist_cnn.pth` | 訓練好的模型權重 |
| `training_history.png` | 訓練損失/準確率曲線 |
| `prediction_samples.png` | 預測範例視覺化 |

### 即時辨識功能

訓練完成後，可以使用以下兩種即時辨識方式：

#### 方式 A：WebCam 即時辨識

使用攝影機拍攝紙張上的手寫數字進行辨識。

```bash
python realtime_webcam.py
```

**操作說明：**
1. 將手寫數字紙張對準畫面中央的綠色框
2. 按 `c` 擷取並辨識
3. 按 `q` 退出程式

**小技巧：**
- 使用深色筆在白紙上書寫，效果最佳
- 確保光線充足，避免陰影
- 數字盡量置中於綠色框內

#### 方式 B：滑鼠手寫辨識

直接用滑鼠在畫布上書寫數字。

```bash
python draw_predict.py
```

**操作說明：**
1. 按住滑鼠左鍵在黑色畫布上書寫數字
2. 按 `p` 進行預測
3. 按 `c` 清除畫布
4. 按 `q` 退出程式

**小技巧：**
- 數字盡量寫大一些
- 筆劃要連貫
- 書寫風格可以參考 MNIST 資料集的樣式

---

## 範例二：貓狗分類器

### 簡介

使用 CNN 和遷移學習建立貓狗二分類器。支援兩種模式：
- **自定義 CNN**：從頭訓練
- **遷移學習**：使用預訓練的 ResNet18（推薦）

### 資料準備

#### 步驟 1：建立資料目錄
```bash
cd DAY2/02_CatDog
python download_sample_data.py
```

#### 步驟 2：放入圖片

將圖片依照類別放入對應目錄：
```
02_CatDog/
└── data/
    ├── cats/          # 放入貓的圖片
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    │   └── ...
    └── dogs/          # 放入狗的圖片
        ├── dog1.jpg
        ├── dog2.jpg
        └── ...
```

#### 資料集下載來源

1. **Kaggle Dogs vs Cats** (推薦)
   - 網址：https://www.kaggle.com/c/dogs-vs-cats/data
   - 約 25,000 張訓練圖片

2. **Oxford-IIIT Pet Dataset**
   - 網址：https://www.robots.ox.ac.uk/~vgg/data/pets/
   - 學術研究用資料集

3. **自己蒐集**
   - 建議每類至少 100 張以上

### 操作步驟

#### 步驟 1：訓練模型
```bash
cd DAY2/02_CatDog
python train.py
```

#### 步驟 2：修改訓練設定 (可選)

編輯 `train.py` 頂部的設定區：

```python
BATCH_SIZE = 32          # 批次大小 (GPU 記憶體不足時調小)
EPOCHS = 10              # 訓練輪數
LEARNING_RATE = 0.001    # 學習率
USE_PRETRAINED = True    # 是否使用預訓練模型
```

#### 步驟 3：預測單張圖片
```bash
python predict.py --image path/to/image.jpg
```

#### 步驟 4：批次預測
```bash
python predict.py --folder path/to/images/
```

### 模型架構

**使用預訓練模型 (ResNet18)**：
```
ResNet18 (預訓練)
├── [凍結] 卷積層 (從 ImageNet 學習的特徵)
└── [訓練] 全連接層
    ├── Linear(512, 256) + ReLU + Dropout
    └── Linear(256, 2)
```

**自定義 CNN**：
```
SimpleCNN
├── Conv Block 1: Conv2d(3, 32) + BN + ReLU + Pool
├── Conv Block 2: Conv2d(32, 64) + BN + ReLU + Pool
├── Conv Block 3: Conv2d(64, 128) + BN + ReLU + Pool
├── Conv Block 4: Conv2d(128, 256) + BN + ReLU + Pool
├── Flatten
├── Linear(256*14*14, 512) + ReLU + Dropout
└── Linear(512, 2)
```

### 預期結果

| 模式 | 訓練時間 | 準確率 |
|------|----------|--------|
| 預訓練 ResNet18 | 較快 | 90-95% |
| 自定義 CNN | 較慢 | 80-90% |

### 輸出檔案

| 檔案 | 說明 |
|------|------|
| `catdog_model.pth` | 訓練好的模型權重 |
| `training_history.png` | 訓練曲線 |
| `prediction_samples.png` | 預測範例 |
| `prediction_result.png` | 單張預測結果 |

---

## 範例三：硬幣正反面分類器

### 簡介

使用自訂資料集訓練硬幣正反面 (Heads/Tails) 分類器。程式會自動處理不同大小的圖片並切分訓練/驗證集。

### 資料準備

#### 步驟 1：建立資料夾結構
```bash
cd DAY2/03_Custom
```

程式會自動建立以下目錄結構：
```
03_Custom/
└── dataset/
    ├── heads/    # 放入硬幣正面圖片
    └── tails/    # 放入硬幣反面圖片
```

#### 步驟 2：放入圖片

- 將硬幣**正面**圖片放入 `dataset/heads/` 資料夾
- 將硬幣**反面**圖片放入 `dataset/tails/` 資料夾
- 支援格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`
- 圖片大小不限，程式會自動調整

**建議**：每類至少準備 20 張以上圖片

### 操作步驟

#### 步驟 1：訓練模型
```bash
cd DAY2/03_Custom
python train_coin.py
```

程式會自動：
1. 載入所有圖片並統一調整大小
2. 隨機切分 80% 訓練集 / 20% 驗證集
3. 套用資料增強（翻轉、旋轉、色彩調整）
4. 訓練 CNN 模型
5. 儲存最佳模型至 `models/coin_classifier.pth`

#### 步驟 2：預測單張圖片
```bash
python predict_coin.py --image path/to/coin.jpg
```

#### 步驟 3：批次預測
```bash
python predict_coin.py --folder path/to/images/
```

### 模型架構

```
CoinCNN
├── Conv Block 1: Conv2d(3, 32) + BN + ReLU + Pool
├── Conv Block 2: Conv2d(32, 64) + BN + ReLU + Pool
├── Conv Block 3: Conv2d(64, 128) + BN + ReLU + Pool
├── Conv Block 4: Conv2d(128, 256) + BN + ReLU + Pool
├── Conv Block 5: Conv2d(256, 512) + BN + ReLU + Pool
├── Global Average Pooling
├── Linear(512, 256) + ReLU + Dropout
└── Linear(256, 2)
```

### 資料增強策略

訓練時會自動套用以下資料增強：
- 隨機裁切
- 水平/垂直翻轉
- 旋轉 (±30°)
- 色彩抖動

### 輸出檔案

| 檔案 | 說明 |
|------|------|
| `models/coin_classifier.pth` | 訓練好的模型 |
| `training_history.png` | 訓練曲線 |
| `prediction_samples.png` | 預測範例 |

---

## 範例四：ROI 工具

### 簡介

ROI (Region of Interest) 是影像處理中常用的技術，用於指定感興趣區域進行處理。本工具提供 GUI 和命令列兩種方式操作。

### 功能

1. **ROI Mask** - 只保留指定區域，其他區域變黑
2. **Inverse ROI Mask** - 遮蔽指定區域，保留其他區域
3. **ROI 裁切** - 直接裁切指定區域

### 方式 A：GUI 工具

```bash
cd DAY2/04_ROI
python roi_tool.py
```

**操作步驟：**
1. 點擊「載入圖片」選擇圖片
2. 在圖片上用滑鼠拖曳選取 ROI 區域
3. 或手動輸入 X, Y, W, H 參數
4. 點擊「套用 ROI」查看結果
5. 可儲存 ROI Mask / Inverse Mask / 裁切結果

### 方式 B：命令列範例

```bash
cd DAY2/04_ROI

# 使用示範圖片
python roi_example.py

# 指定圖片和 ROI 參數
python roi_example.py --image path/to/image.jpg --x 100 --y 100 --w 200 --h 150

# 儲存結果
python roi_example.py --image path/to/image.jpg --x 100 --y 100 --w 200 --h 150 --save
```

### 核心程式碼範例

```python
import cv2
import numpy as np

# 載入圖片
image = cv2.imread("image.jpg")
h, w = image.shape[:2]

# 定義 ROI 參數
roi_x, roi_y, roi_w, roi_h = 100, 100, 200, 150

# 建立 ROI Mask (保留 ROI 區域)
mask = np.zeros((h, w), dtype=np.uint8)
mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = 255
roi_result = cv2.bitwise_and(image, image, mask=mask)

# 建立 Inverse Mask (遮蔽 ROI 區域)
inverse_mask = cv2.bitwise_not(mask)
inverse_result = cv2.bitwise_and(image, image, mask=inverse_mask)

# 裁切 ROI 區域
cropped = image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
```

---

## 核心概念說明

### 卷積神經網路 (CNN)

CNN 專門用於處理圖像資料，主要組成：

1. **卷積層 (Convolution)**：提取圖像特徵
2. **池化層 (Pooling)**：降低維度，增加平移不變性
3. **全連接層 (Fully Connected)**：進行最終分類

### 遷移學習 (Transfer Learning)

利用在大型資料集（如 ImageNet）上預訓練的模型，將學習到的特徵轉移到新任務：

- **優點**：訓練快、準確率高、所需資料量少
- **做法**：凍結預訓練層，只訓練最後的分類層

### 資料增強 (Data Augmentation)

訓練時對影像進行隨機變換，增加資料多樣性：

```python
transforms.RandomHorizontalFlip()    # 隨機水平翻轉
transforms.RandomRotation(15)        # 隨機旋轉
transforms.ColorJitter(...)          # 色彩抖動
```

---

## 常見問題

### Q: CUDA out of memory 錯誤

**A**: GPU 記憶體不足，嘗試：
- 減小 `BATCH_SIZE`（例如：32 → 16 → 8）
- 減小 `IMAGE_SIZE`
- 使用 CPU 訓練

### Q: 訓練很慢

**A**:
- 確認是否使用 GPU：`torch.cuda.is_available()` 應返回 `True`
- 減少訓練資料量進行測試
- 使用預訓練模型加速收斂

### Q: 準確率很低

**A**:
- 確認資料是否正確分類
- 增加訓練資料量
- 調整學習率
- 增加訓練輪數
- 檢查是否過擬合（訓練準確率高但驗證準確率低）

### Q: 如何使用自己的圖片進行預測？

**A**:
1. 確保圖片是常見格式（.jpg, .png）
2. 使用 predict.py：
   ```bash
   python predict.py --image your_image.jpg
   ```

### Q: 如何儲存和載入模型？

**A**:
```python
# 儲存
torch.save(model.state_dict(), 'model.pth')

# 載入
model = YourModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()
```

---

## 練習題

### 初級
1. 修改 MNIST 的訓練輪數，觀察準確率變化
2. 調整學習率，比較訓練效果
3. 嘗試使用不同的優化器（SGD vs Adam）

### 中級
1. 在 MNIST 模型中增加更多卷積層
2. 調整貓狗分類器的資料增強策略
3. 比較預訓練模型和自定義模型的效果

### 進階
1. 實作早停 (Early Stopping) 機制
2. 加入學習率衰減策略
3. 使用不同的預訓練模型（VGG16, ResNet50）

---

## 參考資源

- [PyTorch 官方教學](https://pytorch.org/tutorials/)
- [MNIST 資料集](http://yann.lecun.com/exdb/mnist/)
- [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats)
- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)

---

## 下一步

完成 DAY2 的學習後，你將具備：
- PyTorch 深度學習框架的基礎操作
- CNN 圖片分類模型的建立與訓練
- 遷移學習的實作經驗
- 模型評估與視覺化技巧

繼續學習 DAY3 的進階內容！
