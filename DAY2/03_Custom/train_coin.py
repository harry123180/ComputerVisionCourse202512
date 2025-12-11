"""
硬幣正反面分類器 - 訓練腳本
自動處理大小不一的圖片，切分數據集，使用 CNN 進行分類

資料夾結構:
dataset/
├── heads/    # 放入硬幣正面圖片
└── tails/    # 放入硬幣反面圖片

使用方式:
python train_coin.py
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# ============== 取得腳本所在目錄 (相對路徑基準) ==============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============== 設定區 ==============
BATCH_SIZE = 16          # 批次大小
EPOCHS = 20              # 訓練輪數
LEARNING_RATE = 0.002    # 學習率
IMAGE_SIZE = 224         # 統一影像大小
TRAIN_SPLIT = 0.8        # 訓練集比例
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 相對路徑設定
DATA_DIR = os.path.join(SCRIPT_DIR, "dataset")
MODEL_DIR = os.path.join(SCRIPT_DIR, "..", "models")
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "coin_classifier.pth")

# 類別名稱
CLASS_NAMES = ["heads", "tails"]  # 正面, 反面


# ============== 自訂資料集類別 ==============
class CoinDataset(Dataset):
    """
    硬幣資料集
    自動處理不同大小的圖片
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(CLASS_NAMES)}

        # 掃描所有圖片
        for class_name in CLASS_NAMES:
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"警告: 資料夾不存在 {class_dir}")
                continue

            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
                    filepath = os.path.join(class_dir, filename)
                    self.samples.append((filepath, self.class_to_idx[class_name]))

        print(f"載入資料集: {len(self.samples)} 張圖片")
        for cls in CLASS_NAMES:
            count = sum(1 for s in self.samples if s[1] == self.class_to_idx[cls])
            print(f"  {cls}: {count} 張")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]

        # 載入圖片 (自動處理不同大小)
        image = Image.open(filepath).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


# ============== 資料增強與預處理 ==============
def get_transforms():
    """取得訓練和驗證的資料轉換"""

    # 訓練資料轉換 (包含資料增強)
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # 稍微放大
        transforms.RandomCrop(IMAGE_SIZE),                      # 隨機裁切
        transforms.RandomHorizontalFlip(),                      # 隨機水平翻轉
        transforms.RandomVerticalFlip(),                        # 隨機垂直翻轉
        transforms.RandomRotation(30),                          # 隨機旋轉
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # 驗證資料轉換 (不做資料增強)
    val_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


# ============== 模型定義 ==============
class CoinCNN(nn.Module):
    """硬幣分類 CNN 模型"""

    def __init__(self, num_classes=2):
        super(CoinCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4: 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5: 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
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


def get_pretrained_model(num_classes=2):
    """取得預訓練模型 (MobileNetV2 - 較輕量)"""
    model = models.mobilenet_v2(weights=models.MobileNetV2_Weights.IMAGENET1K_V1)

    # 凍結預訓練層
    for param in model.parameters():
        param.requires_grad = False

    # 替換分類器
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, num_classes)
    )

    return model


# ============== 訓練函數 ==============
def train_one_epoch(model, train_loader, optimizer, criterion, epoch):
    """訓練一個 epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for data, target in pbar:
        data, target = data.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total


def evaluate(model, val_loader, criterion):
    """評估模型"""
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)

            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return val_loss / len(val_loader), 100. * correct / total


# ============== 視覺化 ==============
def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """繪製訓練歷史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs_range = range(1, len(train_losses) + 1)

    ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs_range, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(epochs_range, train_accs, 'b-', label='Train Accuracy')
    ax2.plot(epochs_range, val_accs, 'r-', label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"訓練歷史已儲存至 {save_path}")


def visualize_predictions(val_loader, model, class_names, save_path):
    """視覺化預測結果"""
    model.eval()

    data, target = next(iter(val_loader))
    data, target = data.to(DEVICE), target.to(DEVICE)

    with torch.no_grad():
        output = model(data)
        _, predicted = output.max(1)

    # 反正規化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    num_images = min(8, len(data))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))

    for i, ax in enumerate(axes.flat):
        if i >= num_images:
            ax.axis('off')
            continue

        img = data[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = img.clip(0, 1)

        ax.imshow(img)

        pred_label = class_names[predicted[i]]
        true_label = class_names[target[i]]
        color = 'green' if predicted[i] == target[i] else 'red'

        ax.set_title(f'Pred: {pred_label}\nTrue: {true_label}', color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"預測範例已儲存至 {save_path}")


# ============== 檢查資料 ==============
def check_data():
    """檢查資料是否準備好"""
    print(f"資料目錄: {DATA_DIR}")

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR, exist_ok=True)
        for cls in CLASS_NAMES:
            os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)
        print("\n" + "=" * 50)
        print("請將圖片放入以下資料夾:")
        print(f"  正面 (heads): {os.path.join(DATA_DIR, 'heads')}")
        print(f"  反面 (tails): {os.path.join(DATA_DIR, 'tails')}")
        print("=" * 50)
        return False

    # 檢查每個類別的圖片數量
    total = 0
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.exists(cls_dir):
            count = len([f for f in os.listdir(cls_dir)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))])
            print(f"  {cls}: {count} 張")
            total += count
        else:
            os.makedirs(cls_dir, exist_ok=True)
            print(f"  {cls}: 0 張 (已建立資料夾)")

    if total < 10:
        print("\n" + "=" * 50)
        print("警告: 圖片數量太少，建議每類至少 20 張以上")
        print("=" * 50)
        return False

    return True


# ============== 主程式 ==============
def main():
    print("=" * 50)
    print("硬幣正反面分類器 - CNN 訓練")
    print("=" * 50)
    print(f"使用裝置: {DEVICE}")
    print(f"腳本目錄: {SCRIPT_DIR}")
    print()

    # 確保模型目錄存在
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 檢查資料
    print("[1] 檢查資料...")
    if not check_data():
        return
    print()

    # 載入資料集
    print("[2] 載入資料集...")
    train_transform, val_transform = get_transforms()

    full_dataset = CoinDataset(DATA_DIR, transform=train_transform)

    if len(full_dataset) == 0:
        print("錯誤: 沒有找到任何圖片")
        return

    # 分割訓練集和驗證集
    train_size = int(TRAIN_SPLIT * len(full_dataset))
    val_size = len(full_dataset) - train_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 為驗證集設定不同的轉換
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"訓練集: {len(train_dataset)} 張")
    print(f"驗證集: {len(val_dataset)} 張")
    print()

    # 建立模型
    print("[3] 建立模型...")
    # 使用自定義 CNN (也可以改用 get_pretrained_model())
    model = CoinCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    # model = get_pretrained_model(num_classes=len(CLASS_NAMES)).to(DEVICE)
    print(f"模型參數量: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # 定義損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 訓練歷史
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    best_val_acc = 0.0

    # 開始訓練
    print("[4] 開始訓練...")
    print("-" * 50)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, epoch
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"         Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        # 儲存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'class_names': CLASS_NAMES,
                'val_acc': val_acc,
                'image_size': IMAGE_SIZE,
            }, MODEL_SAVE_PATH)
            print(f"         >> 儲存最佳模型 (Val Acc: {val_acc:.2f}%)")

        print()

    print("-" * 50)
    print()

    # 視覺化
    print("[5] 視覺化結果...")
    history_path = os.path.join(SCRIPT_DIR, "training_history.png")
    samples_path = os.path.join(SCRIPT_DIR, "prediction_samples.png")

    plot_training_history(train_losses, train_accs, val_losses, val_accs, history_path)

    if len(val_dataset) > 0:
        visualize_predictions(val_loader, model, CLASS_NAMES, samples_path)

    print()
    print("=" * 50)
    print("訓練完成!")
    print(f"最佳驗證準確率: {best_val_acc:.2f}%")
    print(f"模型已儲存至: {MODEL_SAVE_PATH}")
    print("=" * 50)


if __name__ == "__main__":
    main()
