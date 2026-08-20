import os
import torch
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils import ClassificationDataset
from models import UNet
from utils import load_weights_npy
from config import (
    WEIGHT_DIR, CLASS_NAMES, FEATURE_DIM, PROXY_CLASSIFIER,
    DATASET_DIR, BATCH_SIZE, DATA_CLASSIFICATION, WEIGHT_CONTRASTIVE, WEIGHT_PROXY,
    CONTRASTIVE_IMAGE_SIZE
)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载预训练的特征提取器（UNet编码器）
    backbone = UNet().to(device)
    weight_path = os.path.join(WEIGHT_DIR, WEIGHT_CONTRASTIVE)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Contrastive weights not found: {weight_path}")
    load_weights_npy(backbone, weight_path, prefix='contrast')
    backbone.eval()

    # 冻结backbone
    for param in backbone.parameters():
        param.requires_grad = False

    # 准备数据集（分类数据集，单细胞图）
    # 使用 CONTRASTIVE_IMAGE_SIZE 构建专用变换（无数据增强，仅缩放和归一化）
    contrastive_transform = transforms.Compose([
        transforms.Resize((CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = ClassificationDataset(DATA_CLASSIFICATION, train=True, transform=contrastive_transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 提取特征
    features_list = []
    labels_list = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            feats = backbone.extract_features(imgs)  # [B, FEATURE_DIM]
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)

    # 标准化特征（可选）
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 训练代理AI
    if PROXY_CLASSIFIER == 'linear':
        clf = LogisticRegression(max_iter=1000)
    elif PROXY_CLASSIFIER == 'svm':
        # 将线性核改为RBF核，并适当调整gamma，捕捉非线性边界
        clf = SVC(kernel='rbf', gamma='scale', probability=True, C=10.0)
    else:
        raise ValueError(f"Unknown classifier: {PROXY_CLASSIFIER}")

    clf.fit(features_scaled, labels)
    acc = clf.score(features_scaled, labels)
    print(f"Proxy classifier training accuracy: {acc:.4f}")

    # 保存代理AI和scaler
    import joblib
    proxy_dir = os.path.join(WEIGHT_DIR, WEIGHT_PROXY)
    os.makedirs(proxy_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(proxy_dir, 'classifier.pkl'))
    joblib.dump(scaler, os.path.join(proxy_dir, 'scaler.pkl'))
    print("Proxy classifier saved.")

if __name__ == '__main__':
    train()