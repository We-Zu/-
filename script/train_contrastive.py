import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import deque
from data_utils import ClassificationDataset
from models import UNet
from utils import save_weights_npy, load_weights_npy
from config import (
    WEIGHT_DIR, BATCH_SIZE, CONTRASTIVE_EPOCHS, CONTRASTIVE_LR,
    IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE, PROJECTION_DIM, FEATURE_DIM, CONTRASTIVE_TEMPERATURE,
    DATASET_DIR, DATA_CLASSIFICATION, WEIGHT_CONTRASTIVE,
    CONTRASTIVE_BATCH_SIZE, CONTRASTIVE_CHECKPOINT_BATCH_INTERVAL, NUM_WORKERS,
    CONTRASTIVE_WEIGHT_DECAY, CONTRASTIVE_LR_SCHEDULER, CONTRASTIVE_LR_STEP_SIZE, CONTRASTIVE_LR_GAMMA,
    CONTRASTIVE_ENHANCE_PAIRS, CONTRASTIVE_ENHANCE_WEIGHT
)

# ---------- 投影头 ----------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim=FEATURE_DIM, proj_dim=PROJECTION_DIM):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, proj_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ---------- 对比学习数据集 ----------
class ContrastiveDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ClassificationDataset(root, train=True, transform=None)
        self.transform = transform or get_contrastive_transform()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        view1 = self.transform(img)
        view2 = self.transform(img)
        return view1, view2, label

# ---------- 对比学习数据增强（简化版，避免卡顿）----------
def get_contrastive_transform():
    """对比学习专用增强（简化版，用于快速调试）"""
    return transforms.Compose([
        transforms.Resize((CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE)),          # 直接缩放，避免 RandomResizedCrop 的随机裁剪开销
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=1.0, contrast=1.0, saturation=1.0, hue=0.05),
        transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 0.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# ---------- 监督对比损失 ----------
def supcon_loss(features, labels, temperature=0.1, enhance_pairs=None, enhance_weight=2.0):
    N = features.size(0)
    sim_matrix = features @ features.T
    mask = torch.eye(N, device=features.device).bool()
    exp_sim = torch.exp(sim_matrix / temperature)
    # 正样本掩码（标签相同），排除自身
    positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~mask
    # 构建权重矩阵（默认全1）
    weight_matrix = torch.ones_like(exp_sim)
    if enhance_pairs is not None and len(enhance_pairs) > 0:
        label_tensor = labels
        for (cls_a, cls_b) in enhance_pairs:
            mask_a = (label_tensor == cls_a)
            mask_b = (label_tensor == cls_b)
            # 使用正确的二维掩码
            mask_ab = mask_a.unsqueeze(1) & mask_b.unsqueeze(0)
            mask_ba = mask_b.unsqueeze(1) & mask_a.unsqueeze(0)
            # 将增强类别对之间的权重设为 enhance_weight
            weight_matrix[mask_ab] = enhance_weight
            weight_matrix[mask_ba] = enhance_weight
    # 正样本位置权重设为1，自身权重设为0
    weight_matrix[positive_mask] = 1.0
    weight_matrix[mask] = 0.0
    # 加权后的相似度指数
    weighted_exp_sim = exp_sim * weight_matrix
    # 正样本指数（不加权）
    positive_exp = exp_sim * positive_mask.float()
    positive_sum = positive_exp.sum(dim=1)
    total_weighted_sum = weighted_exp_sim.sum(dim=1)
    negative_weighted_sum = total_weighted_sum - positive_sum
    loss = -torch.log(positive_sum / (positive_sum + negative_weighted_sum + 1e-8))
    return loss.mean()

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据集
    dataset = ContrastiveDataset(DATA_CLASSIFICATION, transform=get_contrastive_transform())
    print(f"数据集大小: {len(dataset)} 张图片")
    if len(dataset) == 0:
        print("错误: 数据集为空，请检查路径！")
        return

    # 获取所有样本的标签
    labels = [dataset.dataset.samples[i][1] for i in range(len(dataset))]
    
    # 计算每个类别的权重（类别数 = len(CLASS_NAMES)，但实际可能有些类为空）
    class_counts = torch.bincount(torch.tensor(labels))
    # 避免除以零，将空类别的权重设为0（或极小值）
    class_weights = 1.0 / (class_counts.float() + 1e-8)
    sample_weights = class_weights[labels]
    
    # 创建加权采样器，replacement=True 表示有放回采样
    sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # DataLoader 使用 sampler，此时不能使用 shuffle=True
    loader = DataLoader(dataset, batch_size=CONTRASTIVE_BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS)

    # 模型
    backbone = UNet().to(device)
    # 由于 extract_features 返回 avg+max 拼接特征，维度为 2*FEATURE_DIM
    projection = ProjectionHead(input_dim=2 * FEATURE_DIM, proj_dim=PROJECTION_DIM).to(device)

    # ---------- 续训：如果存在对比权重则加载 ----------
    weight_dir = os.path.join(WEIGHT_DIR, WEIGHT_CONTRASTIVE)
    start_epoch = 0
    global_batch_count = 0
    last_checkpoint_batch = 0
    checkpoint_state_path = os.path.join(weight_dir, 'checkpoint_state.pt')

    if os.path.exists(weight_dir) and os.path.exists(os.path.join(weight_dir, 'contrast.pth')):
        print(f"检测到已有对比权重 ({weight_dir})，加载续训...")
        load_weights_npy(backbone, weight_dir, prefix='contrast')
        # 尝试加载投影头权重（若存在则加载，否则随机初始化）
        if os.path.exists(os.path.join(weight_dir, 'proj.pth')):
            load_weights_npy(projection, weight_dir, prefix='proj')
            print("  投影头权重已加载")
        else:
            print("  未找到投影头权重，将随机初始化")
        # 检查是否有训练状态文件（将在 optimizer 定义后加载）
        if os.path.exists(checkpoint_state_path):
            print("  检测到训练状态文件，将在优化器定义后恢复")
        else:
            print("  未找到训练状态文件，将从头开始")
    else:
        print("未检测到对比权重，从头开始训练...")

    optimizer = optim.Adam(list(backbone.parameters()) + list(projection.parameters()), 
                           lr=CONTRASTIVE_LR, weight_decay=CONTRASTIVE_WEIGHT_DECAY)
    # 学习率调度器
    if CONTRASTIVE_LR_SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONTRASTIVE_EPOCHS, eta_min=1e-9)
    else:  # 默认使用 StepLR
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=CONTRASTIVE_LR_STEP_SIZE, gamma=CONTRASTIVE_LR_GAMMA)

    # ---------- 加载优化器、调度器状态（如果存在） ----------
    if os.path.exists(checkpoint_state_path):
        checkpoint_state = torch.load(checkpoint_state_path, map_location='cpu')
        optimizer.load_state_dict(checkpoint_state['optimizer'])
        scheduler.load_state_dict(checkpoint_state['scheduler'])
        global_batch_count = checkpoint_state.get('global_batch_count', 0)
        last_checkpoint_batch = checkpoint_state.get('last_checkpoint_batch', 0)
        start_epoch = checkpoint_state.get('epoch', 0) + 1  # 从下一epoch开始
        print(f"  恢复训练状态: epoch={start_epoch}, batch_count={global_batch_count}")
    # 注意：如果不存在，global_batch_count 和 last_checkpoint_batch 保持之前定义的 0

    # 在 epoch 循环外定义历史队列，跨 epoch 累积 batch 数据
    batch_history = deque(maxlen=500)
    
    for epoch in range(start_epoch, CONTRASTIVE_EPOCHS):
        total_loss = 0.0
        total_acc = 0.0
        total_pos_sim = 0.0
        total_neg_sim = 0.0
        pos_batches = 0
        neg_batches = 0
        # 注意：不再在这里创建 batch_history，使用外部定义
        # 同时保留一个列表用于快速计算窗口10和100的平均（可复用）
        # 但我们直接每次遍历 deque 末尾的 N 个元素，简单可靠

        for batch_idx, (view1, view2, labels) in enumerate(loader):
            view1, view2, labels = view1.to(device), view2.to(device), labels.to(device)

            feat1 = backbone.extract_features(view1)
            feat2 = backbone.extract_features(view2)

            proj1 = projection(feat1)
            proj2 = projection(feat2)
            features = torch.cat([proj1, proj2], dim=0)
            labels_pair = torch.cat([labels, labels], dim=0)
            features = F.normalize(features, dim=1)

            # 将增强类别对从名称转换为索引
            enhance_pairs_indices = []
            if CONTRASTIVE_ENHANCE_PAIRS:
                class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}
                for pair in CONTRASTIVE_ENHANCE_PAIRS:
                    if len(pair) == 2:
                        idx_a = class_to_idx.get(pair[0])
                        idx_b = class_to_idx.get(pair[1])
                        if idx_a is not None and idx_b is not None:
                            enhance_pairs_indices.append((idx_a, idx_b))
            loss = supcon_loss(features, labels_pair, temperature=CONTRASTIVE_TEMPERATURE,
                               enhance_pairs=enhance_pairs_indices if enhance_pairs_indices else None,
                               enhance_weight=CONTRASTIVE_ENHANCE_WEIGHT)

            # 计算对比准确率（正样本对最近邻检索准确率）
            with torch.no_grad():
                sim_matrix = features @ features.T  # [2B, 2B]
                # 排除自身：将自身相似度设为 -inf
                mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
                sim_matrix_masked = sim_matrix.clone()
                sim_matrix_masked[mask] = -float('inf')
                max_indices = sim_matrix_masked.argmax(dim=1)
                pred_labels = labels_pair[max_indices]
                acc = (pred_labels == labels_pair).float().mean()
                total_acc += acc.item()

                # --- 新增：计算正负样本对的平均相似度 ---
                # 正样本掩码（标签相同，且排除自身）
                pos_mask = (labels_pair.unsqueeze(1) == labels_pair.unsqueeze(0)) & ~mask
                # 负样本掩码（标签不同）
                neg_mask = (labels_pair.unsqueeze(1) != labels_pair.unsqueeze(0))
                # 提取相似度
                pos_sim = sim_matrix[pos_mask]
                neg_sim = sim_matrix[neg_mask]
                # 计算平均（忽略空集）
                if pos_sim.numel() > 0:
                    pos_sim_mean = pos_sim.mean().item()
                    total_pos_sim += pos_sim_mean
                    pos_batches += 1
                if neg_sim.numel() > 0:
                    neg_sim_mean = neg_sim.mean().item()
                    total_neg_sim += neg_sim_mean
                    neg_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            global_batch_count += 1
            if (global_batch_count - last_checkpoint_batch) >= CONTRASTIVE_CHECKPOINT_BATCH_INTERVAL:
                checkpoint_dir = os.path.join(WEIGHT_DIR, WEIGHT_CONTRASTIVE)
                os.makedirs(checkpoint_dir, exist_ok=True)
                # 保存当前 batch 检查点
                save_weights_npy(backbone, checkpoint_dir, prefix=f'contrast_batch{global_batch_count}')
                save_weights_npy(projection, checkpoint_dir, prefix=f'proj_batch{global_batch_count}')
                save_weights_npy(backbone, checkpoint_dir, prefix='contrast')
                save_weights_npy(projection, checkpoint_dir, prefix='proj')
                # 保存优化器、调度器、计数等状态
                torch.save({
                    'epoch': epoch,
                    'global_batch_count': global_batch_count,
                    'last_checkpoint_batch': last_checkpoint_batch,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(checkpoint_dir, 'checkpoint_state.pt'))
                # 删除之前的历史 batch 检查点（保留当前最新的）
                import glob
                pattern_contrast = os.path.join(checkpoint_dir, 'contrast_batch*.pth')
                pattern_proj = os.path.join(checkpoint_dir, 'proj_batch*.pth')
                current_prefix = f'contrast_batch{global_batch_count}.pth'
                current_proj_prefix = f'proj_batch{global_batch_count}.pth'
                for f in glob.glob(pattern_contrast):
                    if not f.endswith(current_prefix):
                        try:
                            os.remove(f)
                        except OSError:
                            pass
                for f in glob.glob(pattern_proj):
                    if not f.endswith(current_proj_prefix):
                        try:
                            os.remove(f)
                        except OSError:
                            pass
                print(f"  检查点已保存: batch {global_batch_count} (并更新contrast/proj权重及训练状态，已清理旧batch检查点)")
                last_checkpoint_batch = global_batch_count

            # 计算当前 batch 的指标
            batch_pos_sim = pos_sim_mean if pos_sim.numel() > 0 else 0.0
            batch_neg_sim = neg_sim_mean if neg_sim.numel() > 0 else 0.0
            batch_sep_gap = batch_pos_sim - batch_neg_sim
            current_loss = loss.item()
            current_acc = acc.item()
            # 将当前 batch 的数据存入历史队列
            batch_history.append((current_loss, current_acc, batch_pos_sim, batch_neg_sim, batch_sep_gap))
            # 调试：打印当前队列长度（跨epoch累积）
            if (batch_idx + 1) % 10 == 0:
                print(f"    队列长度: {len(batch_history)} (当前epoch {epoch+1}, batch {batch_idx+1})")

            # 每 10 个 batch 打印一次三个窗口的平均值
            if (batch_idx + 1) % 10 == 0:
                # 从历史队列中取最近的10、100、500个batch
                history_list = list(batch_history)
                n_total = len(history_list)
                
                # 定义辅助函数计算窗口平均
                def calc_window(n):
                    if n_total == 0 or n == 0:
                        return (0.0, 0.0, 0.0, 0.0, 0.0)
                    take = min(n, n_total)
                    recent = history_list[-take:]
                    avg_loss = sum(x[0] for x in recent) / take
                    avg_acc = sum(x[1] for x in recent) / take
                    avg_pos_sim = sum(x[2] for x in recent) / take
                    avg_neg_sim = sum(x[3] for x in recent) / take
                    avg_sep_gap = sum(x[4] for x in recent) / take
                    return avg_loss, avg_acc, avg_pos_sim, avg_neg_sim, avg_sep_gap
                
                avg10 = calc_window(10)
                avg100 = calc_window(100)
                avg500 = calc_window(500)
                
                print(f"  Epoch {epoch+1}, Batch {batch_idx+1}/{len(loader)}")
                print(f"    [10 -batch] Loss: {avg10[0]:.4f}, Acc: {avg10[1]:.4f}, PosSim: {avg10[2]:.4f}, NegSim: {avg10[3]:.4f}, SepGap: {avg10[4]:.4f}")
                print(f"    [100-batch] Loss: {avg100[0]:.4f}, Acc: {avg100[1]:.4f}, PosSim: {avg100[2]:.4f}, NegSim: {avg100[3]:.4f}, SepGap: {avg100[4]:.4f}")
                print(f"    [500-batch] Loss: {avg500[0]:.4f}, Acc: {avg500[1]:.4f}, PosSim: {avg500[2]:.4f}, NegSim: {avg500[3]:.4f}, SepGap: {avg500[4]:.4f}")

        avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
        avg_acc = total_acc / len(loader) if len(loader) > 0 else 0.0
        avg_pos_sim = total_pos_sim / pos_batches if pos_batches > 0 else 0.0
        avg_neg_sim = total_neg_sim / neg_batches if neg_batches > 0 else 0.0
        sep_gap = avg_pos_sim - avg_neg_sim  # 分离度，越大越好

        print(f"Epoch {epoch+1}/{CONTRASTIVE_EPOCHS}, Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, "
              f"PosSim: {avg_pos_sim:.4f}, NegSim: {avg_neg_sim:.4f}, SepGap: {sep_gap:.4f}")
        scheduler.step()   # 更新学习率
        


    # 保存权重（同时保存 backbone 和 projection）
    save_weights_npy(backbone, os.path.join(WEIGHT_DIR, WEIGHT_CONTRASTIVE), prefix='contrast')
    save_weights_npy(projection, os.path.join(WEIGHT_DIR, WEIGHT_CONTRASTIVE), prefix='proj')
    print("Contrastive learning finished, weights saved.")

if __name__ == '__main__':
    train()