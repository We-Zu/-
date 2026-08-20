import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data_utils import DetectionDataset, get_transform
from models import DetectionCNN
from utils import save_weights_npy, load_weights_npy
from config import NUM_DETECT_CLASSES, BATCH_SIZE, EPOCHS, LEARNING_RATE, WEIGHT_DIR, DATA_DETECTION, WEIGHT_DETECTION, NUM_ANCHORS, REG_LOSS_WEIGHT, GRID_SIZE, LR_STEP_SIZE, LR_GAMMA, POSITIVE_WEIGHT, BACKGROUND_WEIGHT, USE_FOCAL_LOSS, FOCAL_ALPHA, FOCAL_GAMMA, CHECKPOINT_INTERVAL, USE_DATA_AUGMENTATION



def train():
    dataset = DetectionDataset(DATA_DETECTION, 'annotations.json', 
                                   transform=get_transform(train=USE_DATA_AUGMENTATION))   # 默认 num_anchors=NUM_ANCHORS, grid_size=GRID_SIZE
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 打印数据集信息
    print(f"检测数据集加载完成，共 {len(dataset)} 张图片可用于训练。")
    if len(dataset) == 0:
        print("警告：数据集中没有图片，请检查数据集路径和标注文件。")
        return

    model = DetectionCNN(num_classes=NUM_DETECT_CLASSES, num_anchors=NUM_ANCHORS)   # grid_size 默认使用 GRID_SIZE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---------- 续训：如果存在检测权重则加载 ----------
    weight_dir = os.path.join(WEIGHT_DIR, WEIGHT_DETECTION)
    if os.path.exists(weight_dir) and os.path.exists(os.path.join(weight_dir, 'det.pth')):
        print(f"检测到已有检测权重 ({weight_dir})，加载续训...")
        load_weights_npy(model, weight_dir, prefix='det')
    else:
        print("未检测到检测权重，从头开始训练...")

    # 使用 Focal Loss，不添加类别权重
    # 使用类别权重交叉熵：背景权重1，所有正类权重10
    class_weights = torch.ones(NUM_DETECT_CLASSES, device=device) * BACKGROUND_WEIGHT
    class_weights[1:] = POSITIVE_WEIGHT
    
    if USE_FOCAL_LOSS:
        # 定义Focal Loss（带类别权重和alpha）
        class FocalLossWithWeight(nn.Module):
            def __init__(self, alpha=0.25, gamma=2.0, weight=None):
                super().__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.weight = weight
            def forward(self, inputs, targets):
                ce_loss = nn.CrossEntropyLoss(weight=self.weight, reduction='none')(inputs, targets)
                pt = torch.exp(-ce_loss)
                focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
                return focal_loss.mean()
        cls_criterion = FocalLossWithWeight(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, weight=class_weights)
    else:
        cls_criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=-1)
    reg_criterion = nn.SmoothL1Loss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        epoch_pos_acc_sum = 0.0
        epoch_pos_count = 0

        for imgs, cls_target, reg_target, obj_mask in loader:
            imgs = imgs.to(device)
            cls_target = cls_target.to(device)
            reg_target = reg_target.to(device)
            obj_mask = obj_mask.to(device)

            optimizer.zero_grad()
            cls_pred, reg_pred = model(imgs)

            B, A, G, G, C = cls_pred.shape
            cls_pred_flat = cls_pred.reshape(-1, C)
            cls_target_flat = cls_target.reshape(-1)

            cls_loss = cls_criterion(cls_pred_flat, cls_target_flat)

            # 累积正样本准确率
            with torch.no_grad():
                pred_classes = cls_pred_flat.argmax(dim=-1)
                pos_mask = obj_mask.reshape(-1)
                if pos_mask.sum() > 0:
                    batch_acc = (pred_classes[pos_mask] == cls_target_flat[pos_mask]).float().mean()
                    epoch_pos_acc_sum += batch_acc.item()
                    epoch_pos_count += 1

            reg_loss = reg_criterion(reg_pred[obj_mask], reg_target[obj_mask])
            loss = cls_loss + REG_LOSS_WEIGHT * reg_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        avg_pos_acc = epoch_pos_acc_sum / epoch_pos_count if epoch_pos_count > 0 else 0.0
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}, 正样本平均准确率: {avg_pos_acc:.4f}')
        scheduler.step()
        
        # ---------- 保存检查点 ----------
        if CHECKPOINT_INTERVAL > 0 and (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            checkpoint_dir = os.path.join(WEIGHT_DIR, WEIGHT_DETECTION)
            os.makedirs(checkpoint_dir, exist_ok=True)
            # 保存当前 epoch 检查点
            save_weights_npy(model, checkpoint_dir, prefix=f'det_epoch{epoch+1}')
            # 同时更新不带 epoch 的 det 权重，以便续训加载
            save_weights_npy(model, checkpoint_dir, prefix='det')
            # 删除之前的历史检查点（保留当前最新的这个）
            import glob
            pattern = os.path.join(checkpoint_dir, 'det_epoch*.pth')
            for f in glob.glob(pattern):
                # 不删除当前保存的文件（包含当前 epoch）
                if not f.endswith(f'det_epoch{epoch+1}.pth'):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            print(f"  检查点已保存: epoch {epoch+1} (并更新det权重，已清理旧检查点)")

    save_weights_npy(model, os.path.join(WEIGHT_DIR, WEIGHT_DETECTION), prefix='det')
    print("Detection training finished.")
    # 在 save_weights_npy 之后添加
    state_dict = model.state_dict()
    print(f"模型参数总数: {len(state_dict)} 个张量")
    print(f"骨干网络参数数: {sum(1 for k in state_dict.keys() if 'backbone' in k)}")

if __name__ == '__main__':
    train()