import os
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from config import IMAGE_SIZE, DATASET_DIR, ANCHORS, IOU_POSITIVE_THRESHOLD, GRID_SIZE, CLASS_NAMES

# ---------- 通用图像变换 ----------
def get_transform(train=True):
    t = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    if train:
        # 增加更多增强
        from torchvision.transforms import RandomRotation, ColorJitter
        t.insert(0, transforms.RandomRotation(10))
        t.insert(0, transforms.ColorJitter(brightness=0.2, contrast=0.2))
        t.insert(0, transforms.RandomHorizontalFlip())
    return transforms.Compose(t)

# ---------- 分类数据集 ----------
class ClassificationDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = os.path.join(DATASET_DIR, root)
        self.transform = transform
# 确保文件夹名称与 CLASS_NAMES 一致，按配置顺序建立索引
        self.classes = CLASS_NAMES
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(self.root, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

# ---------- 检测目标编码器 ----------
from config import NUM_ANCHORS   # 顶部导入
class DetectionTargetEncoder:
    def __init__(self, grid_size=GRID_SIZE, num_anchors=NUM_ANCHORS, anchor_boxes=None):
        self.grid_size = grid_size
        self.num_anchors = num_anchors
        if anchor_boxes is None:
            # 使用 config 中的 ANCHORS（归一化值）
            self.anchor_boxes = torch.tensor(ANCHORS, dtype=torch.float32)
        else:
            self.anchor_boxes = torch.tensor(anchor_boxes, dtype=torch.float32)
        assert self.anchor_boxes.shape[0] == num_anchors

    def encode(self, boxes, labels):
        G = self.grid_size
        A = self.num_anchors
        cls_target = torch.zeros((A, G, G), dtype=torch.long)
        reg_target = torch.zeros((A, G, G, 4), dtype=torch.float32)
        obj_mask = torch.zeros((A, G, G), dtype=torch.bool)
        max_iou = torch.zeros((A, G, G), dtype=torch.float32)  # 记录每个锚框位置的最大IoU，用于解决冲突

        if boxes.numel() == 0:
            return cls_target, reg_target, obj_mask

        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]

        for i in range(boxes.size(0)):
            # 计算GT所在的网格坐标
            gi = int((cx[i] * G).clamp(0, G-1).item())
            gj = int((cy[i] * G).clamp(0, G-1).item())
            # 相对于网格中心的偏移
            tx = cx[i] * G - gi
            ty = cy[i] * G - gj
            
            anchor_boxes = self.anchor_boxes
            # 计算该GT与所有锚框的IoU
            inter_w = torch.min(w[i], anchor_boxes[:, 0])
            inter_h = torch.min(h[i], anchor_boxes[:, 1])
            inter_area = inter_w * inter_h
            union_area = w[i]*h[i] + anchor_boxes[:,0]*anchor_boxes[:,1] - inter_area
            ious = inter_area / (union_area + 1e-6)  # [A]
            
            # 找到所有IoU大于阈值的锚框
            mask = ious > IOU_POSITIVE_THRESHOLD
            if not mask.any():
                continue

            for a in range(A):
                if mask[a]:
                    # 如果当前IoU大于该锚框位置已有的IoU，则更新（保留IoU最大的GT）
                    if ious[a] > max_iou[a, gj, gi]:
                        max_iou[a, gj, gi] = ious[a]
                        tw = torch.log(w[i] / anchor_boxes[a, 0] + 1e-6)
                        th = torch.log(h[i] / anchor_boxes[a, 1] + 1e-6)
                        cls_target[a, gj, gi] = labels[i]
                        reg_target[a, gj, gi] = torch.tensor([tx, ty, tw, th])
                        obj_mask[a, gj, gi] = True

        return cls_target, reg_target, obj_mask

# ---------- 检测数据集 ----------
class DetectionDataset(Dataset):
    def __init__(self, root, ann_file, transform=None, grid_size=GRID_SIZE, num_anchors=NUM_ANCHORS):
        self.root = os.path.join(DATASET_DIR, root, 'images')   # 注意这里增加了 'images'
        with open(os.path.join(DATASET_DIR, root, ann_file), 'r', encoding='utf-8') as f:
            self.coco = json.load(f)
        self.transform = transform
        self.images = {img['id']: img for img in self.coco['images']}
        self.annotations = {}
        for ann in self.coco['annotations']:
            self.annotations.setdefault(ann['image_id'], []).append(ann)
        self.ids = list(self.images.keys())
        self.encoder = DetectionTargetEncoder(grid_size=grid_size, num_anchors=num_anchors)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.root, img_info['file_name'])
        img = Image.open(img_path).convert('RGB')
        w, h = img_info['width'], img_info['height']

        boxes = []
        labels = []
        for ann in self.annotations.get(img_id, []):
            x, y, w_, h_ = ann['bbox']
            x1 = x / w
            y1 = y / h
            x2 = (x + w_) / w
            y2 = (y + h_) / h
            boxes.append([x1, y1, x2, y2])
            labels.append(ann['category_id'])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        cls_target, reg_target, obj_mask = self.encoder.encode(boxes, labels)

        if self.transform:
            img = self.transform(img)

        return img, cls_target, reg_target, obj_mask

# ---------- 分割数据集 ----------
class SegmentationDataset(Dataset):
    def __init__(self, root, transform=None, mask_transform=None):
        self.root = os.path.join(DATASET_DIR, root)
        self.img_dir = os.path.join(self.root, 'images')
        self.mask_dir = os.path.join(self.root, 'masks')
        self.fnames = [f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform
        self.mask_transform = mask_transform or transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=Image.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert('RGB')
        mask = Image.open(os.path.join(self.mask_dir, fname.replace('.jpg', '.png'))).convert('L')
        if self.transform:
            img = self.transform(img)
        mask = self.mask_transform(mask)
        mask = mask.squeeze(0).long()
        return img, mask