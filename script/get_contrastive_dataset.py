"""
批量推理脚本（精简版）- 仅保留检测、裁剪、面积统计、结果可视化
功能：
1. 检测AI找到所有细胞，裁剪并保存单细胞图
2. 计算所有检测到的细胞的面积正态分布统计（均值、标准差）
3. 绘制带框和标签的结果图像（.png）
4. 不生成文本报告，不交互（默认计算面积统计）
"""

import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import joblib
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import get_transform
from models import DetectionCNN, UNet
from utils import load_weights_npy
from config import (
    WEIGHT_DIR, CLASS_NAMES, NUM_DETECT_CLASSES,
    IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE, FEATURE_DIM, WEIGHT_DETECTION, WEIGHT_CONTRASTIVE, WEIGHT_PROXY,
    ANCHORS, NUM_ANCHORS, CONF_THRESH, NMS_IOU_THRESH, AREA_MIN, AREA_MAX, GRID_SIZE
)
from torchvision import transforms

# ---------- 配置 ----------
INPUT_DIR = Path(__file__).parent.parent / "input"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

# ---------- 辅助函数（检测解码、NMS） ----------
def decode_predictions(cls_pred, reg_pred, anchor_boxes, grid_size, conf_thresh=0.8):
    if cls_pred.dim() == 5:
        cls_pred = cls_pred.squeeze(0)
        reg_pred = reg_pred.squeeze(0)
    A, G, G, C = cls_pred.shape
    
    probs = cls_pred.softmax(dim=-1)
    max_probs, max_indices = torch.max(probs, dim=-1)
    labels = max_indices
    
    mask = max_probs > conf_thresh
    if mask.sum() == 0:
        return torch.empty(0, 4), torch.empty(0), torch.empty(0)
    
    anchor_idx, gy, gx = torch.where(mask)
    
    tx = reg_pred[anchor_idx, gy, gx, 0]
    ty = reg_pred[anchor_idx, gy, gx, 1]
    tw = reg_pred[anchor_idx, gy, gx, 2]
    th = reg_pred[anchor_idx, gy, gx, 3]
    
    anchor_w = anchor_boxes[anchor_idx, 0]
    anchor_h = anchor_boxes[anchor_idx, 1]
    
    cx = (gx.float() + tx) / G
    cy = (gy.float() + ty) / G
    w = torch.exp(torch.clamp(tw, max=10)) * anchor_w
    h = torch.exp(torch.clamp(th, max=10)) * anchor_h
    
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2
    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    scores = max_probs[mask]
    labels = labels[mask]
    
    return boxes, scores, labels

def nms(boxes, scores, iou_thresh=0.1):
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort(descending=True)
    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        inter = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = torch.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return torch.tensor(keep)

# ---------- 主函数 ----------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    single_cell_dir = OUTPUT_DIR / "single_cell"
    single_cell_dir.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        print(f"错误: input 目录不存在: {INPUT_DIR}")
        return

    image_paths = list(set(
        [p for ext in SUPPORTED_FORMATS for p in INPUT_DIR.glob(f"*{ext}")] +
        [p for ext in SUPPORTED_FORMATS for p in INPUT_DIR.glob(f"*{ext.upper()}")] 
    ))
    if not image_paths:
        print(f"在 {INPUT_DIR} 中未找到支持的图片文件")
        return

    print(f"找到 {len(image_paths)} 张图片，开始推理...")
    print("将自动收集面积数据并计算全局统计，同时裁剪并保存单细胞图。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = get_transform(train=False)
    cell_transform = transforms.Compose([
        transforms.Resize((CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载字体
    try:
        font = ImageFont.truetype("simhei.ttf", 20)
    except:
        try:
            font = ImageFont.truetype("msyh.ttc", 20)
        except:
            font = ImageFont.load_default()

    # 加载检测模型
    print("加载检测模型...")
    det_model = DetectionCNN(num_classes=NUM_DETECT_CLASSES, num_anchors=NUM_ANCHORS)
    det_weight_dir = os.path.join(WEIGHT_DIR, WEIGHT_DETECTION)
    if not os.path.exists(det_weight_dir):
        print("错误: 检测权重不存在")
        return
    load_weights_npy(det_model, det_weight_dir, prefix='det')
    det_model.to(device)
    det_model.eval()

    # 加载特征提取器
    print("加载分割特征提取器...")
    backbone = UNet()
    backbone_weight_dir = os.path.join(WEIGHT_DIR, WEIGHT_CONTRASTIVE)
    if not os.path.exists(backbone_weight_dir):
        print("错误: 对比学习权重不存在")
        return
    load_weights_npy(backbone, backbone_weight_dir, prefix='contrast')
    backbone.to(device)
    backbone.eval()

    # 加载代理AI
    print("加载代理AI...")
    proxy_dir = os.path.join(WEIGHT_DIR, WEIGHT_PROXY)
    if not os.path.exists(proxy_dir):
        print("错误: 代理AI权重不存在")
        return
    classifier = joblib.load(os.path.join(proxy_dir, 'classifier.pkl'))
    scaler = joblib.load(os.path.join(proxy_dir, 'scaler.pkl'))

    anchor_boxes = torch.tensor(ANCHORS, dtype=torch.float32).to(device)

    print("\n开始批量推理...")
    print("-" * 60)

    # 统计变量
    all_areas = []          # 存储所有细胞面积
    class_stats = {name: 0 for name in CLASS_NAMES}
    total_cells = 0

    for img_path in image_paths:
        print(f"\n处理: {img_path.name}")
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size

        # 检测
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            cls_pred, reg_pred = det_model(input_tensor)

        boxes, scores, labels = decode_predictions(cls_pred, reg_pred, anchor_boxes, grid_size=GRID_SIZE, conf_thresh=CONF_THRESH)

        if len(boxes) == 0:
            print("  未检测到细胞，跳过")
            continue

        # NMS
        keep_nms = nms(boxes, scores, iou_thresh=NMS_IOU_THRESH)
        boxes = boxes[keep_nms]
        scores = scores[keep_nms]
        labels = labels[keep_nms]

        # 面积过滤
        areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
        keep_area = (areas > AREA_MIN) & (areas < AREA_MAX)
        boxes = boxes[keep_area]
        scores = scores[keep_area]
        labels = labels[keep_area]

        # 过滤无效类别和背景
        labels = torch.clamp(labels, 0, NUM_DETECT_CLASSES - 1)
        keep_valid = (labels > 0) & (labels < NUM_DETECT_CLASSES)
        boxes = boxes[keep_valid]
        scores = scores[keep_valid]
        labels = labels[keep_valid]

        print(f"  检测到 {len(boxes)} 个细胞")

        # 裁剪 + 分类 + 保存
        results = []
        for i, box in enumerate(boxes):
            x1 = int(box[0].item() * orig_w)
            y1 = int(box[1].item() * orig_h)
            x2 = int(box[2].item() * orig_w)
            y2 = int(box[3].item() * orig_h)

            cell_img = img.crop((x1, y1, x2, y2))
            cell_tensor = cell_transform(cell_img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = backbone.extract_features(cell_tensor)
                feat_np = feat.cpu().numpy()
                feat_scaled = scaler.transform(feat_np)
                pred_label = classifier.predict(feat_scaled)[0]
                prob = classifier.predict_proba(feat_scaled)[0] if hasattr(classifier, 'predict_proba') else None

            cell_type = CLASS_NAMES[pred_label]
            confidence = prob.max() if prob is not None else 1.0

            # 保存单细胞图
            cell_filename = f"{img_path.stem}_cell_{i+1}_{cell_type}_{confidence:.2f}.png"
            cell_save_path = single_cell_dir / cell_filename
            cell_img.save(cell_save_path)

            area_pixel = (x2 - x1) * (y2 - y1)
            results.append({
                'bbox': [x1, y1, x2, y2],
                'class': cell_type,
                'confidence': confidence,
                'area': area_pixel
            })
            all_areas.append(area_pixel)
            class_stats[cell_type] += 1
            total_cells += 1
            print(f"    细胞 {i+1}: {cell_type} (conf={confidence:.2f}, area={area_pixel})")

        # 绘制结果图（带框和标签）
        draw = ImageDraw.Draw(img)
        for obj in results:
            x1, y1, x2, y2 = obj['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            draw.text((x1, y1-20), f"{obj['class']}: {obj['confidence']:.2f}", fill='red', font=font)

        output_img_path = OUTPUT_DIR / f"{img_path.stem}_result.png"
        img.save(output_img_path)

    # ---------- 计算全局面积统计 ----------
    print("\n" + "=" * 60)
    print("计算全局细胞的面积正态分布统计...")
    count = len(all_areas)
    if count == 0:
        print("  警告: 没有收集到任何面积数据，跳过统计。")
    else:
        min_val = int(np.min(all_areas))
        max_val = int(np.max(all_areas))
        mean = float(np.mean(all_areas))
        std = float(np.std(all_areas, ddof=1)) if count >= 2 else 0.0

        area_stats = {
            'all_cells': {
                'mean': mean,
                'std': std,
                'count': count,
                'min': min_val,
                'max': max_val
            }
        }
        print(f"  全局统计: 样本数={count}, 均值={mean:.2f}, 标准差={std:.2f}, 范围=[{min_val}, {max_val}]")

        stats_file = OUTPUT_DIR / "area_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(area_stats, f, indent=4, ensure_ascii=False)
        print(f"面积统计数据已保存至: {stats_file}")

    # 打印各类别数量
    print("\n" + "=" * 60)
    print("检测结果统计:")
    print(f"总检测细胞数: {total_cells}")
    print("各类细胞数量:")
    for name, count in class_stats.items():
        if count > 0:
            print(f"  {name}: {count}")
    print("\n" + "=" * 60)
    print(f"单细胞图保存在: {single_cell_dir}")
    print(f"结果图像保存在: {OUTPUT_DIR}")
    print("完成！")

if __name__ == '__main__':
    main()