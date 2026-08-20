"""
批量推理脚本 - 新流程：
1. 检测AI找到所有细胞，裁剪
2. 分割AI（特征提取器）提取每个细胞的特征向量
3. 代理AI分类
4. 尺寸验证：利用面积正态分布数据对初级结果进行修正
"""

import os
import sys
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path
import joblib
import json
import math
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import get_transform
from models import DetectionCNN, UNet
from utils import load_weights_npy
from config import (
    WEIGHT_DIR, CLASS_NAMES, NUM_DETECT_CLASSES,
    IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE, FEATURE_DIM, WEIGHT_DETECTION, WEIGHT_CONTRASTIVE, WEIGHT_PROXY,
    ANCHORS, NUM_ANCHORS, CONF_THRESH, NMS_IOU_THRESH, AREA_MIN, AREA_MAX, GRID_SIZE,
    SIZE_VERIFICATION_STRICTNESS
)
from torchvision import transforms

# ---------- 配置 ----------
INPUT_DIR = Path(__file__).parent.parent / "input"
OUTPUT_DIR = Path(__file__).parent.parent / "output"
SCALE_DIR = Path(__file__).parent / "scale"   # 面积统计文件存放目录
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

# ---------- 加载面积统计信息 ----------
def load_area_stats(scale_dir):
    """加载所有类别对应的面积统计（mean, std）"""
    stats = {}
    if not scale_dir.exists():
        print(f"警告: 面积统计目录不存在: {scale_dir}，将跳过尺寸验证")
        return stats
    
    for cls_name in CLASS_NAMES:
        json_file = scale_dir / f"{cls_name}.json"
        if json_file.exists():
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_cells = data.get("all_cells", {})
                    mean = all_cells.get("mean")
                    std = all_cells.get("std")
                    if mean is not None and std is not None and std > 0:
                        stats[cls_name] = {"mean": mean, "std": std}
                    else:
                        print(f"  警告: {cls_name}.json 中缺少有效的 mean/std，跳过")
                print(f"  加载 {cls_name}: mean={stats[cls_name]['mean']:.2f}, std={stats[cls_name]['std']:.2f}")
            except Exception as e:
                print(f"  错误: 加载 {cls_name}.json 失败: {e}")
        else:
            print(f"  未找到 {cls_name}.json，跳过尺寸验证")
    return stats

# ---------- 主函数 ----------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        print(f"错误: input 目录不存在: {INPUT_DIR}")
        return

    # 加载面积统计
    print("\n加载面积统计信息...")
    area_stats = load_area_stats(SCALE_DIR)
    print(f"已加载 {len(area_stats)} 个类别的面积统计")

    image_paths = list(set(
        [p for ext in SUPPORTED_FORMATS for p in INPUT_DIR.glob(f"*{ext}")] +
        [p for ext in SUPPORTED_FORMATS for p in INPUT_DIR.glob(f"*{ext.upper()}")]
    ))
    if not image_paths:
        print(f"在 {INPUT_DIR} 中未找到支持的图片文件")
        return

    print(f"找到 {len(image_paths)} 张图片，开始推理...")

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

    # ---------- 加载检测模型 ----------
    print("加载检测模型...")
    det_model = DetectionCNN(num_classes=NUM_DETECT_CLASSES, num_anchors=NUM_ANCHORS)
    det_weight_dir = os.path.join(WEIGHT_DIR, WEIGHT_DETECTION)
    if not os.path.exists(det_weight_dir):
        print("错误: 检测权重不存在")
        return
    load_weights_npy(det_model, det_weight_dir, prefix='det')
    det_model.to(device)
    det_model.eval()

    # ---------- 加载分割特征提取器 ----------
    print("加载分割特征提取器...")
    backbone = UNet()
    backbone_weight_dir = os.path.join(WEIGHT_DIR, WEIGHT_CONTRASTIVE)
    if not os.path.exists(backbone_weight_dir):
        print("错误: 对比学习权重不存在")
        return
    load_weights_npy(backbone, backbone_weight_dir, prefix='contrast')
    backbone.to(device)
    backbone.eval()

    # ---------- 加载代理AI ----------
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

        results = []  # 存储最终结果

        for i, box in enumerate(boxes):
            x1 = int(box[0].item() * orig_w)
            y1 = int(box[1].item() * orig_h)
            x2 = int(box[2].item() * orig_w)
            y2 = int(box[3].item() * orig_h)
            area_pixel = (x2 - x1) * (y2 - y1)

            cell_img = img.crop((x1, y1, x2, y2))
            cell_tensor = cell_transform(cell_img).unsqueeze(0).to(device)

            with torch.no_grad():
                feat = backbone.extract_features(cell_tensor)
                feat_np = feat.cpu().numpy()
                feat_scaled = scaler.transform(feat_np)
                pred_label = classifier.predict(feat_scaled)[0]
                if hasattr(classifier, 'predict_proba'):
                    prob_array = classifier.predict_proba(feat_scaled)[0]
                else:
                    prob_array = None

            # 初级概率字典
            if prob_array is not None:
                prob_dict = {CLASS_NAMES[j]: prob_array[j] for j in range(len(CLASS_NAMES))}
            else:
                prob_dict = {cls: 0.0 for cls in CLASS_NAMES}
                prob_dict[CLASS_NAMES[pred_label]] = 1.0

            # ---- 尺寸验证 ----
            weighted_scores = {}
            norm_scores = {}
            z_scores = {}
            for cls_name, prob in prob_dict.items():
                if cls_name in area_stats:
                    mean = area_stats[cls_name]['mean']
                    std = area_stats[cls_name]['std']
                    z = (area_pixel - mean) / std
                    z_scores[cls_name] = z
                    norm_score = math.exp(-0.5 * SIZE_VERIFICATION_STRICTNESS * (z ** 2))
                    norm_scores[cls_name] = norm_score
                else:
                    z_scores[cls_name] = float('nan')
                    norm_scores[cls_name] = 1.0
                weighted_scores[cls_name] = prob * norm_scores[cls_name]

            final_class = max(weighted_scores, key=weighted_scores.get)
            final_confidence = weighted_scores[final_class]

            # 记录结果（不保存单细胞图）
            result_item = {
                'bbox': [x1, y1, x2, y2],
                'area': area_pixel,
                'prob_primary': prob_dict,
                'z_scores': z_scores,
                'norm_scores': norm_scores,
                'weighted_scores': weighted_scores,
                'final_class': final_class,
                'final_confidence': final_confidence
            }
            results.append(result_item)

            class_stats[final_class] += 1
            total_cells += 1

            print(f"    细胞 {i+1}: 最终={final_class} (加权分={final_confidence:.4f}), 面积={area_pixel} px²")

        # ---------- 绘制结果图（显示最终类别） ----------
        draw = ImageDraw.Draw(img)
        for obj in results:
            x1, y1, x2, y2 = obj['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            label = f"{obj['final_class']}: {obj['final_confidence']:.2f}"
            draw.text((x1, y1-20), label, fill='red', font=font)

        output_img_path = OUTPUT_DIR / f"{img_path.stem}_result.png"
        img.save(output_img_path)

        # ---------- 保存文本报告 ----------
        txt_path = OUTPUT_DIR / f"{img_path.stem}_result.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"图像: {img_path.name}\n")
            f.write(f"检测到 {len(results)} 个细胞:\n\n")
            for idx, obj in enumerate(results):
                f.write(f"细胞 {idx+1}:\n")
                f.write(f"  位置: {obj['bbox']}\n")
                f.write(f"  检测面积: {obj['area']} px²\n")
                f.write(f"  初级概率（各类别）:\n")
                for cls, prob in obj['prob_primary'].items():
                    f.write(f"      {cls}: {prob:.6f}\n")
                f.write(f"  Z-score (相对于各类别正态分布):\n")
                for cls, z in obj['z_scores'].items():
                    if math.isnan(z):
                        f.write(f"      {cls}: 无统计数据\n")
                    else:
                        f.write(f"      {cls}: {z:.4f}\n")
                f.write(f"  正常程度分数 (由Z-score转换):\n")
                for cls, ns in obj['norm_scores'].items():
                    f.write(f"      {cls}: {ns:.6f}\n")
                f.write(f"  加权得分 (初级概率 × 正常程度):\n")
                for cls, ws in obj['weighted_scores'].items():
                    f.write(f"      {cls}: {ws:.6f}\n")
                f.write(f"  最终结果: {obj['final_class']} (加权得分: {obj['final_confidence']:.6f})\n")
                f.write("\n")

    # 打印统计结果
    print("\n" + "=" * 60)
    print("检测结果统计:")
    print(f"总检测细胞数: {total_cells}")
    print("各类细胞数量（最终结果）:")
    for name, count in class_stats.items():
        if count > 0:
            print(f"  {name}: {count}")
    print("\n" + "=" * 60)
    print(f"推理完成！结果保存在: {OUTPUT_DIR}")

if __name__ == '__main__':
    main()