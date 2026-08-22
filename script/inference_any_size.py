"""
一站式任意尺寸图像推理脚本
- 自动处理 input/ 下所有图片（包括子目录）
- 若图像尺寸 > IMAGE_SIZE，自动分片推理后拼接
- 若图像尺寸 <= IMAGE_SIZE，直接推理（小图填充黑边）
- 输出结果图（大图为原始尺寸，小图为填充后带黑边）和详细文本报告
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
SCALE_DIR = Path(__file__).parent / "scale"
SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

# ---------- 全局变量（模型、设备等） ----------
device = None
det_model = None
backbone = None
classifier = None
scaler = None
anchor_boxes = None
transform = None
cell_transform = None
font = None
area_stats = {}

# ---------- 辅助函数 ----------
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

def load_area_stats(scale_dir):
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

def load_models():
    global device, det_model, backbone, classifier, scaler, anchor_boxes, transform, cell_transform, font, area_stats
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    transform = get_transform(train=False)
    cell_transform = transforms.Compose([
        transforms.Resize((CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

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
        raise FileNotFoundError(f"检测权重不存在: {det_weight_dir}")
    load_weights_npy(det_model, det_weight_dir, prefix='det')
    det_model.to(device)
    det_model.eval()

    # 加载分割特征提取器
    print("加载分割特征提取器...")
    backbone = UNet()
    backbone_weight_dir = os.path.join(WEIGHT_DIR, WEIGHT_CONTRASTIVE)
    if not os.path.exists(backbone_weight_dir):
        raise FileNotFoundError(f"对比学习权重不存在: {backbone_weight_dir}")
    load_weights_npy(backbone, backbone_weight_dir, prefix='contrast')
    backbone.to(device)
    backbone.eval()

    # 加载代理AI
    print("加载代理AI...")
    proxy_dir = os.path.join(WEIGHT_DIR, WEIGHT_PROXY)
    if not os.path.exists(proxy_dir):
        raise FileNotFoundError(f"代理AI权重不存在: {proxy_dir}")
    classifier = joblib.load(os.path.join(proxy_dir, 'classifier.pkl'))
    scaler = joblib.load(os.path.join(proxy_dir, 'scaler.pkl'))

    anchor_boxes = torch.tensor(ANCHORS, dtype=torch.float32).to(device)

    # 加载面积统计
    print("加载面积统计信息...")
    area_stats = load_area_stats(SCALE_DIR)
    print(f"已加载 {len(area_stats)} 个类别的面积统计")

# ---------- 核心推理函数（处理单张 PIL 图像，返回结果图像和结果数据） ----------
def process_single_image(img, img_name, orig_w=None, orig_h=None, draw_bbox=True):
    """
    对单张 PIL 图像进行推理，返回标注后的 PIL 图像、结果列表、检测框数量等。
    若 orig_w, orig_h 未指定，则使用 img.size。
    """
    if orig_w is None or orig_h is None:
        orig_w, orig_h = img.size
    # 检测
    input_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        cls_pred, reg_pred = det_model(input_tensor)

    boxes, scores, labels = decode_predictions(cls_pred, reg_pred, anchor_boxes, grid_size=GRID_SIZE, conf_thresh=CONF_THRESH)
    if len(boxes) == 0:
        return img, [], 0

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

    results = []
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

        if prob_array is not None:
            prob_dict = {CLASS_NAMES[j]: prob_array[j] for j in range(len(CLASS_NAMES))}
        else:
            prob_dict = {cls: 0.0 for cls in CLASS_NAMES}
            prob_dict[CLASS_NAMES[pred_label]] = 1.0

        # 尺寸验证
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

    # 绘制结果
    if draw_bbox:
        draw = ImageDraw.Draw(img)
        for obj in results:
            x1, y1, x2, y2 = obj['bbox']
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            label = f"{obj['final_class']}: {obj['final_confidence']:.2f}"
            draw.text((x1, y1-20), label, fill='red', font=font)

    return img, results, len(boxes)

# ---------- 主流程 ----------
def main():
    global area_stats
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.exists():
        print(f"错误: input 目录不存在: {INPUT_DIR}")
        return

    # 加载模型
    load_models()

    # 递归收集所有图片
    image_paths = []
    for ext in SUPPORTED_FORMATS:
        image_paths.extend(INPUT_DIR.rglob(f"*{ext}"))
        image_paths.extend(INPUT_DIR.rglob(f"*{ext.upper()}"))
    image_paths = list(set(image_paths))

    if not image_paths:
        print(f"在 {INPUT_DIR} 中未找到支持的图片文件")
        return

    print(f"找到 {len(image_paths)} 张图片，开始推理...")
    print("-" * 60)

    total_cells = 0
    class_stats = {name: 0 for name in CLASS_NAMES}

    for img_path in image_paths:
        rel_path = img_path.relative_to(INPUT_DIR)
        output_subdir = OUTPUT_DIR / rel_path.parent
        output_subdir.mkdir(parents=True, exist_ok=True)

        print(f"\n处理: {rel_path}")
        img_orig = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img_orig.size

        # 判断是否需要切瓦片
        use_tiling = (orig_w > IMAGE_SIZE) or (orig_h > IMAGE_SIZE)

        if use_tiling:
            # 大图：填充黑边，分片处理
            new_w = ((orig_w + IMAGE_SIZE - 1) // IMAGE_SIZE) * IMAGE_SIZE
            new_h = ((orig_h + IMAGE_SIZE - 1) // IMAGE_SIZE) * IMAGE_SIZE
            padded = Image.new('RGB', (new_w, new_h), (0,0,0))
            padded.paste(img_orig, (0,0))

            # 创建结果画布（原始尺寸）
            result_canvas = Image.new('RGB', (orig_w, orig_h), (0,0,0))
            # 存储所有检测结果（用于文本报告）
            all_results = []
            tile_results = []

            # 分片处理
            for y in range(0, new_h, IMAGE_SIZE):
                for x in range(0, new_w, IMAGE_SIZE):
                    # 裁剪瓦片
                    tile = padded.crop((x, y, x+IMAGE_SIZE, y+IMAGE_SIZE))
                    # 处理瓦片
                    tile_result_img, tile_results_local, count = process_single_image(tile, f"{img_path.stem}_tile_{y//IMAGE_SIZE}_{x//IMAGE_SIZE}", draw_bbox=True)
                    tile_results.extend(tile_results_local)
                    # 将瓦片结果的有效区域（如果瓦片在原始图像范围内）粘贴到结果画布
                    # 计算该瓦片与原始图像的交集
                    tile_left = max(0, x)
                    tile_top = max(0, y)
                    tile_right = min(orig_w, x+IMAGE_SIZE)
                    tile_bottom = min(orig_h, y+IMAGE_SIZE)
                    if tile_right > tile_left and tile_bottom > tile_top:
                        # 从瓦片结果中裁剪有效区域
                        crop_box = (tile_left - x, tile_top - y, tile_right - x, tile_bottom - y)
                        valid_region = tile_result_img.crop(crop_box)
                        result_canvas.paste(valid_region, (tile_left, tile_top))

            # 合并所有结果
            all_results = tile_results

            # 最终结果图就是 result_canvas
            output_img = result_canvas
            # 生成文本报告（从 all_results 构建）
            final_results = all_results

        else:
            # 小图：填充黑边直接推理
            pad_w = IMAGE_SIZE - orig_w
            pad_h = IMAGE_SIZE - orig_h
            if pad_w > 0 or pad_h > 0:
                padded = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (0,0,0))
                padded.paste(img_orig, (0,0))
            else:
                padded = img_orig.copy()
            # 推理
            output_img, results, count = process_single_image(padded, img_path.stem, draw_bbox=True)
            final_results = results

        # 更新统计
        for obj in final_results:
            class_stats[obj['final_class']] += 1
            total_cells += 1

        # 保存结果图
        if use_tiling:
            out_name = f"{img_path.stem}_result.png"
        else:
            out_name = f"{img_path.stem}_result_padded.png"
        output_path = output_subdir / out_name
        output_img.save(output_path)

        # 保存文本报告
        txt_path = output_subdir / f"{img_path.stem}_result.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"图像: {rel_path}\n")
            f.write(f"检测到 {len(final_results)} 个细胞:\n\n")
            if not final_results:
                f.write("  未检测到任何细胞。\n")
            else:
                for idx, obj in enumerate(final_results):
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

        print(f"  处理完成，保存结果至: {output_path}")

    # 打印统计
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