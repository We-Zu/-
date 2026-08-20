"""
可视化 UI 工具 - 血细胞识别系统
功能：训练流程管理 + 数据标注 + 识别
依赖：tkinter, PIL, torch, numpy, joblib 等（项目已有）
"""

import os
import sys
import json
import subprocess
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
from pathlib import Path
from PIL import Image, ImageTk, ImageDraw
import random
import numpy as np
import torch
from torchvision import transforms
import importlib
# 尝试导入拖拽支持库
try:
    import tkinter
    import tkinterdnd2
    DND_AVAILABLE = True
    # 检查是否可用
    _test_root = tkinterdnd2.Tk()
    _test_root.destroy()
except Exception as e:
    DND_AVAILABLE = False
    print(f"拖拽功能不可用: {e}")

# 添加项目路径
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR / "script"))

# 导入配置和核心函数（用于状态检查）
from script.config import (
    DATASET_DIR, WEIGHT_DIR, CLASS_NAMES, DETECT_CLASSES,
    IMAGE_SIZE, GRID_SIZE, ANCHORS, NUM_ANCHORS,
    WEIGHT_DETECTION, WEIGHT_CONTRASTIVE, WEIGHT_PROXY,
    DATA_DETECTION, DATA_CLASSIFICATION,
    CONTRASTIVE_IMAGE_SIZE
)
# 将字符串路径转换为 Path 对象，便于拼接
DATASET_DIR = Path(DATASET_DIR)
WEIGHT_DIR = Path(WEIGHT_DIR)

# ======================== 工具函数 ========================
def check_dataset():
    """检查数据集是否存在"""
    cls_path = DATASET_DIR / DATA_CLASSIFICATION
    det_path = DATASET_DIR / DATA_DETECTION / "images"
    ann_path = DATASET_DIR / DATA_DETECTION / "annotations.json"
    return cls_path.exists() and det_path.exists() and ann_path.exists()

def check_weight(weight_type):
    """检查权重是否存在"""
    weight_dir = WEIGHT_DIR / weight_type
    if not weight_dir.exists():
        return False
    # 检查是否有 .pth 文件（检测器）或 .pkl（代理器）
    if weight_type == WEIGHT_DETECTION:
        files = list(weight_dir.glob("det_*.pth"))
        return len(files) > 0
    elif weight_type == WEIGHT_CONTRASTIVE:
        files = list(weight_dir.glob("contrast_*.pth"))
        return len(files) > 0
    elif weight_type == WEIGHT_PROXY:
        files = list(weight_dir.glob("*.pkl"))
        return len(files) > 0
    return False

def update_config_image_size(new_size):
    """修改 config.py 中的 IMAGE_SIZE"""
    config_path = BASE_DIR / "script" / "config.py"
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if line.strip().startswith("IMAGE_SIZE"):
            lines[i] = f"IMAGE_SIZE = {new_size}          # 统一缩放尺寸\n"
            break
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    # 重新导入 config 以更新变量（由于 import 已缓存，用 exec 或重启）
    messagebox.showinfo("成功", f"IMAGE_SIZE 已更新为 {new_size}，请重启工具生效。")

# ======================== 标注工具 ========================
class AnnotationTool:
    """交互式标注工具 - 用于生成检测数据集，支持图片裁剪"""
    def __init__(self, parent):
        self.parent = parent
        self.image_folder = tk.StringVar()
        self.image_list = []          # 所有图片路径
        self.current_idx = 0          # 当前显示第几张
        self.annotations = {}         # {image_name: [bbox_list]}
        self.current_bbox = None      # 当前绘制的矩形 (start_x, start_y, end_x, end_y)
        self.rect_id = None
        self.canvas = None
        self.photo = None
        self.tk_img = None
        self.scale = 1.0              # 缩放比例（原图到画布）
        self.orig_size = (512, 512)   # 原图尺寸
        # 初始化类别列表（复制一份，便于动态更新）
        self.class_names = list(CLASS_NAMES)

        # ---------- 裁剪相关 ----------
        self.crop_rect_id = None          # 裁剪框矩形ID
        self.crop_overlay_ids = []        # 遮罩ID列表
        self.crop_box = (0, 0, 0, 0)      # 裁剪框在原图坐标 (x1, y1, x2, y2)
        self.current_cls_ann = None       # 当前选中的标注（用于蓝色框）
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_orig_box = (0, 0, 0, 0)

        self.window = tk.Toplevel(parent)
        self.window.title("标注工具")
        self.window.geometry("1000x700")
        # 仅打开时短暂置顶，随后取消
        self.window.attributes('-topmost', True)
        self.window.after(200, lambda: self.window.attributes('-topmost', False))

        # 顶部控制区
        ctrl_frame = ttk.Frame(self.window)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Label(ctrl_frame, text="图片文件夹:").pack(side=tk.LEFT)
        ttk.Entry(ctrl_frame, textvariable=self.image_folder, width=40).pack(side=tk.LEFT, padx=5)
        ttk.Button(ctrl_frame, text="浏览", command=self.browse_folder).pack(side=tk.LEFT)
        ttk.Button(ctrl_frame, text="加载图片", command=self.load_images).pack(side=tk.LEFT, padx=5)

        # 仅生成检测数据集复选框
        self.only_det_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="仅生成检测数据集", variable=self.only_det_var).pack(side=tk.LEFT, padx=5)

        # 放大选中细胞复选框
        self.zoom_cell_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl_frame, text="放大选中细胞", variable=self.zoom_cell_var).pack(side=tk.LEFT, padx=5)
        # 绑定变化事件，实时更新画布（隐藏/显示蓝色框）
        self.zoom_cell_var.trace('w', lambda *args: self._update_crop_box())



        # 导航
        nav_frame = ttk.Frame(self.window)
        nav_frame.pack(fill=tk.X, padx=5, pady=2)
        ttk.Button(nav_frame, text="上一张", command=self.prev_image).pack(side=tk.LEFT)
        self.idx_label = ttk.Label(nav_frame, text="0/0")
        self.idx_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="下一张", command=self.next_image).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="删除当前框", command=self.delete_last_bbox).pack(side=tk.LEFT, padx=10)
        ttk.Button(nav_frame, text="清空当前图片标注", command=self.clear_current_annotations).pack(side=tk.LEFT)

        # 画布
        canvas_frame = ttk.Frame(self.window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas = tk.Canvas(canvas_frame, bg='gray', cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        # 滚轮缩放（Windows/Linux）
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind("<Button-4>", self._on_mousewheel)   # Linux
        self.canvas.bind("<Button-5>", self._on_mousewheel)   # Linux
        # 右键拖动（全局，用于黄色和蓝色框）
        self.canvas.bind("<ButtonPress-3>", self._on_crop_press)
        self.canvas.bind("<B3-Motion>", self._on_crop_drag)
        self.canvas.bind("<ButtonRelease-3>", self._on_crop_release)
        # 阻止右键菜单弹出（在回调中返回 "break"）
        # 键盘控制：左右方向键切换图片
        self.canvas.bind("<Left>", lambda e: self.prev_image())
        self.canvas.bind("<Right>", lambda e: self.next_image())
        # 状态栏
        self.status_label = ttk.Label(self.window, text="就绪", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(fill=tk.X, padx=5, pady=2)

    def update_status(self, msg):
        self.status_label.config(text=msg)

    def browse_folder(self):
        folder = filedialog.askdirectory(title="选择包含图片的文件夹")
        if folder:
            self.image_folder.set(folder)

    def load_images(self):
        folder = self.image_folder.get()
        if not folder:
            messagebox.showerror("错误", "请先选择文件夹")
            return
        self.image_list = []
        seen = set()
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif'):
            for p in Path(folder).glob(ext):
                key = p.name.lower()
                if key not in seen:
                    seen.add(key)
                    self.image_list.append(p)
            for p in Path(folder).glob(ext.upper()):
                key = p.name.lower()
                if key not in seen:
                    seen.add(key)
                    self.image_list.append(p)
        self.image_list = sorted(self.image_list, key=lambda p: p.name.lower())
        if not self.image_list:
            messagebox.showerror("错误", "未找到支持的图片")
            return
        self.image_list = sorted(self.image_list)
        self.annotations = {}
        self.current_idx = 0
        # 尝试加载已有的标注文件（COCO格式）
        ann_file = Path(folder) / "annotations.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                data = json.load(f)
                id_to_file = {img['id']: img['file_name'] for img in data.get('images', [])}
                for ann in data.get('annotations', []):
                    img_id = ann['image_id']
                    img_name = id_to_file.get(img_id)
                    if img_name is None:
                        continue
                    if img_name not in self.annotations:
                        self.annotations[img_name] = []
                    cat_id = ann['category_id']
                    cat_name = None
                    for cat in data.get('categories', []):
                        if cat['id'] == cat_id:
                            cat_name = cat['name']
                            break
                    self.annotations[img_name].append({
                        'bbox': ann['bbox'],
                        'category_id': cat_id,
                        'category_name': cat_name or str(cat_id)
                    })
        self.refresh_class_names()
        self.show_image()
        self.update_status(f"加载了 {len(self.image_list)} 张图片")

    def refresh_class_names(self):
        try:
            import script.config
            importlib.reload(script.config)
            self.class_names = list(script.config.CLASS_NAMES)
        except:
            pass

    # ---------- 裁剪框绘制与交互 ----------
    def _update_crop_box(self):
        """根据当前 self.crop_box（黄色检测框）和所有标注的 cls_crop_box（蓝色分类框）更新画布"""
        self.canvas.delete("crop_rect")
        self.canvas.delete("crop_overlay")
        self.canvas.delete("crop_rect_cls")
        self.canvas.delete("crop_overlay_cls")
        if self.tk_img is None:
            return

        # ---- 黄色检测框 ----
        x1, y1, x2, y2 = self.crop_box
        cx1 = x1 * self.scale
        cy1 = y1 * self.scale
        cx2 = x2 * self.scale
        cy2 = y2 * self.scale
        self.canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                     outline='yellow', width=2, tags="crop_rect")
        # 透明内部区域（便于点击）
        self.canvas.create_rectangle(cx1+2, cy1+2, cx2-2, cy2-2,
                                     fill='', outline='', tags="crop_rect")
        # 遮罩
        img_w = self.orig_size[0] * self.scale
        img_h = self.orig_size[1] * self.scale
        self.canvas.create_rectangle(0, 0, img_w, cy1, fill='black', stipple='gray50', outline='', tags="crop_overlay")
        self.canvas.create_rectangle(0, cy2, img_w, img_h, fill='black', stipple='gray50', outline='', tags="crop_overlay")
        self.canvas.create_rectangle(0, cy1, cx1, cy2, fill='black', stipple='gray50', outline='', tags="crop_overlay")
        self.canvas.create_rectangle(cx2, cy1, img_w, cy2, fill='black', stipple='gray50', outline='', tags="crop_overlay")

        # ---- 所有蓝色分类框（仅当未勾选“放大选中细胞”时显示） ----
        if not self.zoom_cell_var.get():
            img_name = self.image_list[self.current_idx].name if self.image_list else None
            if img_name and img_name in self.annotations:
                for ann in self.annotations[img_name]:
                    if 'cls_crop_box' in ann:
                        bx1, by1, bx2, by2 = ann['cls_crop_box']
                        bcx1 = bx1 * self.scale
                        bcy1 = by1 * self.scale
                        bcx2 = bx2 * self.scale
                        bcy2 = by2 * self.scale
                        self.canvas.create_rectangle(bcx1, bcy1, bcx2, bcy2,
                                                     outline='blue', width=3, tags="crop_rect_cls")
                        self.canvas.create_rectangle(bcx1+2, bcy1+2, bcx2-2, bcy2-2,
                                                     fill='', outline='', tags="crop_rect_cls")
                        cat_name = ann.get('category_name', '')
                        if cat_name:
                            self.canvas.create_text(bcx1, bcy1-5, text=cat_name,
                                                    fill='blue', anchor=tk.SW, tags="crop_rect_cls")

    def _init_crop_box(self):
        """根据 IMAGE_SIZE 和当前图片尺寸初始化裁剪框（居中，正方形，边长 = IMAGE_SIZE）"""
        if self.orig_size[0] == 0 or self.orig_size[1] == 0:
            return
        target_size = IMAGE_SIZE
        img_w, img_h = self.orig_size
        # 裁剪框边长 = min(IMAGE_SIZE, 图片短边) 确保不超出图片
        self.crop_size = min(img_w, img_h, target_size)
        cx = img_w // 2
        cy = img_h // 2
        half = self.crop_size // 2
        self.crop_box = (cx - half, cy - half, cx + half, cy + half)
        self._clamp_crop_box()

    def _clamp_crop_box(self):
        """确保裁剪框不超出图片边界"""
        x1, y1, x2, y2 = self.crop_box
        img_w, img_h = self.orig_size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        # 如果宽或高为0，重新初始化
        if x2 <= x1 or y2 <= y1:
            self._init_crop_box()
        else:
            self.crop_box = (x1, y1, x2, y2)

    def _on_crop_press(self, event):
        """开始拖动裁剪框（右键按下）"""
        x, y = event.x, event.y
        img_name = self.image_list[self.current_idx].name if self.image_list else None
        # 先遍历所有标注的蓝色框（仅当未勾选“放大选中细胞”）
        if not self.zoom_cell_var.get():
            if img_name and img_name in self.annotations:
                for ann in self.annotations[img_name]:
                    if 'cls_crop_box' in ann:
                        bx1, by1, bx2, by2 = ann['cls_crop_box']
                        bcx1 = bx1 * self.scale
                        bcy1 = by1 * self.scale
                        bcx2 = bx2 * self.scale
                        bcy2 = by2 * self.scale
                        if bcx1 <= x <= bcx2 and bcy1 <= y <= bcy2:
                            self.is_dragging = True
                            self.drag_start_x = event.x
                            self.drag_start_y = event.y
                            self.drag_orig_box = ann['cls_crop_box']
                            self.drag_target = 'cls'
                            self.current_cls_ann = ann
                            return "break"
        # 再检查黄色框
        x1, y1, x2, y2 = self.crop_box
        cx1 = x1 * self.scale
        cy1 = y1 * self.scale
        cx2 = x2 * self.scale
        cy2 = y2 * self.scale
        if cx1 <= x <= cx2 and cy1 <= y <= cy2:
            self.is_dragging = True
            self.drag_start_x = event.x
            self.drag_start_y = event.y
            self.drag_orig_box = self.crop_box
            self.drag_target = 'det'
            self.current_cls_ann = None   # 取消选中蓝色框
        return "break"  # 阻止右键菜单

    def _on_crop_drag(self, event):
        """拖动裁剪框（基于起始位置的总偏移，保持正方形）"""
        if not self.is_dragging:
            return
        dx_total = (event.x - self.drag_start_x) / self.scale
        dy_total = (event.y - self.drag_start_y) / self.scale
        x1, y1, x2, y2 = self.drag_orig_box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        new_cx = cx + dx_total
        new_cy = cy + dy_total

        if self.drag_target == 'det':
            half = self.crop_size // 2
            size = self.crop_size
        else:  # cls
            half = CONTRASTIVE_IMAGE_SIZE // 2
            size = CONTRASTIVE_IMAGE_SIZE

        new_box = (new_cx - half, new_cy - half, new_cx + half, new_cy + half)
        img_w, img_h = self.orig_size
        x1_new = max(0, new_box[0])
        y1_new = max(0, new_box[1])
        x2_new = min(img_w, new_box[2])
        y2_new = min(img_h, new_box[3])

        # 调整保持尺寸
        if x2_new - x1_new < size:
            if x1_new == 0:
                x2_new = size
            elif x2_new == img_w:
                x1_new = img_w - size
        if y2_new - y1_new < size:
            if y1_new == 0:
                y2_new = size
            elif y2_new == img_h:
                y1_new = img_h - size
        x1_new = int(max(0, x1_new))
        y1_new = int(max(0, y1_new))
        x2_new = int(min(img_w, x2_new))
        y2_new = int(min(img_h, y2_new))

        new_box = (x1_new, y1_new, x2_new, y2_new)

        if self.drag_target == 'det':
            self.crop_box = new_box
        else:
            if self.current_cls_ann is not None:
                self.current_cls_ann['cls_crop_box'] = new_box

        self._update_crop_box()

    def _on_crop_release(self, event):
        self.is_dragging = False
        self.drag_target = None

    def _on_crop_drag_left(self, event):
        """左键拖动裁剪框（仅在裁剪框上按下时触发）"""
        if not self.is_dragging:
            return
        # 计算从按下到当前的总偏移（画布坐标转原图坐标）
        dx_total = (event.x - self.drag_start_x) / self.scale
        dy_total = (event.y - self.drag_start_y) / self.scale
        x1, y1, x2, y2 = self.drag_orig_box
        new_box = (x1 + dx_total, y1 + dy_total, x2 + dx_total, y2 + dy_total)
        # 限制不超出图片
        img_w, img_h = self.orig_size
        x1_new = max(0, new_box[0])
        y1_new = max(0, new_box[1])
        x2_new = min(img_w, new_box[2])
        y2_new = min(img_h, new_box[3])
        # 保持宽高不变（使用原始宽高）
        w = x2 - x1
        h = y2 - y1
        # 边界修正
        if x1_new < 0:
            x1_new = 0
            x2_new = w
        if y1_new < 0:
            y1_new = 0
            y2_new = h
        if x2_new > img_w:
            x2_new = img_w
            x1_new = img_w - w
        if y2_new > img_h:
            y2_new = img_h
            y1_new = img_h - h
        # 再次确保非负
        x1_new = max(0, x1_new)
        y1_new = max(0, y1_new)
        x2_new = min(img_w, x2_new)
        y2_new = min(img_h, y2_new)
        self.crop_box = (int(x1_new), int(y1_new), int(x2_new), int(y2_new))
        self._update_crop_box()        

    def _on_mousewheel(self, event):
        """滚轮缩放蓝色分类裁剪框（黄色框固定不变），缩放仅影响框大小，保存时始终缩放到 CONTRASTIVE_IMAGE_SIZE"""
        # 如果没有选中蓝色框，不处理
        if self.current_cls_ann is None or 'cls_crop_box' not in self.current_cls_ann:
            return
        # 获取滚轮方向
        if event.num == 4 or event.delta > 0:
            factor = 1.1
        elif event.num == 5 or event.delta < 0:
            factor = 0.9
        else:
            return
        # 获取当前蓝色框
        x1, y1, x2, y2 = self.current_cls_ann['cls_crop_box']
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        # 当前边长
        current_size = x2 - x1
        new_size = current_size * factor
        # 限制最小和最大尺寸（至少 10，最大不超过图片短边的 90%）
        min_size = 10
        max_size = min(self.orig_size) * 0.9
        if new_size < min_size or new_size > max_size:
            return
        # 计算新框（保持中心）
        half = new_size // 2
        new_box = (cx - half, cy - half, cx + half, cy + half)
        # 限制不超出图片边界
        img_w, img_h = self.orig_size
        x1_new = max(0, new_box[0])
        y1_new = max(0, new_box[1])
        x2_new = min(img_w, new_box[2])
        y2_new = min(img_h, new_box[3])
        # 调整保持尺寸（如果超出边界，则调整中心）
        if x2_new - x1_new < new_size:
            if x1_new == 0:
                x2_new = new_size
            elif x2_new == img_w:
                x1_new = img_w - new_size
        if y2_new - y1_new < new_size:
            if y1_new == 0:
                y2_new = new_size
            elif y2_new == img_h:
                y1_new = img_h - new_size
        x1_new = int(max(0, x1_new))
        y1_new = int(max(0, y1_new))
        x2_new = int(min(img_w, x2_new))
        y2_new = int(min(img_h, y2_new))
        # 更新蓝色框
        self.current_cls_ann['cls_crop_box'] = (x1_new, y1_new, x2_new, y2_new)
        self._update_crop_box()

    # ---------- 图像显示和标注 ----------
    def show_image(self):
        if not self.image_list or self.current_idx >= len(self.image_list):
            return
        img_path = self.image_list[self.current_idx]
        self.idx_label.config(text=f"{self.current_idx+1}/{len(self.image_list)}")
        img = Image.open(img_path).convert('RGB')
        self.orig_size = img.size
        # 缩放以适应画布
        canvas_w = self.canvas.winfo_width() if self.canvas.winfo_width() > 50 else 800
        canvas_h = self.canvas.winfo_height() if self.canvas.winfo_height() > 50 else 600
        scale_w = canvas_w / img.width
        scale_h = canvas_h / img.height
        self.scale = min(scale_w, scale_h, 1.0)  # 最大不放大
        new_size = (int(img.width * self.scale), int(img.height * self.scale))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_resized)
        self.canvas.delete("all")
        self.canvas.config(scrollregion=(0,0,new_size[0], new_size[1]))
        self.canvas.create_image(0,0, anchor=tk.NW, image=self.tk_img, tags="bg_img")

        # 初始化裁剪框（如果尚未初始化或图片尺寸变化）
        if not hasattr(self, 'crop_box') or self.crop_box == (0,0,0,0):
            self._init_crop_box()
        else:
            # 确保裁剪框在当前图片范围内
            self._clamp_crop_box()
        self._update_crop_box()

        # 绘制已有标注（在裁剪框内的标注）
        img_name = img_path.name
        if img_name in self.annotations:
            for ann in self.annotations[img_name]:
                bbox = ann['bbox']
                # 检查标注是否在裁剪框内（可选）
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
                # 转换到画布坐标
                cx1 = x1 * self.scale
                cy1 = y1 * self.scale
                cx2 = x2 * self.scale
                cy2 = y2 * self.scale
                self.canvas.create_rectangle(cx1,cy1,cx2,cy2, outline='red', width=2, tags="annotation")
                self.canvas.create_text(cx1, cy1-10, text=ann['category_name'], fill='red', anchor=tk.SW, tags="annotation")
        self.current_bbox = None
        self.current_cls_ann = None   # 切换图片后清除蓝色框选中状态
        self._update_crop_box()       # 重新绘制（此时不显示蓝色框）
        self.canvas.focus_set()

    def on_mouse_down(self, event):
        # 左键标注，与裁剪拖动无关
        self.start_x = event.x
        self.start_y = event.y
        if self.rect_id:
            self.canvas.delete(self.rect_id)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='blue', width=1)

    def on_mouse_move(self, event):
        if self.rect_id:
            self.canvas.coords(self.rect_id, self.start_x, self.start_y, event.x, event.y)

    def on_mouse_up(self, event):
        if not self.rect_id:
            return
        end_x, end_y = event.x, event.y
        if abs(end_x - self.start_x) < 5 or abs(end_y - self.start_y) < 5:
            self.canvas.delete(self.rect_id)
            self.rect_id = None
            return
        # 转换为原图坐标
        x1 = min(self.start_x, end_x) / self.scale
        y1 = min(self.start_y, end_y) / self.scale
        x2 = max(self.start_x, end_x) / self.scale
        y2 = max(self.start_y, end_y) / self.scale
        w = x2 - x1
        h = y2 - y1

        # 如果勾选“放大选中细胞”，强制标注框为正方形
        if self.zoom_cell_var.get():
            size = max(w, h)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            half = size // 2
            x1_new = max(0, cx - half)
            y1_new = max(0, cy - half)
            x2_new = min(self.orig_size[0], cx + half)
            y2_new = min(self.orig_size[1], cy + half)
            # 如果边界导致宽高不等，用较小值重新调整
            if x2_new - x1_new != y2_new - y1_new:
                size = min(x2_new - x1_new, y2_new - y1_new)
                half = size // 2
                cx = (x1_new + x2_new) // 2
                cy = (y1_new + y2_new) // 2
                x1_new = max(0, cx - half)
                y1_new = max(0, cy - half)
                x2_new = min(self.orig_size[0], cx + half)
                y2_new = min(self.orig_size[1], cy + half)
            x1, y1, x2, y2 = int(x1_new), int(y1_new), int(x2_new), int(y2_new)
            w = x2 - x1
            h = y2 - y1

        # 保存临时框信息
        self.temp_bbox = [int(x1), int(y1), int(w), int(h)]
        self.temp_img_name = self.image_list[self.current_idx].name
        # 弹出类别选择窗口
        self.show_category_selector(event.x_root, event.y_root)

    # ---------- 类别选择（不变） ----------
    def show_category_selector(self, x_root, y_root):
        win = tk.Toplevel(self.window)
        win.title("选择细胞类别")
        win.attributes('-topmost', True)
        win.geometry(f"280x600+{x_root+10}+{y_root-10}")
        win.resizable(False, True)
        win.focus_force()
        
        # 顶部标签
        ttk.Label(win, text="请选择类别：", font=('Arial', 10, 'bold')).pack(pady=5)
        
        # 使用 grid 布局：滚动区域占满剩余空间，底部按钮固定在底部
        main_frame = ttk.Frame(win)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=0)
        
        # 滚动区域
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        inner_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=inner_frame, anchor="nw")
        
        # 数字键映射：1-9, 0
        key_mapping = ['1','2','3','4','5','6','7','8','9','0']
        
        def select_category(cat_name):
            win.destroy()
            self.add_annotation_with_category(cat_name)
        
        # 创建类别按钮
        for idx, cat in enumerate(self.class_names):
            if idx < 10:
                key = key_mapping[idx]
                btn_text = f"{key}. {cat}"
            else:
                btn_text = cat
            btn = ttk.Button(inner_frame, text=btn_text, width=25,
                             command=lambda c=cat: select_category(c))
            btn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
            # 为前10个按钮绑定数字键
            if idx < 10:
                key = key_mapping[idx]
                win.bind(f"<Key-{key}>", lambda e, c=cat: select_category(c))
        
        def configure_inner_frame(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner_frame.bind("<Configure>", configure_inner_frame)
        
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        
        # 底部按钮框架
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=5)
        ttk.Label(bottom_frame, text="").pack(side=tk.LEFT, padx=5)
        new_btn = ttk.Button(bottom_frame, text="新建类别", 
                             command=lambda: self.add_new_category(win))
        new_btn.pack(side=tk.LEFT, padx=5)
        cancel_btn = ttk.Button(bottom_frame, text="取消", command=lambda: [self.canvas.focus_set(), win.destroy()])
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        main_frame.grid_rowconfigure(1, weight=0)
        
        def on_close():
            if hasattr(self, 'temp_bbox'):
                self.canvas.delete(self.rect_id)
                self.rect_id = None
                self.temp_bbox = None
                self.temp_img_name = None
            self.canvas.focus_set()
            win.destroy()
        win.protocol("WM_DELETE_WINDOW", on_close)
        
        # 确保窗口能接收键盘事件
        win.focus_set()

    def add_new_category(self, parent_win):
        new_name = simpledialog.askstring("新建类别", "请输入新类别名称（英文）：", parent=parent_win)
        if not new_name:
            return
        if new_name in self.class_names:
            messagebox.showwarning("警告", f"类别 '{new_name}' 已存在！")
            return
        config_path = BASE_DIR / "script" / "config.py"
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip().startswith("CLASS_NAMES"):
                    import ast
                    if '=' in line:
                        list_str = line.split('=', 1)[1].strip()
                        try:
                            existing_list = ast.literal_eval(list_str)
                            if not isinstance(existing_list, list):
                                existing_list = []
                        except:
                            existing_list = []
                        if new_name not in existing_list:
                            existing_list.append(new_name)
                        new_line = f"CLASS_NAMES = {repr(existing_list)}          # 可自由增删，顺序与文件夹名一致\n"
                        lines[i] = new_line
                        break
            with open(config_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)
            import script.config
            importlib.reload(script.config)
            self.class_names = list(script.config.CLASS_NAMES)
            parent_win.destroy()
            self.add_annotation_with_category(new_name)
            messagebox.showinfo("成功", f"已添加新类别 '{new_name}'，并已更新 config.py。")
        except Exception as e:
            messagebox.showerror("错误", f"更新 config.py 失败：{e}")

    def add_annotation_with_category(self, cat_name):
        if not hasattr(self, 'temp_bbox') or not self.temp_bbox:
            return
        img_name = self.temp_img_name
        bbox = self.temp_bbox
        if cat_name not in self.class_names:
            self.refresh_class_names()
            if cat_name not in self.class_names:
                messagebox.showerror("错误", f"类别 '{cat_name}' 不在列表中，请重新选择。")
                return
        cat_id = self.class_names.index(cat_name) + 1
        if img_name not in self.annotations:
            self.annotations[img_name] = []
        # 计算默认分类裁剪框（以标注框中心为中心，边长为 CONTRASTIVE_IMAGE_SIZE）
        x, y, w, h = bbox
        center_x = x + w // 2
        center_y = y + h // 2
        half = CONTRASTIVE_IMAGE_SIZE // 2
        cls_box = (center_x - half, center_y - half, center_x + half, center_y + half)
        # 限制在图片范围内
        img_w, img_h = self.orig_size
        cls_box = (max(0, cls_box[0]), max(0, cls_box[1]),
                   min(img_w, cls_box[2]), min(img_h, cls_box[3]))
        # 如果超出导致尺寸不对，调整回有效
        if cls_box[2] - cls_box[0] < CONTRASTIVE_IMAGE_SIZE:
            if cls_box[0] == 0:
                cls_box = (0, cls_box[1], CONTRASTIVE_IMAGE_SIZE, cls_box[1] + CONTRASTIVE_IMAGE_SIZE)
            elif cls_box[2] == img_w:
                cls_box = (img_w - CONTRASTIVE_IMAGE_SIZE, cls_box[1], img_w, cls_box[1] + CONTRASTIVE_IMAGE_SIZE)
        if cls_box[3] - cls_box[1] < CONTRASTIVE_IMAGE_SIZE:
            if cls_box[1] == 0:
                cls_box = (cls_box[0], 0, cls_box[0] + CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE)
            elif cls_box[3] == img_h:
                cls_box = (cls_box[0], img_h - CONTRASTIVE_IMAGE_SIZE, cls_box[0] + CONTRASTIVE_IMAGE_SIZE, img_h)
        # 存入标注
        ann_dict = {
            'bbox': bbox,
            'category_id': cat_id,
            'category_name': cat_name,
            'cls_crop_box': cls_box   # 新增字段
        }
        self.annotations[img_name].append(ann_dict)
        # 设置为当前选中的标注（用于显示蓝色裁剪框）
        self.current_cls_ann = ann_dict

        # 在画布上绘制红色框
        x1 = bbox[0] * self.scale
        y1 = bbox[1] * self.scale
        x2 = (bbox[0] + bbox[2]) * self.scale
        y2 = (bbox[1] + bbox[3]) * self.scale
        self.canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
        self.canvas.create_text(x1, y1-10, text=cat_name, fill='red', anchor=tk.SW)
        self.canvas.delete(self.rect_id)
        self.rect_id = None
        self.temp_bbox = None
        self.temp_img_name = None
        self.update_status(f"添加标注: {cat_name} 框 ({bbox[2]}x{bbox[3]})")
        # 更新画布显示（重绘蓝色框）
        self._update_crop_box()
        self.canvas.focus_set()

        # ---------- 若勾选“仅生成检测数据集”，立即保存并跳转下一张 ----------
        if self.only_det_var.get():
            # 立即保存当前图片的检测数据（跳过分类）
            cur_img_path = self.image_list[self.current_idx]
            cur_anns = self.annotations.get(img_name, [])
            cur_crop = self.crop_box
            # 后台导出检测数据（skip_cls=True）
            if cur_anns and cur_crop != (0, 0, 0, 0):
                threading.Thread(
                    target=self._export_image_data,
                    args=(cur_img_path, cur_anns, cur_crop, self.class_names, True),  # 增加 skip_cls=True
                    daemon=True
                ).start()
            # 自动跳转到下一张（如果存在）
            if self.current_idx < len(self.image_list) - 1:
                self.next_image()

    # ---------- 导航和删除 ----------
    def delete_last_bbox(self):
        img_name = self.image_list[self.current_idx].name
        if img_name in self.annotations and self.annotations[img_name]:
            self.annotations[img_name].pop()
            self.show_image()
            self.update_status("删除最后一个标注")

    def clear_current_annotations(self):
        img_name = self.image_list[self.current_idx].name
        if img_name in self.annotations:
            del self.annotations[img_name]
            self.show_image()
            self.update_status("清空当前图片标注")

    def _export_current_image(self):
        """将当前显示的图片根据裁剪框裁剪并导出到数据集（检测+分类）"""
        if not self.image_list or self.current_idx >= len(self.image_list):
            return
        img_path = self.image_list[self.current_idx]
        img_name = img_path.name
        if img_name not in self.annotations or not self.annotations[img_name]:
            # 没有标注，不导出
            return
        if not hasattr(self, 'crop_box') or self.crop_box == (0,0,0,0):
            return
        crop_box = self.crop_box
        src_folder = Path(self.image_folder.get())
        img = Image.open(img_path).convert('RGB')
        # 裁剪并缩放到 IMAGE_SIZE
        cropped = img.crop(crop_box)
        cropped = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        # 保存到检测数据集 images
        det_img_dir = DATASET_DIR / DATA_DETECTION / "images"
        det_img_dir.mkdir(parents=True, exist_ok=True)
        dst_path = det_img_dir / img_name
        cropped.save(dst_path)
        # 更新 annotations.json
        ann_file = DATASET_DIR / DATA_DETECTION / "annotations.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                coco_data = json.load(f)
        else:
            coco_data = {"images": [], "annotations": [], "categories": [{"id": i+1, "name": name} for i, name in enumerate(self.class_names)]}
        # 查找或添加图片记录
        img_id = None
        for img in coco_data['images']:
            if img['file_name'] == img_name:
                img_id = img['id']
                break
        if img_id is None:
            img_id = len(coco_data['images']) + 1
            coco_data['images'].append({
                "id": img_id,
                "file_name": img_name,
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE
            })
        # 删除该图片的旧标注（如果有）
        coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['image_id'] != img_id]
        # 添加新的标注（坐标转换）
        x1, y1, x2, y2 = crop_box
        crop_w = x2 - x1
        crop_h = y2 - y1
        for ann in self.annotations[img_name]:
            bbox = ann['bbox']
            new_x = (bbox[0] - x1) / crop_w * IMAGE_SIZE
            new_y = (bbox[1] - y1) / crop_h * IMAGE_SIZE
            new_w = bbox[2] / crop_w * IMAGE_SIZE
            new_h = bbox[3] / crop_h * IMAGE_SIZE
            new_ann = {
                "image_id": img_id,
                "bbox": [new_x, new_y, new_w, new_h],
                "category_id": ann['category_id'],
                "area": new_w * new_h,
                "iscrowd": 0,
                "id": len(coco_data['annotations']) + 1
            }
            coco_data['annotations'].append(new_ann)
        # 写回
        with open(ann_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        # 生成分类数据集
        cls_root = DATASET_DIR / DATA_CLASSIFICATION
        cls_root.mkdir(parents=True, exist_ok=True)
        for cls_name in self.class_names:
            (cls_root / cls_name).mkdir(exist_ok=True)
        # 计数
        cls_counters = {}
        for cls_name in self.class_names:
            existing = list((cls_root / cls_name).glob("*.png"))
            max_num = 0
            for f in existing:
                try:
                    num = int(f.stem.split('_')[-1])
                    if num > max_num:
                        max_num = num
                except:
                    pass
            cls_counters[cls_name] = max_num + 1
        # 获取原图尺寸（用于边界修正）
        orig_w, orig_h = self.orig_size
        for ann in self.annotations[img_name]:
            cat_name = ann['category_name']
            if cat_name not in self.class_names:
                continue
            # 决定分类裁剪框
            if self.zoom_cell_var.get():
                # 使用红色标注框，并强制为正方形（以标注框中心，边长为最大边长）
                x, y, w, h = ann['bbox']
                size = max(w, h)
                cx = x + w // 2
                cy = y + h // 2
                half = size // 2
                x1_new = max(0, cx - half)
                y1_new = max(0, cy - half)
                x2_new = min(orig_w, cx + half)
                y2_new = min(orig_h, cy + half)
                # 如果边界导致宽高不等，用较小值重新调整
                if x2_new - x1_new != y2_new - y1_new:
                    size = min(x2_new - x1_new, y2_new - y1_new)
                    half = size // 2
                    cx = (x1_new + x2_new) // 2
                    cy = (y1_new + y2_new) // 2
                    x1_new = max(0, cx - half)
                    y1_new = max(0, cy - half)
                    x2_new = min(orig_w, cx + half)
                    y2_new = min(orig_h, cy + half)
                cls_box = (int(x1_new), int(y1_new), int(x2_new), int(y2_new))
            else:
                # 使用蓝色裁剪框
                cls_box = ann.get('cls_crop_box')
                if cls_box is None:
                    # 默认生成（以标注框中心）
                    x, y, w, h = ann['bbox']
                    center_x = x + w // 2
                    center_y = y + h // 2
                    half = CONTRASTIVE_IMAGE_SIZE // 2
                    cls_box = (center_x - half, center_y - half, center_x + half, center_y + half)
                    cls_box = (max(0, cls_box[0]), max(0, cls_box[1]),
                               min(orig_w, cls_box[2]), min(orig_h, cls_box[3]))
                    if cls_box[2] - cls_box[0] < CONTRASTIVE_IMAGE_SIZE:
                        if cls_box[0] == 0:
                            cls_box = (0, cls_box[1], CONTRASTIVE_IMAGE_SIZE, cls_box[1] + CONTRASTIVE_IMAGE_SIZE)
                        elif cls_box[2] == orig_w:
                            cls_box = (orig_w - CONTRASTIVE_IMAGE_SIZE, cls_box[1], orig_w, cls_box[1] + CONTRASTIVE_IMAGE_SIZE)
                    if cls_box[3] - cls_box[1] < CONTRASTIVE_IMAGE_SIZE:
                        if cls_box[1] == 0:
                            cls_box = (cls_box[0], 0, cls_box[0] + CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE)
                        elif cls_box[3] == orig_h:
                            cls_box = (cls_box[0], orig_h - CONTRASTIVE_IMAGE_SIZE, cls_box[0] + CONTRASTIVE_IMAGE_SIZE, orig_h)
            # 裁剪并缩放（直接从原图裁剪）
            cls_crop_img = img.crop(cls_box)
            cls_crop_img = cls_crop_img.resize((CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE), Image.Resampling.LANCZOS)
            save_dir = cls_root / cat_name
            save_dir.mkdir(exist_ok=True)
            idx = cls_counters[cat_name]
            save_path = save_dir / f"{cat_name}_{idx:04d}.png"
            cls_crop_img.save(save_path)
            cls_counters[cat_name] = idx + 1

    def _export_image_data(self, img_path, anns, crop_box, class_names, skip_cls=False):
        """后台线程执行：将指定图片的标注和裁剪数据导出到数据集
           skip_cls: True 时只生成检测数据，不生成分类数据
        """
        import threading
        try:
            img_name = img_path.name
            img = Image.open(img_path).convert('RGB')
            cropped = img.crop(crop_box)
            cropped = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)

            # 保存检测图片
            det_img_dir = DATASET_DIR / DATA_DETECTION / "images"
            det_img_dir.mkdir(parents=True, exist_ok=True)
            dst_path = det_img_dir / img_name
            cropped.save(dst_path)

            # 更新 annotations.json
            ann_file = DATASET_DIR / DATA_DETECTION / "annotations.json"
            coco_data = None
            if ann_file.exists() and ann_file.stat().st_size > 0:
                try:
                    with open(ann_file, 'r') as f:
                        coco_data = json.load(f)
                except json.JSONDecodeError:
                    coco_data = None
            if coco_data is None:
                coco_data = {"images": [], "annotations": [], "categories": [{"id": i+1, "name": name} for i, name in enumerate(class_names)]}

            # 查找或添加图片记录
            img_id = None
            for img_rec in coco_data['images']:
                if img_rec['file_name'] == img_name:
                    img_id = img_rec['id']
                    break
            if img_id is None:
                img_id = len(coco_data['images']) + 1
                coco_data['images'].append({
                    "id": img_id,
                    "file_name": img_name,
                    "width": IMAGE_SIZE,
                    "height": IMAGE_SIZE
                })
            # 删除旧标注
            coco_data['annotations'] = [ann for ann in coco_data['annotations'] if ann['image_id'] != img_id]
            # 添加新标注
            x1, y1, x2, y2 = crop_box
            crop_w = x2 - x1
            crop_h = y2 - y1
            for ann in anns:
                bbox = ann['bbox']
                new_x = (bbox[0] - x1) / crop_w * IMAGE_SIZE
                new_y = (bbox[1] - y1) / crop_h * IMAGE_SIZE
                new_w = bbox[2] / crop_w * IMAGE_SIZE
                new_h = bbox[3] / crop_h * IMAGE_SIZE
                new_ann = {
                    "image_id": img_id,
                    "bbox": [new_x, new_y, new_w, new_h],
                    "category_id": ann['category_id'],
                    "area": new_w * new_h,
                    "iscrowd": 0,
                    "id": len(coco_data['annotations']) + 1
                }
                coco_data['annotations'].append(new_ann)
            with open(ann_file, 'w') as f:
                json.dump(coco_data, f, indent=2)

            # ---------- 分类数据集（仅在 skip_cls=False 时生成） ----------
            if not skip_cls:
                cls_root = DATASET_DIR / DATA_CLASSIFICATION
                cls_root.mkdir(parents=True, exist_ok=True)
                for cls_name in class_names:
                    (cls_root / cls_name).mkdir(exist_ok=True)

                cls_counters = {}
                for cls_name in class_names:
                    existing = list((cls_root / cls_name).glob("*.png"))
                    max_num = 0
                    for f in existing:
                        try:
                            num = int(f.stem.split('_')[-1])
                            if num > max_num:
                                max_num = num
                        except:
                            pass
                    cls_counters[cls_name] = max_num + 1

                orig_w, orig_h = img.size
                for ann in anns:
                    cat_name = ann['category_name']
                    if cat_name not in class_names:
                        continue
                    # 决定分类裁剪框
                    if self.zoom_cell_var.get():
                        # 使用红色标注框，并强制为正方形（以标注框中心，边长为最大边长）
                        x, y, w, h = ann['bbox']
                        size = max(w, h)
                        cx = x + w // 2
                        cy = y + h // 2
                        half = size // 2
                        x1_new = max(0, cx - half)
                        y1_new = max(0, cy - half)
                        x2_new = min(orig_w, cx + half)
                        y2_new = min(orig_h, cy + half)
                        # 如果边界导致宽高不等，用较小值重新调整
                        if x2_new - x1_new != y2_new - y1_new:
                            size = min(x2_new - x1_new, y2_new - y1_new)
                            half = size // 2
                            cx = (x1_new + x2_new) // 2
                            cy = (y1_new + y2_new) // 2
                            x1_new = max(0, cx - half)
                            y1_new = max(0, cy - half)
                            x2_new = min(orig_w, cx + half)
                            y2_new = min(orig_h, cy + half)
                        cls_box = (int(x1_new), int(y1_new), int(x2_new), int(y2_new))
                    else:
                        # 使用蓝色裁剪框
                        cls_box = ann.get('cls_crop_box')
                        if cls_box is None:
                            # 默认生成（以标注框中心）
                            x, y, w, h = ann['bbox']
                            center_x = x + w // 2
                            center_y = y + h // 2
                            half = CONTRASTIVE_IMAGE_SIZE // 2
                            cls_box = (center_x - half, center_y - half, center_x + half, center_y + half)
                            cls_box = (max(0, cls_box[0]), max(0, cls_box[1]),
                                       min(orig_w, cls_box[2]), min(orig_h, cls_box[3]))
                            if cls_box[2] - cls_box[0] < CONTRASTIVE_IMAGE_SIZE:
                                if cls_box[0] == 0:
                                    cls_box = (0, cls_box[1], CONTRASTIVE_IMAGE_SIZE, cls_box[1] + CONTRASTIVE_IMAGE_SIZE)
                                elif cls_box[2] == orig_w:
                                    cls_box = (orig_w - CONTRASTIVE_IMAGE_SIZE, cls_box[1], orig_w, cls_box[1] + CONTRASTIVE_IMAGE_SIZE)
                            if cls_box[3] - cls_box[1] < CONTRASTIVE_IMAGE_SIZE:
                                if cls_box[1] == 0:
                                    cls_box = (cls_box[0], 0, cls_box[0] + CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE)
                                elif cls_box[3] == orig_h:
                                    cls_box = (cls_box[0], orig_h - CONTRASTIVE_IMAGE_SIZE, cls_box[0] + CONTRASTIVE_IMAGE_SIZE, orig_h)
                    # 裁剪并缩放（直接从原图裁剪）
                    cls_crop_img = img.crop(cls_box)
                    cls_crop_img = cls_crop_img.resize((CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE), Image.Resampling.LANCZOS)
                    save_dir = cls_root / cat_name
                    save_dir.mkdir(exist_ok=True)
                    idx = cls_counters[cat_name]
                    save_path = save_dir / f"{cat_name}_{idx:04d}.png"
                    cls_crop_img.save(save_path)
                    cls_counters[cat_name] = idx + 1

        except Exception as e:
            print(f"导出图片 {img_path.name} 失败: {e}")

    def prev_image(self):
        if self.current_idx > 0:
            # 保存当前图片的信息（用于后台导出）
            cur_img_path = self.image_list[self.current_idx]
            cur_img_name = cur_img_path.name
            cur_anns = self.annotations.get(cur_img_name, [])
            cur_crop = self.crop_box
            # 更新索引
            self.current_idx -= 1
            # 清理临时标注状态
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None
            self.temp_bbox = None
            self.temp_img_name = None
            # 立即显示新图片
            self.show_image()
            self.canvas.focus_set()
            # 异步导出旧图片（若存在标注和有效裁剪框）
            if cur_anns and cur_crop != (0, 0, 0, 0):
                threading.Thread(
                    target=self._export_image_data,
                    args=(cur_img_path, cur_anns, cur_crop, self.class_names),
                    daemon=True
                ).start()

    def next_image(self):
        if self.current_idx < len(self.image_list) - 1:
            # 保存当前图片的信息（用于后台导出）
            cur_img_path = self.image_list[self.current_idx]
            cur_img_name = cur_img_path.name
            cur_anns = self.annotations.get(cur_img_name, [])
            cur_crop = self.crop_box
            # 更新索引
            self.current_idx += 1
            # 清理临时标注状态
            if self.rect_id:
                self.canvas.delete(self.rect_id)
                self.rect_id = None
            self.temp_bbox = None
            self.temp_img_name = None
            # 立即显示新图片
            self.show_image()
            self.canvas.focus_set()
            # 异步导出旧图片（若存在标注和有效裁剪框）
            if cur_anns and cur_crop != (0, 0, 0, 0):
                threading.Thread(
                    target=self._export_image_data,
                    args=(cur_img_path, cur_anns, cur_crop, self.class_names),
                    daemon=True
                ).start()

    # ---------- 保存标注和生成数据集（应用裁剪） ----------
    def save_annotations(self):
        folder = self.image_folder.get()
        if not folder:
            messagebox.showerror("错误", "请先选择文件夹")
            return
        ann_file = Path(folder) / "annotations.json"
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": i+1, "name": name} for i, name in enumerate(self.class_names)]
        }
        annotated_images = set(self.annotations.keys())
        if not annotated_images:
            self.update_status("当前没有标注，无法保存")
            return

        # 建立文件名到图片尺寸的映射（使用裁剪后的尺寸 IMAGE_SIZE）
        img_info_map = {}
        for img_path in self.image_list:
            img_name = img_path.name
            if img_name in annotated_images:
                img_info_map[img_name] = {"width": IMAGE_SIZE, "height": IMAGE_SIZE}

        image_id_map = {}
        for idx, img_name in enumerate(sorted(annotated_images)):
            image_id = idx + 1
            image_id_map[img_name] = image_id
            coco_data["images"].append({
                "id": image_id,
                "file_name": img_name,
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE
            })

        ann_id = 1
        for img_name, anns in self.annotations.items():
            image_id = image_id_map.get(img_name)
            if image_id is None:
                continue
            for ann in anns:
                # 注意：bbox 需要根据裁剪框进行坐标变换
                # 这里我们保存原始坐标（相对于原图），但在生成数据集时会裁剪图片，坐标也需变换
                # 为简化，此处保存原始坐标，生成数据集时再变换
                coco_data["annotations"].append({
                    "image_id": image_id,
                    "bbox": ann['bbox'],  # 暂存原始坐标
                    "category_id": ann['category_id'],
                    "area": ann['bbox'][2] * ann['bbox'][3],
                    "iscrowd": 0,
                    "id": ann_id
                })
                ann_id += 1

        with open(ann_file, 'w') as f:
            json.dump(coco_data, f, indent=2)
        self.update_status(f"标注已保存到 {ann_file}")

    def generate_dataset(self):
        """生成完整数据集：检测 + 分类，应用裁剪框"""
        self.save_annotations()
        src_folder = Path(self.image_folder.get())
        dst_det_images = DATASET_DIR / DATA_DETECTION / "images"
        dst_det_images.mkdir(parents=True, exist_ok=True)
        ann_src = src_folder / "annotations.json"
        ann_dst = DATASET_DIR / DATA_DETECTION / "annotations.json"

        annotated_images = set(self.annotations.keys())
        if not annotated_images:
            messagebox.showerror("错误", "没有标注任何图片，无法生成完整数据集")
            return

        import shutil
        from PIL import Image

        # 读取已有的裁剪框信息（如果有保存的话），这里我们使用当前显示的裁剪框
        # 但每张图片的裁剪框可能不同，我们为每张图片保存裁剪框信息
        # 为了简化，我们使用当前裁剪框（即最后显示的裁剪框）应用于所有图片？不合理。
        # 更好的做法：每张图片独立保存裁剪框，这里我们使用当前裁剪框，但用户可能为每张图片调整。
        # 我们添加功能：保存标注时同时保存裁剪框信息，但暂不实现，先使用当前裁剪框统一处理。
        # 为演示，我们直接使用 self.crop_box 对所有图片裁剪，但用户可能不满意。
        # 我们提供一个简单方案：在生成数据集时，让用户确认或使用当前裁剪框。
        # 但由于需求是“每次标注完都自动裁剪”，更合理的是在标注完成后（比如切换图片时）自动裁剪并保存裁剪后的图片？
        # 我们采用另一种方式：在点击“生成完整数据集”时，根据每张图片的当前裁剪框（用户调整）进行裁剪。
        # 但不同图片的裁剪框可能不同，我们需要存储每张图片的裁剪框。
        # 我们在 show_image 时初始化裁剪框，但用户调整后，我们需要保存调整结果。
        # 我们可以在 self.annotations 中额外存储裁剪框？或者简单起见，统一使用当前裁剪框。
        # 更合理：在标注工具中，裁剪框是全局的，应用于所有图片（可能用户希望统一裁剪）。
        # 如果用户需要每张图片不同，可以手动调整。我们采用全局裁剪框，所有图片统一裁剪。

        # 读取当前裁剪框（原图坐标）
        crop_box = self.crop_box
        if crop_box == (0,0,0,0):
            self._init_crop_box()
            crop_box = self.crop_box

        # 复制并裁剪图片
        for img_name in annotated_images:
            src_path = src_folder / img_name
            if not src_path.exists():
                continue
            img = Image.open(src_path).convert('RGB')
            # 裁剪
            cropped = img.crop(crop_box)
            # 缩放到 IMAGE_SIZE
            cropped = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            dst_path = dst_det_images / img_name
            cropped.save(dst_path)

        # 复制标注文件（需要转换坐标）
        if ann_src.exists():
            with open(ann_src, 'r') as f:
                coco_data = json.load(f)
            # 转换 bbox 坐标：从原图坐标映射到裁剪后的坐标
            # 裁剪框 (x1, y1, x2, y2)
            x1, y1, x2, y2 = crop_box
            crop_w = x2 - x1
            crop_h = y2 - y1
            for ann in coco_data['annotations']:
                bbox = ann['bbox']  # [x, y, w, h] 原图坐标
                # 映射到裁剪框内
                new_x = (bbox[0] - x1) / crop_w * IMAGE_SIZE
                new_y = (bbox[1] - y1) / crop_h * IMAGE_SIZE
                new_w = bbox[2] / crop_w * IMAGE_SIZE
                new_h = bbox[3] / crop_h * IMAGE_SIZE
                ann['bbox'] = [new_x, new_y, new_w, new_h]
                ann['area'] = new_w * new_h
            # 更新图像尺寸
            for img in coco_data['images']:
                img['width'] = IMAGE_SIZE
                img['height'] = IMAGE_SIZE
            with open(ann_dst, 'w') as f:
                json.dump(coco_data, f, indent=2)

        # ---------- 生成分类数据集（同样裁剪） ----------
        cls_root = DATASET_DIR / DATA_CLASSIFICATION
        cls_root.mkdir(parents=True, exist_ok=True)
        for cls_name in self.class_names:
            (cls_root / cls_name).mkdir(exist_ok=True)

        cls_counters = {}
        for cls_name in self.class_names:
            existing = list((cls_root / cls_name).glob("*.png"))
            max_num = 0
            for f in existing:
                try:
                    num = int(f.stem.split('_')[-1])
                    if num > max_num:
                        max_num = num
                except:
                    pass
            cls_counters[cls_name] = max_num + 1

        total_cls_images = 0
        for img_name, anns in self.annotations.items():
            src_path = src_folder / img_name
            if not src_path.exists():
                continue
            img = Image.open(src_path).convert('RGB')
            # 裁剪原图
            cropped_img = img.crop(crop_box)
            # 缩放
            cropped_img = cropped_img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
            for ann in anns:
                bbox = ann['bbox']
                cat_name = ann['category_name']
                if cat_name not in self.class_names:
                    continue
                # 在原图坐标下裁剪细胞，然后映射到裁剪后的坐标
                x, y, w, h = bbox
                # 映射到裁剪框内
                new_x = (x - x1) / crop_w * IMAGE_SIZE
                new_y = (y - y1) / crop_h * IMAGE_SIZE
                new_w = w / crop_w * IMAGE_SIZE
                new_h = h / crop_h * IMAGE_SIZE
                # 裁剪细胞（从裁剪后的图片上裁剪）
                # 但为了更准确，应该从原图裁剪然后缩放，但简单起见从裁剪后图片裁剪
                # 由于坐标已变换，可以直接从 cropped_img 裁剪
                # 注意：cropped_img 已经是 IMAGE_SIZE 大小
                cell_box = (int(new_x), int(new_y), int(new_x + new_w), int(new_y + new_h))
                # 确保在范围内
                cell_box = (max(0, cell_box[0]), max(0, cell_box[1]),
                            min(IMAGE_SIZE, cell_box[2]), min(IMAGE_SIZE, cell_box[3]))
                if cell_box[2] <= cell_box[0] or cell_box[3] <= cell_box[1]:
                    continue
                cell_img = cropped_img.crop(cell_box)
                save_dir = cls_root / cat_name
                save_dir.mkdir(exist_ok=True)
                idx = cls_counters[cat_name]
                save_path = save_dir / f"{cat_name}_{idx:04d}.png"
                cell_img.save(save_path)
                cls_counters[cat_name] = idx + 1
                total_cls_images += 1

        messagebox.showinfo(
            "成功",
            f"完整数据集已生成（已按裁剪框裁剪）：\n"
            f"检测数据集：{len(annotated_images)} 张图片 → {DATASET_DIR / DATA_DETECTION}\n"
            f"分类数据集：{total_cls_images} 张裁剪细胞 → {cls_root}"
        )

# ======================== 主窗口 ========================
class MainApp:
    def __init__(self, root):
        self.root = root
        root.title("细胞识别系统")
        root.geometry("1200x700")
        # 如果是 TkinterDnD 窗口，启用拖放
        if DND_AVAILABLE and isinstance(root, tkinterdnd2.Tk):
            root.drop_target_register('DND_Files')
            root.dnd_bind('<<Drop>>', self.on_drop)
        elif DND_AVAILABLE:
            # 如果 root 不是 TkinterDnD 类型，尝试包装
            pass  # 实际中如果用户用普通 Tk，则拖拽不可用

        # 笔记本（标签页）
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 训练标签页
        self.train_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.train_frame, text="训练")

        # 识别标签页
        self.infer_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.infer_frame, text="识别")

        # ---------- 裁剪相关属性（识别预览） ----------
        self.crop_box = (0, 0, 0, 0)          # 裁剪框在原图坐标 (x1,y1,x2,y2)
        self.crop_size = IMAGE_SIZE            # 固定边长
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_orig_box = (0, 0, 0, 0)
        self.current_image_path = None         # 当前加载的图片路径
        self.orig_image = None                 # PIL Image 对象
        self.infer_mode = 'idle'               # 'idle', 'preview', 'result'
        self.scale = 1.0                       # 画布缩放比例
        self.orig_size = (0, 0)                # 原图尺寸
        self.infer_tk_img = None               # 用于保持图片引用

        # 初始化指示灯定时器ID（必须在 build_train_tab 之前）
        self.indicator_timer_id = None
        self.training_queue = []
        self.current_training_index = -1
        self.training_blink_ids = {}
        self.script_to_weight_key = {
            'train_detection': WEIGHT_DETECTION,
            'train_contrastive': WEIGHT_CONTRASTIVE,
            'train_proxy': WEIGHT_PROXY
        }
        self.current_process = None          # 当前运行的子进程
        self.disabled_btns = []              # 被禁用的训练按钮键名列表
        self.log_file = None                 # 训练日志文件对象

        self.build_train_tab()
        self.build_infer_tab()
        self.build_edit_tab()

        # 初次检查状态（build_train_tab 内已经调用 update_status_indicators，但显式再调一次确保）
        self.update_status_indicators()

    # -------------------- 训练标签页 --------------------
    def build_train_tab(self):
        frame = self.train_frame
        # 左侧控制区
        left_panel = ttk.Frame(frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)

        # ---------- 状态指示器区域 ----------
        # 整体容器：上下结构，上为标题+右侧大灯，下为四行状态+操作按钮
        status_container = ttk.Frame(left_panel)
        status_container.pack(fill=tk.X, pady=5)

        # 标题行：左侧文字，右侧大指示灯
        header_frame = ttk.Frame(status_container)
        header_frame.pack(fill=tk.X)
        ttk.Label(header_frame, text="", font=('Arial', 14, 'bold')).pack(side=tk.LEFT)
        # 大指示灯放在右侧
        indicator_frame = ttk.Frame(header_frame)
        indicator_frame.pack(side=tk.RIGHT, padx=5)
        self.main_indicator_canvas = tk.Canvas(indicator_frame, width=70, height=70, highlightthickness=0, bg='#f0f0f0')
        self.main_indicator_canvas.pack()
        # 外发光光圈
        self.glow_id = self.main_indicator_canvas.create_oval(5,5,65,65, outline='#88ff88', width=4, stipple='gray50')
        # 主圆
        self.main_indicator_circle = self.main_indicator_canvas.create_oval(10,10,60,60, fill='gray', outline='')
        self.main_indicator_label = ttk.Label(indicator_frame, text="", font=('Arial', 8))
        self.main_indicator_label.pack()

        # 四行状态+按钮（不再需要单独的 dot_container）
        self.status_vars = {}
        self.train_btns = {}
        status_items = [
            ("数据集", "dataset", "数据标注", self.open_annotation_tool, "dataset"),
            ("检测器权重", WEIGHT_DETECTION, "训练检测器", lambda: self.run_training("train_detection"), "train_detection"),
            ("对比器权重", WEIGHT_CONTRASTIVE, "训练对比器", lambda: self.run_training("train_contrastive"), "train_contrastive"),
            ("代理器权重", WEIGHT_PROXY, "训练代理器", lambda: self.run_training("train_proxy"), "train_proxy")
        ]
        for label, key, btn_text, cmd, cmd_key in status_items:
            row_frame = ttk.Frame(status_container)
            row_frame.pack(fill=tk.X, pady=2)
            # 状态指示：圆点
            canvas = tk.Canvas(row_frame, width=20, height=20, highlightthickness=0)
            canvas.pack(side=tk.LEFT)
            circle = canvas.create_oval(2,2,18,18, fill='gray', outline='')
            self.status_vars[key] = (canvas, circle)
            # 标签
            ttk.Label(row_frame, text=label, font=('Arial', 9)).pack(side=tk.LEFT, padx=5)
            # 弹性空间（将按钮推到右侧）
            ttk.Frame(row_frame).pack(side=tk.LEFT, expand=True, fill=tk.X)
            # 操作按钮
            if key == "dataset":
                # 两个按钮：标准化数据集 + 数据标注（同一行）
                btn_container = ttk.Frame(row_frame)
                btn_container.pack(side=tk.RIGHT, padx=2)
                btn_norm = ttk.Button(btn_container, text="标准化数据集", command=self.normalize_dataset, width=12)
                btn_norm.pack(side=tk.LEFT, padx=2)
                btn_ann = ttk.Button(btn_container, text="数据标注", command=cmd, width=12)
                btn_ann.pack(side=tk.LEFT, padx=2)
            else:
                # 训练按钮 + 删除按钮（同一行）
                btn_container = ttk.Frame(row_frame)
                btn_container.pack(side=tk.RIGHT, padx=2)
                # 删除按钮（垃圾桶图标）
                del_btn = ttk.Button(btn_container, text="🗑", width=3,
                                     command=lambda k=key: self.delete_weight(k))
                del_btn.pack(side=tk.RIGHT, padx=2)
                # 训练按钮
                train_btn = ttk.Button(btn_container, text=btn_text, command=cmd, width=20)
                train_btn.pack(side=tk.RIGHT, padx=2)
                if key in (WEIGHT_DETECTION, WEIGHT_CONTRASTIVE, WEIGHT_PROXY):
                    self.train_btns[cmd_key] = train_btn

        # 图片尺寸调整区域已移除
        pass

        # 下方按钮区域（检测数据集完整性 + 打开配置文件）
        bottom_btn_frame = ttk.Frame(left_panel)
        bottom_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(bottom_btn_frame, text="检测数据集完整性", 
                   command=self.update_status_indicators, width=20).pack(side=tk.LEFT, padx=5)
        ttk.Button(bottom_btn_frame, text="打开配置文件", 
                   command=self.open_config_file, width=20).pack(side=tk.LEFT, padx=5)

        # 一键训练区域
        train_control_frame = ttk.Frame(left_panel)
        train_control_frame.pack(fill=tk.X, pady=5)
        self.btn_one_click_train = ttk.Button(train_control_frame, text="一键训练", 
                                              command=self.start_sequential_training, width=15)
        self.btn_one_click_train.pack(side=tk.LEFT, padx=5)
        self.shutdown_var = tk.BooleanVar(value=False)
        self.chk_shutdown = ttk.Checkbutton(train_control_frame, text="训练完成后关机", 
                                            variable=self.shutdown_var)
        self.chk_shutdown.pack(side=tk.LEFT, padx=10)

        # 终止训练按钮（位于一键训练下方）
        stop_frame = ttk.Frame(left_panel)
        stop_frame.pack(fill=tk.X, pady=2)
        self.btn_stop_train = ttk.Button(stop_frame, text="终止训练", 
                                         command=self.terminate_training, width=15, state=tk.DISABLED)
        self.btn_stop_train.pack(side=tk.LEFT, padx=5)

        # 右侧日志区
        right_panel = ttk.Frame(frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ttk.Label(right_panel, text="训练日志", font=('Arial', 12, 'bold')).pack(anchor=tk.W)
        self.log_text = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, font=('Consolas', 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # 启动时检查状态
        self.update_status_indicators()

    def update_status_indicators(self):
        """刷新四个状态指示灯并更新大指示灯的闪烁模式"""
        # 检查各状态
        status = {
            "dataset": check_dataset(),
            WEIGHT_DETECTION: check_weight(WEIGHT_DETECTION),
            WEIGHT_CONTRASTIVE: check_weight(WEIGHT_CONTRASTIVE),
            WEIGHT_PROXY: check_weight(WEIGHT_PROXY)
        }
        # 更新四个小圆点
        all_green = True
        for key, (canvas, circle) in self.status_vars.items():
            # 如果该按键正在闪烁（训练中），跳过更新
            if key in self.training_blink_ids:
                continue
            ok = status.get(key, False)
            color = 'green' if ok else 'red'
            canvas.itemconfig(circle, fill=color)
            if not ok:
                all_green = False

        # 设置大指示灯的闪烁模式
        self.indicator_mode = 'green' if all_green else 'redgreen'

        # 启动或继续闪烁循环
        self._start_indicator_cycle()

    def _start_indicator_cycle(self):
        """启动指示灯闪烁循环（若已启动则先取消）"""
        if self.indicator_timer_id is not None:
            self.root.after_cancel(self.indicator_timer_id)
            self.indicator_timer_id = None
        # 立即执行第一次切换
        self._cycle_indicator()

    def _cycle_indicator(self):
        """指示灯闪烁循环，每500ms切换一次"""
        if not self.root.winfo_exists():
            return
        # 获取当前颜色
        current_color = self.main_indicator_canvas.itemcget(self.main_indicator_circle, 'fill')
        if self.indicator_mode == 'green':
            # 绿色闪烁：绿/灰交替
            new_color = 'green' if current_color == 'gray' else 'gray'
        else:  # redgreen
            # 红绿交替：红/绿切换
            new_color = 'red' if current_color == 'green' else 'green'
        self.main_indicator_canvas.itemconfig(self.main_indicator_circle, fill=new_color)
        # 定时下一次
        self.indicator_timer_id = self.root.after(500, self._cycle_indicator)

    def _start_blink(self, key):
        """启动指定按键对应圆点的红绿闪烁"""
        if key not in self.status_vars:
            return
        canvas, circle = self.status_vars[key]
        # 取消已有的闪烁
        self._stop_blink(key)
        # 开始闪烁
        def toggle():
            if key not in self.status_vars:
                return
            current = canvas.itemcget(circle, 'fill')
            new_color = 'green' if current == 'red' else 'red'
            canvas.itemconfig(circle, fill=new_color)
            if key in self.training_blink_ids:
                self.training_blink_ids[key] = self.root.after(500, toggle)
        self.training_blink_ids[key] = self.root.after(0, toggle)

    def _stop_blink(self, key):
        """停止指定按键的闪烁"""
        if key in self.training_blink_ids:
            self.root.after_cancel(self.training_blink_ids[key])
            del self.training_blink_ids[key]
        # 停止闪烁后，重新调用 update_status_indicators 以设置正确颜色
        self.update_status_indicators()

    def start_sequential_training(self):
        if self.training_queue:
            messagebox.showinfo("提示", "训练正在进行中，请等待完成")
            return
        self.terminated_by_user = False  # 重置终止标志
        # 创建日志文件夹和文件
        log_dir = BASE_DIR / "log"
        log_dir.mkdir(exist_ok=True)
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"train_{timestamp}.log"
        self.log_file = open(log_path, 'w', encoding='utf-8')
        # 写入开始标记
        self.log_file.write(f"=== 一键训练开始 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.log_file.flush()
        # 禁用一键训练按钮
        self.btn_one_click_train.config(state=tk.DISABLED)
        # 设置训练队列
        self.training_queue = ['train_detection', 'train_contrastive', 'train_proxy']
        self.current_training_index = 0
        # 启动第一个
        self._run_next_training()

    def _run_next_training(self):
        if self.terminated_by_user:
            return  # 已终止，不再继续
        if self.current_training_index >= len(self.training_queue):
            # 所有训练完成
            self.training_queue = []
            self.current_training_index = -1
            self.btn_one_click_train.config(state=tk.NORMAL)
            # 检查是否关机
            if self.shutdown_var.get():
                self._shutdown_system()
            return
        script_key = self.training_queue[self.current_training_index]
        # 启动闪烁（使用权重键）
        weight_key = self.script_to_weight_key.get(script_key)
        if weight_key:
            self._start_blink(weight_key)
        # 调用训练（run_training 会禁用对应的训练按钮）
        self.run_training(script_key)

    def _shutdown_system(self):
        """执行关机"""
        try:
            if sys.platform == 'win32':
                os.system('shutdown /s /t 5')
            elif sys.platform == 'darwin':
                os.system('sudo shutdown -h +1')
            else:
                os.system('shutdown -h +1')
            messagebox.showinfo("提示", "系统将在5秒后关机")
        except Exception as e:
            messagebox.showerror("错误", f"关机失败：{e}")

    def terminate_training(self):
        """终止当前正在运行的训练任务"""
        if self.current_process is None:
            return
        # 弹窗确认
        if not messagebox.askyesno("确认终止", "确定要终止当前训练任务吗？", icon='warning'):
            return
        try:
            self.current_process.terminate()
            self.terminated_by_user = True
            # 重置队列状态
            self.training_queue = []
            self.current_training_index = -1
            # 恢复所有训练按钮
            for key in self.train_btns:
                self.train_btns[key].config(state=tk.NORMAL)
            self.disabled_btns = []
            # 禁用终止按钮
            self.btn_stop_train.config(state=tk.DISABLED)
            # 恢复一键训练按钮
            self.btn_one_click_train.config(state=tk.NORMAL)
            # 停止所有闪烁
            for key in list(self.training_blink_ids.keys()):
                self._stop_blink(key)
            self.update_status_indicators()
            self.log_text.insert(tk.END, ">>> 用户终止了训练任务\n")
            self.log_text.see(tk.END)
            # 关闭日志文件
            if self.log_file:
                try:
                    self.log_file.close()
                except:
                    pass
                self.log_file = None
        except Exception as e:
            messagebox.showerror("错误", f"终止失败：{e}")

    def update_size(self):
        try:
            new_size = int(self.size_var.get())
            if new_size < 32:
                messagebox.showerror("错误", "尺寸不能小于32")
                return
            update_config_image_size(new_size)
            # 提示重启
        except ValueError:
            messagebox.showerror("错误", "请输入整数")

    def open_config_file(self):
        """使用系统默认编辑器打开 config.py"""
        config_path = BASE_DIR / "script" / "config.py"
        if not config_path.exists():
            messagebox.showerror("错误", f"config.py 不存在：{config_path}")
            return
        try:
            if sys.platform == 'win32':
                os.startfile(config_path)
            elif sys.platform == 'darwin':
                subprocess.run(['open', config_path])
            else:
                subprocess.run(['xdg-open', config_path])
        except Exception as e:
            messagebox.showerror("错误", f"无法打开配置文件：{e}")

    def open_annotation_tool(self):
        AnnotationTool(self.root)

    def delete_weight(self, weight_type):
        """删除指定类型的权重文件夹（带确认弹窗）"""
        weight_dir = WEIGHT_DIR / weight_type
        if not weight_dir.exists():
            messagebox.showwarning("警告", f"权重文件夹不存在：{weight_dir}")
            return

        # 确认删除
        reply = messagebox.askyesno(
            "确认删除",
            f"确定要删除 {weight_type} 权重文件夹吗？\n路径：{weight_dir}\n此操作不可恢复！",
            icon='warning'
        )
        if not reply:
            return

        try:
            import shutil
            shutil.rmtree(weight_dir)
            messagebox.showinfo("成功", f"已删除 {weight_type} 权重文件夹")
            # 刷新状态指示灯
            self.update_status_indicators()
        except Exception as e:
            messagebox.showerror("错误", f"删除失败：{e}")

    def normalize_dataset(self):
        """标准化数据集：将 detection/images 下的图片缩放到 IMAGE_SIZE，并同步更新 annotations.json；
        同时将 classification 子目录下的图片裁剪为最大正方形（1:1）"""
        from PIL import Image
        import json
        import shutil

        det_img_dir = DATASET_DIR / DATA_DETECTION / "images"
        ann_file = DATASET_DIR / DATA_DETECTION / "annotations.json"

        if not det_img_dir.exists():
            messagebox.showerror("错误", f"检测图片目录不存在：{det_img_dir}")
            return
        if not ann_file.exists():
            messagebox.showerror("错误", f"标注文件不存在：{ann_file}")
            return

        # 读取标注文件
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            messagebox.showerror("错误", f"读取标注文件失败：{e}")
            return

        # 备份原标注文件
        backup_file = ann_file.with_suffix(".json.backup")
        shutil.copy2(ann_file, backup_file)

        # 构建文件名到图片记录和标注的映射
        img_id_map = {img['id']: img for img in data.get('images', [])}
        file_to_anns = {}
        for ann in data.get('annotations', []):
            img_id = ann['image_id']
            img_rec = img_id_map.get(img_id)
            if img_rec:
                fname = img_rec['file_name']
                file_to_anns.setdefault(fname, []).append(ann)

        # 遍历检测图片文件
        img_paths = list(det_img_dir.glob("*"))
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif'}
        total = 0
        updated = 0
        for img_path in img_paths:
            if img_path.suffix.lower() not in img_exts:
                continue
            total += 1
            try:
                with Image.open(img_path) as img:
                    old_w, old_h = img.size
                    if old_w == IMAGE_SIZE and old_h == IMAGE_SIZE:
                        continue
                    # 缩放
                    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
                    img_resized.save(img_path)
                    # 更新标注
                    fname = img_path.name
                    if fname in file_to_anns:
                        scale_x = IMAGE_SIZE / old_w
                        scale_y = IMAGE_SIZE / old_h
                        # 更新图片记录
                        for img_rec in data['images']:
                            if img_rec['file_name'] == fname:
                                img_rec['width'] = IMAGE_SIZE
                                img_rec['height'] = IMAGE_SIZE
                                break
                        # 更新该图片的所有标注
                        for ann in file_to_anns[fname]:
                            bbox = ann.get('bbox')
                            if bbox and len(bbox) == 4:
                                # bbox: [x, y, w, h]
                                bbox[0] = round(bbox[0] * scale_x, 2)
                                bbox[1] = round(bbox[1] * scale_y, 2)
                                bbox[2] = round(bbox[2] * scale_x, 2)
                                bbox[3] = round(bbox[3] * scale_y, 2)
                                if 'area' in ann:
                                    ann['area'] = round(bbox[2] * bbox[3], 2)
                    updated += 1
            except Exception as e:
                self.log_text.insert(tk.END, f"处理检测图片 {img_path.name} 失败: {e}\n")
                self.log_text.see(tk.END)

        # 写回标注文件
        try:
            with open(ann_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            messagebox.showerror("错误", f"写入标注文件失败：{e}")
            return

        # ---------- 新增：处理分类图片，裁剪为1:1正方形并缩放到 CONTRASTIVE_IMAGE_SIZE ----------
        cls_root = DATASET_DIR / DATA_CLASSIFICATION
        cls_total = 0
        cls_updated = 0
        if cls_root.exists():
            for cls_dir in cls_root.iterdir():
                if cls_dir.is_dir():
                    for img_path in cls_dir.glob("*"):
                        if img_path.suffix.lower() in img_exts:
                            cls_total += 1
                            try:
                                with Image.open(img_path) as img:
                                    w, h = img.size
                                    # 中心裁剪为最大正方形
                                    size = min(w, h)
                                    left = (w - size) // 2
                                    top = (h - size) // 2
                                    cropped = img.crop((left, top, left + size, top + size))
                                    # 缩放到 CONTRASTIVE_IMAGE_SIZE
                                    if size != CONTRASTIVE_IMAGE_SIZE:
                                        cropped = cropped.resize((CONTRASTIVE_IMAGE_SIZE, CONTRASTIVE_IMAGE_SIZE), Image.Resampling.LANCZOS)
                                    cropped.save(img_path)
                                    cls_updated += 1
                            except Exception as e:
                                self.log_text.insert(tk.END, f"处理分类图片 {img_path.name} 失败: {e}\n")
                                self.log_text.see(tk.END)
            if cls_total > 0:
                self.log_text.insert(tk.END, f"分类图片处理完成：共 {cls_total} 张，已缩放为 {CONTRASTIVE_IMAGE_SIZE}x{CONTRASTIVE_IMAGE_SIZE}。\n")
                self.log_text.see(tk.END)
        else:
            self.log_text.insert(tk.END, f"分类目录不存在，跳过：{cls_root}\n")
            self.log_text.see(tk.END)

        # 保持原消息框内容不变（仅提示检测图片）
        messagebox.showinfo("完成", f"标准化完成！\n共处理 {total} 张检测图片，其中 {updated} 张被缩放。\n原标注文件已备份为 {backup_file.name}")
        self.update_status_indicators()  # 刷新状态指示灯

    def run_training(self, script_key):
        script_map = {
            "train_detection": "train_detection.py",
            "train_contrastive": "train_contrastive.py",
            "train_proxy": "train_proxy.py"
        }
        script_name = script_map.get(script_key)
        if not script_name:
            return

        # 记录本次需要禁用的按钮
        self.disabled_btns = []
        if self.training_queue:  # 一键训练模式，禁用所有训练按钮
            for key in self.train_btns:
                self.train_btns[key].config(state=tk.DISABLED)
                self.disabled_btns.append(key)
        else:  # 单独训练模式
            if script_key == 'train_detection':
                self.train_btns['train_detection'].config(state=tk.DISABLED)
                self.disabled_btns.append('train_detection')
            elif script_key == 'train_contrastive':
                self.train_btns['train_contrastive'].config(state=tk.DISABLED)
                self.train_btns['train_proxy'].config(state=tk.DISABLED)
                self.disabled_btns.extend(['train_contrastive', 'train_proxy'])
            elif script_key == 'train_proxy':
                self.train_btns['train_proxy'].config(state=tk.DISABLED)
                self.disabled_btns.append('train_proxy')

        # 启用终止按钮
        self.btn_stop_train.config(state=tk.NORMAL)

        self.log_text.insert(tk.END, f">>> 开始训练 {script_name} ...\n")
        self.log_text.see(tk.END)

        def run():
            script_path = BASE_DIR / "script" / script_name
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(BASE_DIR / "script")
            )
            self.current_process = proc  # 保存进程引用
            for line in proc.stdout:
                self.log_text.insert(tk.END, line)
                self.log_text.see(tk.END)
                # 写入日志文件
                if self.log_file:
                    try:
                        self.log_file.write(line)
                        self.log_file.flush()
                    except:
                        pass
            proc.stdout.close()
            return_code = proc.wait()
            self.current_process = None  # 进程结束
            self.root.after(0, lambda: self.training_finished(script_key, return_code))

        threading.Thread(target=run, daemon=True).start()

    def training_finished(self, script_key, code):
        # 恢复所有被禁用的按钮
        for key in self.disabled_btns:
            self.train_btns[key].config(state=tk.NORMAL)
        self.disabled_btns = []
        # 禁用终止按钮
        self.btn_stop_train.config(state=tk.DISABLED)

        # 停止闪烁（使用权重键）
        weight_key = self.script_to_weight_key.get(script_key)
        if weight_key:
            self._stop_blink(weight_key)

        # 如果被用户终止，不继续后续训练，不触发关机
        if self.terminated_by_user:
            self.terminated_by_user = False
            self.training_queue = []
            self.current_training_index = -1
            self.btn_one_click_train.config(state=tk.NORMAL)
            self.log_text.insert(tk.END, f">>> {script_key} 已被用户终止\n")
            self.log_text.see(tk.END)
            self.update_status_indicators()
            # 关闭日志文件
            if self.log_file:
                try:
                    self.log_file.close()
                except:
                    pass
                self.log_file = None
            return

        if code == 0:
            self.log_text.insert(tk.END, f">>> {script_key} 训练完成！\n")
            self.update_status_indicators()
        else:
            self.log_text.insert(tk.END, f">>> {script_key} 训练失败，返回码 {code}\n")
            # 失败则中断一键训练
            self.training_queue = []
            self.current_training_index = -1
            self.btn_one_click_train.config(state=tk.NORMAL)
            self.update_status_indicators()
            self.log_text.see(tk.END)
            # 关闭日志文件
            if self.log_file:
                try:
                    self.log_file.close()
                except:
                    pass
                self.log_file = None
            return
        # 如果是一键训练模式，继续下一个
        if self.training_queue:
            self.current_training_index += 1
            self._run_next_training()
        else:
            # 所有训练正常完成，关闭日志文件
            if self.log_file:
                try:
                    self.log_file.close()
                except:
                    pass
                self.log_file = None
        self.log_text.see(tk.END)

    # -------------------- 识别标签页 --------------------
    def build_infer_tab(self):
        frame = self.infer_frame

        # 左侧控制面板
        left_panel = ttk.Frame(frame, width=280)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        left_panel.pack_propagate(False)

        ttk.Label(left_panel, text="", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)

        # 拖拽接收区域（替代单张识别按钮）
        drop_frame = ttk.LabelFrame(left_panel, text="单张识别")
        drop_frame.pack(fill=tk.X, pady=5)
        self.drop_label = ttk.Label(drop_frame, text="将图片文件拖入此区域", anchor=tk.CENTER, background='#f0f0f0')
        self.drop_label.pack(fill=tk.X, padx=5, pady=10, ipadx=10, ipady=20)
        if DND_AVAILABLE and isinstance(self.root, tkinterdnd2.Tk):
            self.drop_label.drop_target_register('DND_Files')
            self.drop_label.dnd_bind('<<Drop>>', self.on_drop)
            self.drop_label.config(text="拖拽图片文件到此自动识别")
        else:
            self.drop_label.config(text="拖拽功能不可用，请安装 tkinterdnd2\npip install tkinterdnd2", background='#ffe0e0')

        # 开始识别按钮（初始禁用）
        self.infer_btn_start = ttk.Button(left_panel, text="开始识别", command=self._start_inference_with_crop, width=25, state=tk.DISABLED)
        self.infer_btn_start.pack(pady=5)

        # 批量识别按钮
        self.infer_btn_batch = ttk.Button(left_panel, text="批量识别", command=self.batch_infer, width=25)
        self.infer_btn_batch.pack(pady=5)

        # 结果显示浏览
        result_frame = ttk.LabelFrame(left_panel, text="结果浏览")
        result_frame.pack(fill=tk.X, pady=10)
        nav_frame = ttk.Frame(result_frame)
        nav_frame.pack(fill=tk.X, pady=2)
        ttk.Button(nav_frame, text="◀", command=self.prev_result, width=3).pack(side=tk.LEFT)
        ttk.Button(nav_frame, text="▶", command=self.next_result, width=3).pack(side=tk.LEFT)
        self.result_idx_label = ttk.Label(nav_frame, text="0/0")
        self.result_idx_label.pack(side=tk.LEFT, padx=5)
        ttk.Button(nav_frame, text="刷新", command=self.update_result_images, width=6).pack(side=tk.LEFT, padx=5)
        self.result_name_label = ttk.Label(result_frame, text="当前: 无结果", anchor=tk.W)
        self.result_name_label.pack(fill=tk.X, pady=2)

        # 打开输出文件夹
        ttk.Button(left_panel, text="输出文件夹", command=self.open_output_dir, width=25).pack(pady=5)

        # 配置信息
        config_frame = ttk.LabelFrame(left_panel, text="当前配置")
        config_frame.pack(fill=tk.X, pady=10)
        ttk.Label(config_frame, text=f"IMAGE_SIZE: {IMAGE_SIZE}").pack(anchor=tk.W)
        ttk.Label(config_frame, text=f"CLASS_NAMES: {', '.join(CLASS_NAMES)}").pack(anchor=tk.W)
        ttk.Label(config_frame, text=f"CONF_THRESH: {getattr(sys.modules['script.config'], 'CONF_THRESH', 0.6)}").pack(anchor=tk.W)

        # 右侧显示与日志
        right_panel = ttk.Frame(frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 画布显示识别结果 / 裁剪预览
        self.infer_canvas = tk.Canvas(right_panel, bg='white')
        self.infer_canvas.pack(fill=tk.BOTH, expand=True, pady=(0,5))
        # 绑定裁剪交互事件（仅在预览模式生效）
        self.infer_canvas.bind("<ButtonPress-1>", self._on_crop_press)
        self.infer_canvas.bind("<B1-Motion>", self._on_crop_drag)
        self.infer_canvas.bind("<ButtonRelease-1>", self._on_crop_release)
        self.infer_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.infer_canvas.bind("<Button-4>", self._on_mousewheel)   # Linux
        self.infer_canvas.bind("<Button-5>", self._on_mousewheel)   # Linux

        # 识别日志
        ttk.Label(right_panel, text="识别日志", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.infer_log_text = scrolledtext.ScrolledText(right_panel, wrap=tk.WORD, font=('Consolas', 9), height=8)
        self.infer_log_text.pack(fill=tk.BOTH, expand=True)

        # 初始化结果列表
        self.result_images = []
        self.result_index = -1
        self.tk_result_img = None
        self.update_result_images()

    # -------------------- 数据集编辑标签页 --------------------
    def build_edit_tab(self):
        """构建数据集编辑标签页"""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="数据集编辑")

        # 左侧为分类数据集，右侧为检测数据集（使用 PanedWindow）
        paned = ttk.PanedWindow(frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # -------- 左半部分：分类数据集 --------
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)

        ttk.Label(left_frame, text="分类数据集 (classification)", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)

        # 类别树（显示文件夹）
        cls_tree_frame = ttk.Frame(left_frame)
        cls_tree_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.cls_tree = ttk.Treeview(cls_tree_frame, columns=('count',), show='tree headings', height=15)
        self.cls_tree.heading('#0', text='类别')
        self.cls_tree.heading('count', text='图片数')
        self.cls_tree.column('#0', width=150)
        self.cls_tree.column('count', width=60, anchor=tk.CENTER)
        cls_scroll = ttk.Scrollbar(cls_tree_frame, orient=tk.VERTICAL, command=self.cls_tree.yview)
        self.cls_tree.configure(yscrollcommand=cls_scroll.set)
        self.cls_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cls_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.cls_tree.bind('<<TreeviewSelect>>', self.on_cls_tree_select)

        # 分类图片列表
        cls_img_frame = ttk.Frame(left_frame)
        cls_img_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Label(cls_img_frame, text="图片列表").pack(anchor=tk.W)
        self.cls_img_listbox = tk.Listbox(cls_img_frame, height=6)  # 减少高度留出预览空间
        self.cls_img_listbox.pack(fill=tk.BOTH, expand=True)
        self.cls_img_listbox.bind('<<ListboxSelect>>', self.on_cls_img_select)

        # 分类图片预览
        cls_preview_frame = ttk.Frame(left_frame)
        cls_preview_frame.pack(fill=tk.BOTH, expand=False, pady=2)
        ttk.Label(cls_preview_frame, text="预览").pack(anchor=tk.W)
        self.cls_preview_canvas = tk.Canvas(cls_preview_frame, bg='white', width=200, height=200)
        self.cls_preview_canvas.pack(fill=tk.BOTH, expand=True)
        # 操作按钮
        cls_btn_frame = ttk.Frame(left_frame)
        cls_btn_frame.pack(fill=tk.X, pady=5)
        ttk.Button(cls_btn_frame, text="删除选中图片", command=self.delete_cls_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(cls_btn_frame, text="添加图片", command=self.add_cls_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(cls_btn_frame, text="刷新", command=self.refresh_cls_tree).pack(side=tk.LEFT, padx=2)

        # -------- 右半部分：检测数据集 --------
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)

        ttk.Label(right_frame, text="检测数据集 (detection)", font=('Arial', 12, 'bold')).pack(anchor=tk.W, pady=5)

        # 上部：图片列表
        det_list_frame = ttk.Frame(right_frame)
        det_list_frame.pack(fill=tk.X, pady=5)
        ttk.Label(det_list_frame, text="图片列表").pack(side=tk.LEFT)
        self.det_img_listbox = tk.Listbox(det_list_frame, height=8)
        self.det_img_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        det_scroll = ttk.Scrollbar(det_list_frame, orient=tk.VERTICAL, command=self.det_img_listbox.yview)
        self.det_img_listbox.configure(yscrollcommand=det_scroll.set)
        det_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.det_img_listbox.bind('<<ListboxSelect>>', self.on_det_img_select)

        # 检测图片操作按钮
        det_btn_frame = ttk.Frame(right_frame)
        det_btn_frame.pack(fill=tk.X, pady=2)
        ttk.Button(det_btn_frame, text="删除选中图片", command=self.delete_det_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(det_btn_frame, text="添加图片", command=self.add_det_image).pack(side=tk.LEFT, padx=2)
        ttk.Button(det_btn_frame, text="刷新列表", command=self.refresh_det_list).pack(side=tk.LEFT, padx=2)

        # 下部：图片预览 + 标注信息
        preview_frame = ttk.Frame(right_frame)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.det_canvas = tk.Canvas(preview_frame, bg='white', width=400, height=400)
        self.det_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # 标注信息表格
        ann_info_frame = ttk.Frame(preview_frame)
        ann_info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)
        ttk.Label(ann_info_frame, text="标注信息", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.ann_tree = ttk.Treeview(ann_info_frame, columns=('类别', 'x', 'y', 'w', 'h'), show='headings', height=10)
        self.ann_tree.heading('类别', text='类别')
        self.ann_tree.heading('x', text='x')
        self.ann_tree.heading('y', text='y')
        self.ann_tree.heading('w', text='w')
        self.ann_tree.heading('h', text='h')
        self.ann_tree.column('类别', width=80)
        self.ann_tree.column('x', width=60)
        self.ann_tree.column('y', width=60)
        self.ann_tree.column('w', width=60)
        self.ann_tree.column('h', width=60)
        self.ann_tree.pack(fill=tk.BOTH, expand=True)
        # 删除标注按钮
        ttk.Button(ann_info_frame, text="删除选中标注", command=self.delete_annotation).pack(pady=2)

        # 当前选中的检测图片路径
        self.current_det_img_path = None
        self.current_det_img_data = None  # 存放 PIL 图像
        self.current_det_img_id = None    # annotations.json 中的 image_id

        # 初始化加载数据
        self.refresh_cls_tree()
        self.refresh_det_list()

    # ---------- 分类数据集操作 ----------
    def refresh_cls_tree(self):
        """刷新分类树"""
        for item in self.cls_tree.get_children():
            self.cls_tree.delete(item)
        cls_root = DATASET_DIR / DATA_CLASSIFICATION
        if not cls_root.exists():
            return
        for cls_dir in sorted(cls_root.iterdir()):
            if cls_dir.is_dir():
                count = len(list(cls_dir.glob('*.png'))) + len(list(cls_dir.glob('*.jpg'))) + len(list(cls_dir.glob('*.jpeg')))
                self.cls_tree.insert('', tk.END, text=cls_dir.name, values=(count,))

    def on_cls_tree_select(self, event):
        """点击类别时显示该类别的图片列表"""
        sel = self.cls_tree.selection()
        if not sel:
            return
        cls_name = self.cls_tree.item(sel[0], 'text')
        cls_dir = DATASET_DIR / DATA_CLASSIFICATION / cls_name
        self.cls_img_listbox.delete(0, tk.END)
        if not cls_dir.exists():
            return
        for img_path in sorted(cls_dir.glob('*')):
            if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                self.cls_img_listbox.insert(tk.END, img_path.name)

    def delete_cls_image(self):
        """删除选中的分类图片"""
        sel = self.cls_img_listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "请先选择一张图片")
            return
        img_name = self.cls_img_listbox.get(sel[0])
        # 获取当前选中的类别
        tree_sel = self.cls_tree.selection()
        if not tree_sel:
            return
        cls_name = self.cls_tree.item(tree_sel[0], 'text')
        img_path = DATASET_DIR / DATA_CLASSIFICATION / cls_name / img_name
        if not img_path.exists():
            messagebox.showerror("错误", "文件不存在")
            return
        if messagebox.askyesno("确认删除", f"确定要删除分类图片 {img_name} 吗？\n此操作不可恢复！", icon='warning'):
            try:
                img_path.unlink()
                # 不自动刷新，不弹成功提示，用户手动点刷新
            except Exception as e:
                messagebox.showerror("错误", f"删除失败：{e}")

    def on_cls_img_select(self, event):
        """分类图片列表选择事件，显示预览"""
        sel = self.cls_img_listbox.curselection()
        if not sel:
            self.cls_preview_canvas.delete("all")
            return
        img_name = self.cls_img_listbox.get(sel[0])
        # 获取当前选中的类别
        tree_sel = self.cls_tree.selection()
        if not tree_sel:
            return
        cls_name = self.cls_tree.item(tree_sel[0], 'text')
        img_path = DATASET_DIR / DATA_CLASSIFICATION / cls_name / img_name
        if not img_path.exists():
            self.cls_preview_canvas.delete("all")
            return
        try:
            img = Image.open(img_path).convert('RGB')
            # 缩放以适应画布
            canvas_w = self.cls_preview_canvas.winfo_width() if self.cls_preview_canvas.winfo_width() > 20 else 200
            canvas_h = self.cls_preview_canvas.winfo_height() if self.cls_preview_canvas.winfo_height() > 20 else 200
            scale = min(canvas_w / img.width, canvas_h / img.height, 1.0)
            new_size = (int(img.width * scale), int(img.height * scale))
            img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
            self.cls_preview_img = ImageTk.PhotoImage(img_resized)
            self.cls_preview_canvas.delete("all")
            self.cls_preview_canvas.config(scrollregion=(0,0,new_size[0], new_size[1]))
            self.cls_preview_canvas.create_image(0,0, anchor=tk.NW, image=self.cls_preview_img)
        except Exception as e:
            self.cls_preview_canvas.delete("all")
            self.cls_preview_canvas.create_text(10,10, text=f"加载失败: {e}", anchor=tk.NW)

    def add_cls_image(self):
        """向当前选中的类别添加图片（复制）"""
        tree_sel = self.cls_tree.selection()
        if not tree_sel:
            messagebox.showwarning("提示", "请先在左侧选择一个类别")
            return
        cls_name = self.cls_tree.item(tree_sel[0], 'text')
        file_path = filedialog.askopenfilename(
            title="选择要添加的图片",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return
        src = Path(file_path)
        dst_dir = DATASET_DIR / DATA_CLASSIFICATION / cls_name
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / src.name
        # 如果同名则询问是否覆盖
        if dst.exists():
            if not messagebox.askyesno("文件已存在", f"{dst.name} 已存在，是否覆盖？"):
                return
        try:
            import shutil
            shutil.copy2(src, dst)
            self.on_cls_tree_select(None)
            self.refresh_cls_tree()
            messagebox.showinfo("成功", f"已添加图片 {dst.name}")
        except Exception as e:
            messagebox.showerror("错误", f"复制失败：{e}")

    # ---------- 检测数据集操作 ----------
    def refresh_det_list(self):
        """刷新检测图片列表"""
        self.det_img_listbox.delete(0, tk.END)
        det_img_dir = DATASET_DIR / DATA_DETECTION / "images"
        if not det_img_dir.exists():
            return
        for img_path in sorted(det_img_dir.glob('*')):
            if img_path.suffix.lower() in ('.png', '.jpg', '.jpeg'):
                self.det_img_listbox.insert(tk.END, img_path.name)

    def on_det_img_select(self, event):
        """选择检测图片后显示预览和标注"""
        sel = self.det_img_listbox.curselection()
        if not sel:
            self.det_canvas.delete("all")
            self.current_det_img_path = None
            self.current_det_img_data = None
            self.ann_tree.delete(*self.ann_tree.get_children())
            return
        img_name = self.det_img_listbox.get(sel[0])
        img_path = DATASET_DIR / DATA_DETECTION / "images" / img_name
        if not img_path.exists():
            return
        self.current_det_img_path = img_path
        # 加载图片并显示
        try:
            img = Image.open(img_path).convert('RGB')
            self.current_det_img_data = img
            self.show_det_image_with_ann()
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片：{e}")

    def show_det_image_with_ann(self):
        """在画布上显示图片并绘制标注框"""
        if self.current_det_img_data is None:
            return
        img = self.current_det_img_data
        # 缩放以适应画布
        canvas_w = self.det_canvas.winfo_width() if self.det_canvas.winfo_width() > 50 else 400
        canvas_h = self.det_canvas.winfo_height() if self.det_canvas.winfo_height() > 50 else 400
        scale_w = canvas_w / img.width
        scale_h = canvas_h / img.height
        scale = min(scale_w, scale_h, 1.0)
        new_size = (int(img.width * scale), int(img.height * scale))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        self.det_tk_img = ImageTk.PhotoImage(img_resized)
        self.det_canvas.delete("all")
        self.det_canvas.config(scrollregion=(0,0,new_size[0], new_size[1]))
        self.det_canvas.create_image(0,0, anchor=tk.NW, image=self.det_tk_img)

        # 加载标注信息
        self.ann_tree.delete(*self.ann_tree.get_children())
        if self.current_det_img_path is None:
            return
        # 从 annotations.json 中查找该图片的标注
        ann_file = DATASET_DIR / DATA_DETECTION / "annotations.json"
        if not ann_file.exists():
            return
        with open(ann_file, 'r') as f:
            data = json.load(f)
        # 找到该图片的 image_id
        img_name = self.current_det_img_path.name
        img_id = None
        for img_rec in data.get('images', []):
            if img_rec['file_name'] == img_name:
                img_id = img_rec['id']
                break
        if img_id is None:
            return
        self.current_det_img_id = img_id
        # 收集该图片的所有标注
        anns = [ann for ann in data.get('annotations', []) if ann['image_id'] == img_id]
        # 绘制框并填充表格
        for ann in anns:
            bbox = ann['bbox']  # [x, y, w, h]
            cat_id = ann['category_id']
            cat_name = None
            for cat in data.get('categories', []):
                if cat['id'] == cat_id:
                    cat_name = cat['name']
                    break
            # 画框（缩放坐标）
            x1 = bbox[0] * scale
            y1 = bbox[1] * scale
            x2 = (bbox[0] + bbox[2]) * scale
            y2 = (bbox[1] + bbox[3]) * scale
            self.det_canvas.create_rectangle(x1, y1, x2, y2, outline='red', width=2)
            self.det_canvas.create_text(x1, y1-5, text=cat_name or str(cat_id), fill='red', anchor=tk.SW)
            # 添加到表格
            self.ann_tree.insert('', tk.END, values=(cat_name or str(cat_id),
                                                     f"{bbox[0]:.1f}", f"{bbox[1]:.1f}",
                                                     f"{bbox[2]:.1f}", f"{bbox[3]:.1f}"))

    def delete_det_image(self):
        """删除选中的检测图片及其标注"""
        sel = self.det_img_listbox.curselection()
        if not sel:
            messagebox.showwarning("提示", "请先选择一张图片")
            return
        img_name = self.det_img_listbox.get(sel[0])
        img_path = DATASET_DIR / DATA_DETECTION / "images" / img_name
        if not img_path.exists():
            messagebox.showerror("错误", "文件不存在")
            return
        if not messagebox.askyesno("确认删除", f"确定要删除检测图片 {img_name} 及其所有标注吗？\n此操作不可恢复！", icon='warning'):
            return
        # 删除图片文件
        try:
            img_path.unlink()
        except Exception as e:
            messagebox.showerror("错误", f"删除图片失败：{e}")
            return
        # 更新 annotations.json
        ann_file = DATASET_DIR / DATA_DETECTION / "annotations.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                data = json.load(f)
            # 找到 image_id
            img_id = None
            for img_rec in data.get('images', []):
                if img_rec['file_name'] == img_name:
                    img_id = img_rec['id']
                    break
            if img_id is not None:
                # 移除该图片记录
                data['images'] = [img_rec for img_rec in data.get('images', []) if img_rec['id'] != img_id]
                # 移除该图片的所有标注
                data['annotations'] = [ann for ann in data.get('annotations', []) if ann['image_id'] != img_id]
                # 写回
                with open(ann_file, 'w') as f:
                    json.dump(data, f, indent=2)
        # 不自动刷新列表，不清空预览，不弹成功提示，用户手动点刷新

    def add_det_image(self):
        """向检测数据集添加图片（复制到 detection/images，并添加 annotations.json 记录）"""
        file_path = filedialog.askopenfilename(
            title="选择要添加的图片",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return
        src = Path(file_path)
        det_img_dir = DATASET_DIR / DATA_DETECTION / "images"
        det_img_dir.mkdir(parents=True, exist_ok=True)
        dst = det_img_dir / src.name
        if dst.exists():
            if not messagebox.askyesno("文件已存在", f"{dst.name} 已存在，是否覆盖？"):
                return
        try:
            import shutil
            shutil.copy2(src, dst)
        except Exception as e:
            messagebox.showerror("错误", f"复制失败：{e}")
            return
        # 更新 annotations.json：添加图片记录（无标注）
        ann_file = DATASET_DIR / DATA_DETECTION / "annotations.json"
        if ann_file.exists():
            with open(ann_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"images": [], "annotations": [], "categories": []}
            # 若 categories 为空，从 CLASS_NAMES 重建（但最好保留原有）
            if not data.get('categories'):
                data['categories'] = [{"id": i+1, "name": name} for i, name in enumerate(CLASS_NAMES)]
        # 检查图片是否已存在记录（按文件名）
        exist = any(img['file_name'] == dst.name for img in data.get('images', []))
        if not exist:
            new_id = max([img['id'] for img in data.get('images', [])] + [0]) + 1
            data['images'].append({
                "id": new_id,
                "file_name": dst.name,
                "width": IMAGE_SIZE,
                "height": IMAGE_SIZE
            })
            with open(ann_file, 'w') as f:
                json.dump(data, f, indent=2)
        self.refresh_det_list()
        messagebox.showinfo("成功", f"已添加图片 {dst.name}")

    def delete_annotation(self):
        """删除选中的标注"""
        if self.current_det_img_id is None:
            messagebox.showwarning("提示", "请先选择一张检测图片")
            return
        sel = self.ann_tree.selection()
        if not sel:
            messagebox.showwarning("提示", "请先选择一条标注")
            return
        # 获取选中行的值（用于定位）
        item = sel[0]
        values = self.ann_tree.item(item, 'values')
        if not values:
            return
        # 由于表格中未存储唯一 ID，我们用 bbox 近似匹配（实际应该存储 ann id，但简化处理）
        # 更好的方法是存储 ann id，但为简化，我们根据坐标匹配（可能存在精度问题）
        # 我们重新读取 annotations.json，找到匹配的标注（按 image_id + bbox）
        ann_file = DATASET_DIR / DATA_DETECTION / "annotations.json"
        if not ann_file.exists():
            return
        with open(ann_file, 'r') as f:
            data = json.load(f)
        # 找到该图片的所有标注
        anns = [ann for ann in data.get('annotations', []) if ann['image_id'] == self.current_det_img_id]
        if not anns:
            return
        # 匹配（通过 bbox 和 category_id）
        # 将表格中的值转为 float
        try:
            bbox = [float(values[1]), float(values[2]), float(values[3]), float(values[4])]
        except:
            messagebox.showerror("错误", "无法解析标注坐标")
            return
        # 查找匹配的标注（允许小误差）
        matched = None
        for ann in anns:
            if (abs(ann['bbox'][0] - bbox[0]) < 1e-3 and
                abs(ann['bbox'][1] - bbox[1]) < 1e-3 and
                abs(ann['bbox'][2] - bbox[2]) < 1e-3 and
                abs(ann['bbox'][3] - bbox[3]) < 1e-3):
                matched = ann
                break
        if matched is None:
            messagebox.showerror("错误", "未找到匹配的标注，请刷新后重试")
            return
        if not messagebox.askyesno("确认删除", "确定要删除该标注吗？", icon='warning'):
            return
        # 从 data 中移除该标注
        data['annotations'] = [ann for ann in data.get('annotations', []) if ann != matched]
        with open(ann_file, 'w') as f:
            json.dump(data, f, indent=2)
        # 刷新显示
        self.on_det_img_select(None)  # 重新加载
        self.show_det_image_with_ann()
        messagebox.showinfo("成功", "标注已删除")

    def on_drop(self, event):
        """处理拖拽放入的文件（显示裁剪预览）"""
        if not DND_AVAILABLE:
            return
        data = event.data
        if data.startswith('{') and data.endswith('}'):
            data = data[1:-1]
        files = data.split()
        if not files:
            return
        file_path = Path(files[0])
        img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
        if file_path.suffix.lower() not in img_exts:
            messagebox.showerror("错误", "拖入的文件不是支持的图片格式")
            return
        # 显示图片并启用裁剪
        self._show_crop_preview(file_path)

    def _show_crop_preview(self, img_path):
        """加载图片并显示裁剪预览"""
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片：{e}")
            return
        self.current_image_path = img_path
        self.orig_image = img
        self.orig_size = img.size
        # 缩放以适应画布
        canvas_w = self.infer_canvas.winfo_width() if self.infer_canvas.winfo_width() > 50 else 800
        canvas_h = self.infer_canvas.winfo_height() if self.infer_canvas.winfo_height() > 50 else 600
        scale_w = canvas_w / img.width
        scale_h = canvas_h / img.height
        self.scale = min(scale_w, scale_h, 1.0)
        new_size = (int(img.width * self.scale), int(img.height * self.scale))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        self.infer_tk_img = ImageTk.PhotoImage(img_resized)
        self.infer_canvas.delete("all")
        self.infer_canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))
        self.infer_canvas.create_image(0, 0, anchor=tk.NW, image=self.infer_tk_img, tags="bg_img")
        # 初始化裁剪框
        self._init_crop_box()
        self._update_crop_box()
        # 更新模式
        self.infer_mode = 'preview'
        self.infer_btn_start.config(state=tk.NORMAL)
        self.result_name_label.config(text=f"预览: {img_path.name}")

    def _init_crop_box(self):
        """根据 IMAGE_SIZE 和当前图片尺寸初始化裁剪框（居中，正方形）"""
        if self.orig_size[0] == 0 or self.orig_size[1] == 0:
            return
        target_size = IMAGE_SIZE
        img_w, img_h = self.orig_size
        self.crop_size = min(img_w, img_h, target_size)
        cx = img_w // 2
        cy = img_h // 2
        half = self.crop_size // 2
        self.crop_box = (cx - half, cy - half, cx + half, cy + half)
        self._clamp_crop_box()

    def _clamp_crop_box(self):
        """确保裁剪框不超出图片边界"""
        x1, y1, x2, y2 = self.crop_box
        img_w, img_h = self.orig_size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w, x2)
        y2 = min(img_h, y2)
        if x2 <= x1 or y2 <= y1:
            self._init_crop_box()
        else:
            self.crop_box = (x1, y1, x2, y2)

    def _update_crop_box(self):
        """根据当前 self.crop_box 更新画布上的裁剪框和遮罩"""
        self.infer_canvas.delete("crop_rect")
        self.infer_canvas.delete("crop_overlay")
        if self.infer_tk_img is None:
            return
        x1, y1, x2, y2 = self.crop_box
        # 转换为画布坐标
        cx1 = x1 * self.scale
        cy1 = y1 * self.scale
        cx2 = x2 * self.scale
        cy2 = y2 * self.scale
        # 绘制裁剪框（黄色边框）
        self.infer_canvas.create_rectangle(
            cx1, cy1, cx2, cy2,
            outline='yellow', width=2, tags="crop_rect"
        )
        # 透明内部区域（扩大点击区域）
        self.infer_canvas.create_rectangle(
            cx1+2, cy1+2, cx2-2, cy2-2,
            fill='', outline='', tags="crop_rect"
        )
        # 绘制半透明遮罩（四个矩形）
        img_w = self.orig_size[0] * self.scale
        img_h = self.orig_size[1] * self.scale
        self.infer_canvas.create_rectangle(0, 0, img_w, cy1, fill='black', stipple='gray50', outline='', tags="crop_overlay")
        self.infer_canvas.create_rectangle(0, cy2, img_w, img_h, fill='black', stipple='gray50', outline='', tags="crop_overlay")
        self.infer_canvas.create_rectangle(0, cy1, cx1, cy2, fill='black', stipple='gray50', outline='', tags="crop_overlay")
        self.infer_canvas.create_rectangle(cx2, cy1, img_w, cy2, fill='black', stipple='gray50', outline='', tags="crop_overlay")

    def _on_crop_press(self, event):
        if self.infer_mode != 'preview':
            return
        self.is_dragging = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.drag_orig_box = self.crop_box

    def _on_crop_drag(self, event):
        if not self.is_dragging or self.infer_mode != 'preview':
            return
        dx_total = (event.x - self.drag_start_x) / self.scale
        dy_total = (event.y - self.drag_start_y) / self.scale
        x1, y1, x2, y2 = self.drag_orig_box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        new_cx = cx + dx_total
        new_cy = cy + dy_total
        half = self.crop_size // 2
        new_box = (new_cx - half, new_cy - half, new_cx + half, new_cy + half)
        img_w, img_h = self.orig_size
        x1_new = max(0, new_box[0])
        y1_new = max(0, new_box[1])
        x2_new = min(img_w, new_box[2])
        y2_new = min(img_h, new_box[3])
        if x2_new - x1_new < self.crop_size:
            if x1_new == 0:
                x2_new = self.crop_size
            elif x2_new == img_w:
                x1_new = img_w - self.crop_size
        if y2_new - y1_new < self.crop_size:
            if y1_new == 0:
                y2_new = self.crop_size
            elif y2_new == img_h:
                y1_new = img_h - self.crop_size
        x1_new = int(max(0, x1_new))
        y1_new = int(max(0, y1_new))
        x2_new = int(min(img_w, x2_new))
        y2_new = int(min(img_h, y2_new))
        self.crop_box = (x1_new, y1_new, x2_new, y2_new)
        self._update_crop_box()

    def _on_crop_release(self, event):
        self.is_dragging = False

    def _on_mousewheel(self, event):
        """滚轮缩放裁剪框（保持正方形）"""
        if self.infer_mode != 'preview':
            return
        if event.num == 4 or event.delta > 0:
            factor = 1.1
        elif event.num == 5 or event.delta < 0:
            factor = 0.9
        else:
            return
        x1, y1, x2, y2 = self.crop_box
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        new_size = self.crop_size * factor
        min_size = 20
        max_size = min(self.orig_size) * 0.9
        if new_size < min_size or new_size > max_size:
            return
        self.crop_size = new_size
        half = new_size // 2
        new_box = (cx - half, cy - half, cx + half, cy + half)
        img_w, img_h = self.orig_size
        x1_new = max(0, new_box[0])
        y1_new = max(0, new_box[1])
        x2_new = min(img_w, new_box[2])
        y2_new = min(img_h, new_box[3])
        if x2_new - x1_new < new_size:
            if x1_new == 0:
                x2_new = new_size
            elif x2_new == img_w:
                x1_new = img_w - new_size
        if y2_new - y1_new < new_size:
            if y1_new == 0:
                y2_new = new_size
            elif y2_new == img_h:
                y1_new = img_h - new_size
        x1_new = int(max(0, x1_new))
        y1_new = int(max(0, y1_new))
        x2_new = int(min(img_w, x2_new))
        y2_new = int(min(img_h, y2_new))
        self.crop_box = (x1_new, y1_new, x2_new, y2_new)
        self._update_crop_box()

    def _start_inference_with_crop(self):
        """裁剪当前预览图片并执行识别"""
        if self.infer_mode != 'preview' or self.orig_image is None:
            return
        if self.crop_box == (0, 0, 0, 0):
            messagebox.showerror("错误", "裁剪框无效")
            return
        # 裁剪并缩放到 IMAGE_SIZE
        crop_box = self.crop_box
        img = self.orig_image
        cropped = img.crop(crop_box)
        cropped = cropped.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
        # 保存到临时目录（避免被 run_inference 清空 input 时误删）
        temp_dir = BASE_DIR / "temp"
        temp_dir.mkdir(exist_ok=True)
        crop_path = temp_dir / "crop_temp.png"
        cropped.save(crop_path)
        # 调用识别
        self.infer_btn_start.config(state=tk.DISABLED)
        self.run_inference([crop_path])

    def update_result_images(self):
        """扫描 output 目录，更新结果图片列表并显示第一张"""
        output_dir = BASE_DIR / "output"
        if not output_dir.exists():
            self.result_images = []
            self.result_index = -1
            self.result_idx_label.config(text="0/0")
            self.result_name_label.config(text="当前: 无结果")
            self.infer_canvas.delete("all")
            self.tk_result_img = None
            return

        # 收集所有 *_result.png，按修改时间排序（最新的在前）
        result_files = sorted(output_dir.glob("*_result.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        self.result_images = result_files
        if result_files:
            self.result_index = 0
            self.show_result_image(0)
        else:
            self.result_index = -1
            self.result_idx_label.config(text="0/0")
            self.result_name_label.config(text="当前: 无结果")
            self.infer_canvas.delete("all")
            self.tk_result_img = None

    def show_result_image(self, index):
        """显示指定索引的结果图片"""
        if not self.result_images or index < 0 or index >= len(self.result_images):
            self.infer_canvas.delete("all")
            self.tk_result_img = None
            self.result_idx_label.config(text=f"{max(0,index+1)}/{len(self.result_images)}")
            self.result_name_label.config(text="当前: 无结果")
            return

        img_path = self.result_images[index]
        self.result_index = index
        self.result_idx_label.config(text=f"{index+1}/{len(self.result_images)}")
        self.result_name_label.config(text=f"当前: {img_path.name}")

        # 加载并缩放显示
        img = Image.open(img_path).convert('RGB')
        # 计算缩放比例适应画布
        canvas_w = self.infer_canvas.winfo_width() if self.infer_canvas.winfo_width() > 50 else 800
        canvas_h = self.infer_canvas.winfo_height() if self.infer_canvas.winfo_height() > 50 else 600
        scale_w = canvas_w / img.width
        scale_h = canvas_h / img.height
        scale = min(scale_w, scale_h, 1.0)
        new_size = (int(img.width * scale), int(img.height * scale))
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        self.tk_result_img = ImageTk.PhotoImage(img_resized)
        self.infer_canvas.delete("all")
        self.infer_canvas.config(scrollregion=(0,0,new_size[0], new_size[1]))
        self.infer_canvas.create_image(0,0, anchor=tk.NW, image=self.tk_result_img)

    def prev_result(self):
        if self.result_index > 0:
            self.show_result_image(self.result_index - 1)

    def next_result(self):
        if self.result_index < len(self.result_images) - 1:
            self.show_result_image(self.result_index + 1)

    def open_output_dir(self):
        output_dir = BASE_DIR / "output"
        if output_dir.exists():
            # 跨平台打开文件夹
            if sys.platform == 'win32':
                os.startfile(output_dir)
            elif sys.platform == 'darwin':
                subprocess.run(['open', output_dir])
            else:
                subprocess.run(['xdg-open', output_dir])
        else:
            messagebox.showinfo("提示", "输出目录不存在，将在识别时自动创建。")

    def batch_infer(self):
        folder = filedialog.askdirectory(title="选择包含图片的文件夹")
        if not folder:
            return
        # 收集所有图片
        img_paths = []
        for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif'):
            img_paths.extend(Path(folder).glob(ext))
            img_paths.extend(Path(folder).glob(ext.upper()))
        if not img_paths:
            messagebox.showerror("错误", "未找到图片")
            return
        self.run_inference(img_paths)

    def run_inference(self, img_paths):
        """运行识别，img_paths 为 Path 列表"""
        if not img_paths:
            return

        # 禁用按钮
        self.infer_btn_start.config(state=tk.DISABLED)
        self.infer_btn_batch.config(state=tk.DISABLED)

        self.infer_log_text.insert(tk.END, f">>> 开始识别，共 {len(img_paths)} 张图片...\n")
        self.infer_log_text.see(tk.END)

        # 将图片复制到 input 目录
        input_dir = BASE_DIR / "input"
        input_dir.mkdir(exist_ok=True)
        # 清空旧文件
        for f in input_dir.glob("*"):
            if f.is_file():
                f.unlink()
        import shutil
        for p in img_paths:
            shutil.copy2(p, input_dir / p.name)

        def run():
            script_path = BASE_DIR / "script" / "batch_inference.py"
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            proc = subprocess.Popen(
                [sys.executable, str(script_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                cwd=str(BASE_DIR / "script")
            )
            for line in proc.stdout:
                self.infer_log_text.insert(tk.END, line)
                self.infer_log_text.see(tk.END)
            proc.stdout.close()
            return_code = proc.wait()
            self.root.after(0, lambda: self.inference_finished(return_code))

        threading.Thread(target=run, daemon=True).start()

    def inference_finished(self, code):
        """识别完成回调"""
        self.infer_btn_start.config(state=tk.NORMAL)
        self.infer_btn_batch.config(state=tk.NORMAL)
        # 识别完成后清除预览模式
        self.infer_mode = 'result'
        if code == 0:
            self.infer_log_text.insert(tk.END, ">>> 识别完成！\n")
            self.update_result_images()   # 刷新结果列表并显示
        else:
            self.infer_log_text.insert(tk.END, f">>> 识别失败，返回码 {code}\n")
        self.infer_log_text.see(tk.END)


# ======================== 主程序 ========================
if __name__ == "__main__":
    if DND_AVAILABLE:
        root = tkinterdnd2.Tk()
    else:
        root = tk.Tk()
    app = MainApp(root)
    root.mainloop()