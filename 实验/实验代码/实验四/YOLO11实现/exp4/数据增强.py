# -*- coding: utf-8 -*-
"""
功能：对 YOLO 格式数据集做离线数据增强（图片 + 同名 txt 标签）
输入：
  dataset/images/train/*.jpg|png...
  dataset/labels/train/*.txt   （YOLO标签：class x_center y_center w h，均为0~1归一化）
输出：
  dataset_aug/images/train/...
  dataset_aug/labels/train/...

说明：
- 使用 Albumentations 做增强，会自动同步变换边界框（bbox）。
- 默认会把原图也复制到输出目录（copy_original=True），并额外为每张图生成 n_aug_per_image 张增强图。
"""

import os
import glob
import random
from pathlib import Path

import cv2
import albumentations as A


def read_yolo_labels(label_path: str):
    """
    读取 YOLO 标签文件
    每行格式：class x_center y_center w h（全部为归一化比例）
    返回：
      bboxes: [[x_center, y_center, w, h], ...]
      class_ids: [class_id, ...]
    """
    bboxes, class_ids = [], []
    if not os.path.exists(label_path):
        # 没有标签文件就当作空目标（有些数据集可能存在负样本）
        return bboxes, class_ids

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            # 正常 YOLO 标签应该是5列
            if len(parts) != 5:
                continue

            cls = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            bboxes.append([x, y, w, h])
            class_ids.append(cls)

    return bboxes, class_ids


def write_yolo_labels(label_path: str, bboxes, class_ids):
    """
    写入 YOLO 标签文件
    每行格式：class x_center y_center w h（归一化）
    """
    with open(label_path, "w", encoding="utf-8") as f:
        for cls, (x, y, w, h) in zip(class_ids, bboxes):
            f.write(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


def build_augment_pipeline(img_size=640):
    """
    构建数据增强流水线（Albumentations）
    注意：
    - format='yolo' 表示 bbox 使用 [x_center, y_center, w, h] 且归一化到 0~1
    - label_fields 用于保持 bbox 与 class_ids 一一对应
    """
    return A.Compose(
        [
            # ========== 几何增强 ==========
            A.HorizontalFlip(p=0.5),  # 随机水平翻转

            # 平移/缩放/旋转（不会改变图片大小，空白区域用黑色填充）
            A.ShiftScaleRotate(
                shift_limit=0.05,      # 平移范围（占宽/高比例）
                scale_limit=0.2,       # 缩放范围
                rotate_limit=10,       # 旋转角度范围（±10度）
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.7
            ),

            # 随机裁剪再缩放到固定大小（模拟目标远近变化）
            A.RandomResizedCrop(
                size=(img_size, img_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                p=0.5
            ),

            # 确保最终输入尺寸一致（YOLO常用 640x640）
            A.Resize(img_size, img_size, p=1.0),

            # ========== 颜色/光照增强 ==========
            A.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7
            ),
            A.RandomBrightnessContrast(p=0.3),

            # ========== 模糊/噪声增强（提高鲁棒性）==========
            A.GaussianBlur(blur_limit=(3, 5), p=0.15),
            A.MotionBlur(blur_limit=5, p=0.15),
            A.GaussNoise(var_limit=(10.0, 40.0), p=0.2),
        ],
        bbox_params=A.BboxParams(
            format="yolo",
            label_fields=["class_ids"],
            min_visibility=0.2,  # 如果增强后目标可见面积太小，就丢弃该 bbox
            clip=True            # 将 bbox 裁剪到图像范围内
        ),
    )


def ensure_dir(p: str):
    """创建目录（若不存在则创建）"""
    Path(p).mkdir(parents=True, exist_ok=True)


def main(
    images_dir="dataset/images/train",
    labels_dir="dataset/labels/train",
    out_images_dir="dataset_aug/images/train",
    out_labels_dir="dataset_aug/labels/train",
    n_aug_per_image=2,
    img_size=640,
    copy_original=True,
    seed=42
):
    """
    参数说明：
    - images_dir/labels_dir：原始训练集图片与标签目录
    - out_images_dir/out_labels_dir：增强后输出目录
    - n_aug_per_image：每张原图生成多少张增强图
    - img_size：输出图片统一尺寸（通常 640）
    - copy_original：是否把原始数据也复制到输出目录（建议 True，方便直接训练）
    """
    random.seed(seed)

    ensure_dir(out_images_dir)
    ensure_dir(out_labels_dir)

    # 支持多种图片后缀
    img_paths = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        img_paths.extend(glob.glob(os.path.join(images_dir, ext)))

    if not img_paths:
        raise RuntimeError(f"在目录中未找到图片：{images_dir}")

    # 构建增强流水线
    aug = build_augment_pipeline(img_size=img_size)

    # 1）可选：先把原始数据复制到输出目录
    if copy_original:
        for img_path in img_paths:
            base = Path(img_path).stem
            label_path = os.path.join(labels_dir, base + ".txt")

            img = cv2.imread(img_path)
            if img is None:
                continue

            # 保存原图
            out_img_path = os.path.join(out_images_dir, Path(img_path).name)
            cv2.imwrite(out_img_path, img)

            # 保存原标签（若无标签则写空文件）
            bboxes, class_ids = read_yolo_labels(label_path)
            out_label_path = os.path.join(out_labels_dir, base + ".txt")
            write_yolo_labels(out_label_path, bboxes, class_ids)

    # 2）生成增强样本
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        base = Path(img_path).stem
        label_path = os.path.join(labels_dir, base + ".txt")
        bboxes, class_ids = read_yolo_labels(label_path)

        for k in range(n_aug_per_image):
            # Albumentations 会自动同步变换 bbox
            transformed = aug(image=img, bboxes=bboxes, class_ids=class_ids)

            aug_img = transformed["image"]
            aug_bboxes = transformed["bboxes"]
            aug_class_ids = transformed["class_ids"]

            # 若增强后 bbox 全丢了（比如裁剪太狠），可选择跳过
            if len(aug_bboxes) == 0:
                continue

            out_name = f"{base}_aug{k}"
            out_img_path = os.path.join(out_images_dir, out_name + ".jpg")
            out_label_path = os.path.join(out_labels_dir, out_name + ".txt")

            cv2.imwrite(out_img_path, aug_img)
            write_yolo_labels(out_label_path, aug_bboxes, aug_class_ids)

    print("✅ 数据增强完成！")
    print("增强图片目录：", out_images_dir)
    print("增强标签目录：", out_labels_dir)


if __name__ == "__main__":
    # ======= 你只需要改这里的路径即可 =======
    main(
        images_dir="dataset/images/train",      # 原始训练图片目录
        labels_dir="dataset/labels/train",      # 原始训练标签目录
        out_images_dir="dataset_aug/images/train",  # 输出增强图片目录
        out_labels_dir="dataset_aug/labels/train",  # 输出增强标签目录
        n_aug_per_image=3,                      # 每张图生成3张增强图
        img_size=640,                           # 输出尺寸
        copy_original=True                      # 把原图也复制到输出
    )
