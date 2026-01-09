"""
单阶段训练脚本（纯 Python，不用命令行）：
- 直接用本地 YOLO11 权重（models/yolo11n.pt 或你自己的 yolov11.pt）
- 对你的 60 张图片（3 类：0自行车 1电动车 2汽车）进行迁移学习训练
- 如果你没有手动划分 train/val，本脚本会从 vehicle3/raw 自动拆分生成 YOLO 标准结构
"""

import random
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

# =========================
# 路径配置（按你当前目录结构）
# =========================

# 你的本地 YOLO11 权重
BASE_WEIGHTS = r"models/yolo11n.pt"   # 如果你文件叫 yolov11.pt，就改成 r"models/yolov11.pt"

# 你的数据集根目录
DATA_ROOT = r"vehicle3"

# 原始数据（未分 train/val 时使用）
RAW_IMAGES = r"vehicle3/raw/images"
RAW_LABELS = r"vehicle3/raw/labels"

# 自动拆分比例
VAL_RATIO = 0.2
SEED = 42

# 类别名（必须与标签 id：0/1/2 对齐）
NAMES = {
    0: "bicycle",
    1: "e-bike",
    2: "car",
}

# =========================
# 训练参数（60 张小数据集建议）
# =========================
EPOCHS = 300
IMGSZ = 640
BATCH = 8          # 显存不够就改小：4/2
PATIENCE = 50      # 早停

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def pick_device():
    """优先使用 GPU（如果可用），否则用 CPU"""
    return 0 if torch.cuda.is_available() else "cpu"


def ensure_yolo_split_structure(data_root: Path, raw_images: Path, raw_labels: Path, val_ratio=0.2, seed=42):
    """
    如果 data_root 下存在标准 YOLO 结构：
        images/train, images/val, labels/train, labels/val
    则直接使用；
    否则从 raw_images/raw_labels 自动拆分并拷贝生成上述结构。
    """
    images_train = data_root / "images" / "train"
    images_val = data_root / "images" / "val"
    labels_train = data_root / "labels" / "train"
    labels_val = data_root / "labels" / "val"

    if images_train.exists() and images_val.exists() and labels_train.exists() and labels_val.exists():
        print("[OK] 已检测到标准 YOLO 数据结构：", data_root)
        return

    print("[INFO] 未检测到 train/val 结构，准备从 raw/images + raw/labels 自动拆分生成...")

    raw_images = Path(raw_images)
    raw_labels = Path(raw_labels)

    if not raw_images.exists() or not raw_labels.exists():
        raise FileNotFoundError(
            "没有找到标准结构，也没找到 raw/images 或 raw/labels。\n"
            f"raw_images={raw_images}\nraw_labels={raw_labels}\n"
            "请确认路径，或手动创建 images/labels 的 train/val 结构。"
        )

    imgs = [p for p in raw_images.iterdir() if p.suffix.lower() in IMG_EXTS]
    if not imgs:
        raise RuntimeError(f"raw/images 下没有图片：{raw_images}")

    pairs = []
    missing = 0
    for img in imgs:
        lab = raw_labels / f"{img.stem}.txt"
        if lab.exists():
            pairs.append((img, lab))
        else:
            missing += 1

    if not pairs:
        raise RuntimeError("没有找到任何 (图片, 标签) 配对。请确认标签文件名与图片同名（仅后缀不同）。")

    if missing:
        print(f"[WARN] 有 {missing} 张图片缺少标签文件，将自动跳过这些图片。")

    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_val = max(1, int(n_total * val_ratio))
    val_set = pairs[:n_val]
    train_set = pairs[n_val:]

    for d in [images_train, images_val, labels_train, labels_val]:
        d.mkdir(parents=True, exist_ok=True)

    def copy_pairs(pairs_list, img_dst, lab_dst):
        for img, lab in pairs_list:
            shutil.copy2(img, img_dst / img.name)
            shutil.copy2(lab, lab_dst / lab.name)

    copy_pairs(train_set, images_train, labels_train)
    copy_pairs(val_set, images_val, labels_val)

    print(f"[OK] 拆分完成：train={len(train_set)} 张, val={len(val_set)} 张")
    print("[OK] 数据集根目录：", data_root)


def write_dataset_yaml(data_root: Path, names: dict, yaml_path: Path):
    """写 Ultralytics YOLO 所需的数据集 yaml（Windows 路径安全写法）"""
    root_posix = data_root.resolve().as_posix()

    yaml_text = (
        "path: " + root_posix + "\n"
        "train: images/train\n"
        "val: images/val\n\n"
        "names:\n"
    )
    for k in sorted(names.keys()):
        yaml_text += f"  {k}: {names[k]}\n"

    yaml_path.write_text(yaml_text, encoding="utf-8")
    print("[OK] 已写入数据集 YAML：", yaml_path)


def train_vehicle_only():
    device = pick_device()
    print(f"[INFO] device={device} (cuda_available={torch.cuda.is_available()})")

    base_weights_path = Path(BASE_WEIGHTS)
    if not base_weights_path.exists():
        raise FileNotFoundError(f"找不到模型权重文件：{base_weights_path}")

    data_root = Path(DATA_ROOT)

    # 确保数据集结构存在（否则从 raw 自动拆分）
    ensure_yolo_split_structure(
        data_root=data_root,
        raw_images=Path(RAW_IMAGES),
        raw_labels=Path(RAW_LABELS),
        val_ratio=VAL_RATIO,
        seed=SEED,
    )

    # 写 dataset yaml
    dataset_yaml = data_root / "vehicle3.yaml"
    write_dataset_yaml(data_root, NAMES, dataset_yaml)

    # 加载预训练权重（迁移学习）
    model = YOLO(str(base_weights_path))

    # 开始训练（强增强 + 冻结部分层 + 早停）
    results = model.train(
        data=str(dataset_yaml),

        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device=device,
        workers=4,
        cache=True,
        patience=PATIENCE,

        optimizer="AdamW",
        lr0=0.002,
        lrf=0.01,
        weight_decay=0.01,
        warmup_epochs=3,
        cos_lr=True,
        label_smoothing=0.05,

        # 小数据集建议冻结一部分层
        freeze=10,

        # 强增强（适合 60 张）
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.10,
        scale=0.50,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.5,
        flipud=0.0,

        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.10,
        close_mosaic=15,

        amp=True,
        pretrained=True,

        project=str(data_root / "runs_vehicle"),
        name="yolo11_vehicle3_only",
    )

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    print("[OK] 训练完成。")
    if best_pt.exists():
        print("[OK] best.pt 路径：", best_pt)
    else:
        print("[WARN] 没找到 best.pt，请检查训练输出目录：", results.save_dir)


if __name__ == "__main__":
    train_vehicle_only()
