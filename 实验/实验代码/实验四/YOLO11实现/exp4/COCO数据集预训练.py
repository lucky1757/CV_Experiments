# pretrain_on_coco.py
from ultralytics import YOLO

def main():
    # 1) 从结构yaml构建新模型：随机初始化（=你自己的“预训练”起点）
    model = YOLO("yolo11n.yaml")   # 可换 yolo11s.yaml / yolo11m.yaml 等

    # 2) 在 COCO 上训练（会使用 coco.yaml；COCO 会在首次使用时自动下载）
    results = model.train(
        data="coco.yaml",
        epochs=300,          # 预训练一般更长，可按算力调整
        imgsz=640,
        batch=16,            # 显存够就调大
        device=0,            # 多卡可写 device=[0,1]
        workers=8,
        project="runs_pretrain",
        name="coco_your_pretrain",
        save=True,
    )

    print("Done. Check weights under runs_pretrain/coco_your_pretrain/weights/")

if __name__ == "__main__":
    main()
