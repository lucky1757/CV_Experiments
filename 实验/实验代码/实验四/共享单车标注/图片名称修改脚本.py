import os
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

def rename_images_in_subfolders(base_dir: Path):
    for sub in sorted([p for p in base_dir.iterdir() if p.is_dir()]):
        imgs = sorted(
            [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: p.name.lower()
        )

        if not imgs:
            continue

        # 先改成临时名，避免 1.jpg 已存在导致冲突
        tmp_paths = []
        for i, p in enumerate(imgs, start=1):
            tmp = sub / f"__tmp__{i:06d}{p.suffix.lower()}"
            p.rename(tmp)
            tmp_paths.append(tmp)

        # 再改成最终名：1.xxx / 2.xxx ...
        for i, tmp in enumerate(tmp_paths, start=1):
            final = sub / f"{i}{tmp.suffix.lower()}"
            tmp.rename(final)

        print(f"[OK] {sub.name}: renamed {len(imgs)} images")

if __name__ == "__main__":
    rename_images_in_subfolders(Path.cwd())
