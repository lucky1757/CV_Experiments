import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO

# =========================
# 配置项（根据你的路径修改）
# =========================
BEST_MODEL_PATH = r"实验四\YOLO11实现\exp4\vehicle3\runs_vehicle\yolo11_vehicle3_only2\weights\best.pt"
DETECT_SAVE_DIR = r"vehicle3/detect_results"

# 类别配置（0=哈啰单车 1=电动车 2=汽车）
CLASS_NAMES = {
    0: "哈啰单车",
    1: "e-bike",
    2: "car"
}
CLASS_COLORS = {
    0: (0, 255, 0),    # 绿色-哈啰单车
    1: (255, 0, 0),    # 红色-电动车
    2: (0, 0, 255)     # 蓝色-汽车
}


class YOLODetectorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("YOLO11 车辆检测工具（哈啰单车/电动车/汽车）")
        self.root.geometry("1000x700")

        # 初始化变量
        self.model = None
        self.selected_img_path: Path | None = None

        # 原始图片（保持原始尺寸，不要被缩放函数修改）
        self.original_img_full: Image.Image | None = None
        # 检测后图片（原始尺寸）
        self.detected_img_full: Image.Image | None = None

        # 加载模型
        self.load_model()

        # 创建GUI组件
        self.create_widgets()

    def load_model(self):
        """加载训练好的YOLO模型"""
        try:
            model_path = Path(BEST_MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"未找到模型文件：{model_path}")

            self.model = YOLO(str(model_path))
            print(f"[成功] 加载模型：{BEST_MODEL_PATH}")
            messagebox.showinfo("提示", "模型加载成功！")
        except Exception as e:
            messagebox.showerror("错误", f"模型加载失败：{str(e)}")
            self.root.quit()

    def create_widgets(self):
        """创建GUI按钮、画布等组件"""
        # 按钮框架
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        # 选择图片按钮
        self.select_btn = tk.Button(
            btn_frame, text="选择图片", command=self.select_image,
            width=15, height=2, font=("微软雅黑", 12)
        )
        self.select_btn.grid(row=0, column=0, padx=10)

        # 检测按钮
        self.detect_btn = tk.Button(
            btn_frame, text="开始检测", command=self.detect_image,
            width=15, height=2, font=("微软雅黑", 12), state=tk.DISABLED
        )
        self.detect_btn.grid(row=0, column=1, padx=10)

        # 保存结果按钮
        self.save_btn = tk.Button(
            btn_frame, text="保存检测结果", command=self.save_detected_img,
            width=15, height=2, font=("微软雅黑", 12), state=tk.DISABLED
        )
        self.save_btn.grid(row=0, column=2, padx=10)

        # 图片显示画布
        canvas_frame = tk.Frame(self.root)
        canvas_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # 左画布：原图
        self.original_canvas = tk.Canvas(canvas_frame, bg="gray")
        self.original_canvas.grid(row=0, column=0, padx=10, sticky="nsew")
        tk.Label(canvas_frame, text="原图", font=("微软雅黑", 10)).grid(row=1, column=0)

        # 右画布：检测结果
        self.detected_canvas = tk.Canvas(canvas_frame, bg="gray")
        self.detected_canvas.grid(row=0, column=1, padx=10, sticky="nsew")
        tk.Label(
            canvas_frame,
            text="检测结果（绿色=哈啰单车 红色=汽车 蓝色=电动车）",
            font=("微软雅黑", 10)
        ).grid(row=1, column=1)

        # 自适应布局
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_columnconfigure(1, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)

    def select_image(self):
        """选择本地图片"""
        file_types = [
            ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("所有文件", "*.*")
        ]
        img_path = filedialog.askopenfilename(title="选择检测图片", filetypes=file_types)
        if not img_path:
            return

        self.selected_img_path = Path(img_path)

        try:
            # ✅ 始终保留“原始尺寸”的 PIL 图像（不要在显示时被缩放修改）
            self.original_img_full = Image.open(img_path).convert("RGB")

            # 确保画布尺寸已计算（提升首次显示的稳定性）
            self.root.update_idletasks()

            # 显示原图（用副本缩放展示，不改变 original_img_full）
            self.show_image_on_canvas(self.original_img_full, self.original_canvas)

            # 启用检测按钮
            self.detect_btn.config(state=tk.NORMAL)
            self.save_btn.config(state=tk.DISABLED)

            # 清空上一次检测结果
            self.detected_img_full = None
            self.detected_canvas.delete("all")

        except Exception as e:
            messagebox.showerror("错误", f"图片加载失败：{str(e)}")

    def detect_image(self):
        """检测图片并绘制结果"""
        if not self.selected_img_path or not self.model:
            return
        if self.original_img_full is None:
            messagebox.showwarning("提示", "请先选择图片！")
            return

        try:
            # 执行检测（results 会映射回原图尺寸坐标）
            results = self.model(str(self.selected_img_path))[0]

            # ✅ 用原始尺寸图像来绘制（不要用被缩放过的图）
            img_cv = cv2.cvtColor(np.array(self.original_img_full), cv2.COLOR_RGB2BGR)

            # 绘制检测框
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                cls_name = CLASS_NAMES.get(cls_id, f"未知类别{cls_id}")
                color = CLASS_COLORS.get(cls_id, (255, 255, 0))

                # 绘制矩形框
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), color, 2)

                # 绘制标签（注意：OpenCV 对中文可能显示为方块，不影响框）
                label = f"{cls_name} {conf:.2f}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # 标签背景位置（尽量放在框上方，放不下就放下方）
                y_text = y1 - 8
                y_bg_top = y_text - th - 6
                if y_bg_top < 0:
                    y_text = y1 + th + 12
                    y_bg_top = y_text - th - 6

                x_bg_left = x1
                x_bg_right = x1 + tw + 6
                y_bg_bottom = y_text + 4

                cv2.rectangle(img_cv, (x_bg_left, y_bg_top), (x_bg_right, y_bg_bottom), color, -1)
                cv2.putText(
                    img_cv, label, (x1 + 3, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                )

            # 转回PIL（保持原始尺寸）
            self.detected_img_full = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

            # 显示检测结果（显示时再缩放副本）
            self.root.update_idletasks()
            self.show_image_on_canvas(self.detected_img_full, self.detected_canvas)

            # 启用保存按钮
            self.save_btn.config(state=tk.NORMAL)
            messagebox.showinfo("提示", "检测完成！")

        except Exception as e:
            messagebox.showerror("错误", f"检测失败：{str(e)}")

    def show_image_on_canvas(self, img: Image.Image, canvas: tk.Canvas):
        """适配画布显示图片（✅ 永远用副本缩放，避免修改原图）"""
        canvas.delete("all")

        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()

        # 防止第一次渲染时拿到 1/0
        if canvas_width <= 2 or canvas_height <= 2:
            canvas_width, canvas_height = 400, 400

        # ✅ 用副本做缩放展示，不要改 img 本身
        show_img = img.copy()
        show_img.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(show_img)

        # 居中显示
        x = (canvas_width - img_tk.width()) // 2
        y = (canvas_height - img_tk.height()) // 2

        canvas.create_image(x, y, anchor=tk.NW, image=img_tk)
        canvas.image = img_tk  # 防止被回收

    def save_detected_img(self):
        """保存检测结果（保存原始尺寸的检测图）"""
        if self.detected_img_full is None:
            messagebox.warning("提示", "暂无检测结果可保存！")
            return
        if self.selected_img_path is None:
            messagebox.warning("提示", "未选择图片！")
            return

        save_dir = Path(DETECT_SAVE_DIR)
        save_dir.mkdir(parents=True, exist_ok=True)

        save_name = f"{self.selected_img_path.stem}_detected{self.selected_img_path.suffix}"
        save_path = save_dir / save_name

        try:
            self.detected_img_full.save(save_path)
            messagebox.showinfo("提示", f"检测结果已保存：\n{save_path}")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败：{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODetectorGUI(root)
    root.mainloop()
