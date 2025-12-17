# ==============================
# 必须是整个脚本的前几行！不能有任何 import 在上面！
# ==============================
import os
import sys

# 设置环境变量以避免中文路径问题
os.environ["HOME"] = r"F:\paddle_cache"          # 覆盖 ~ 的解析
os.environ["USERPROFILE"] = r"F:\paddle_cache"   # Windows 关键！
os.environ["PPX_HOME"] = r"F:\paddle_cache"
os.environ["PADDLEOCR_HOME"] = r"F:\paddle_cache"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["FLAGS_allocator_strategy"] = "auto_growth"

# 可选：强制重载（防止 Jupyter/IDE 缓存）
for m in list(sys.modules):
    if "paddle" in m or "paddlex" in m or "ppocr" in m:
        del sys.modules[m]

# ==============================
# 现在才安全导入
# ==============================
import logging
from paddleocr import PaddleOCR
import glob
from pathlib import Path

logging.getLogger("ppocr").setLevel(logging.WARNING)

def ocr_all_png_in_folder(folder_path: str, output_txt: str = "ocr_output.txt"):
    """
    读取 folder_path 下所有 .png 图片，OCR 识别后写入 output_txt
    """
    print("🚀 正在初始化 PaddleOCR（中文，CPU 模式）...")
    ocr = PaddleOCR(
        lang="ch",
        use_textline_orientation=False,
        device="cpu"
    )
    print("✅ 初始化完成！")

    png_files = sorted(glob.glob(os.path.join(folder_path, "*.png")))
    if not png_files:
        print(f"❌ 文件夹中未找到任何 .png 文件: {folder_path}")
        return

    print(f"📁 找到 {len(png_files)} 张 PNG 图片，开始 OCR...")
    all_results = []

    for idx, img_path in enumerate(png_files, start=1):
        print(f"📄 正在处理: {Path(img_path).name} ({idx}/{len(png_files)})")
        try:
            result = ocr.predict(img_path)
            page_text_lines = []
            if result and result[0]:  # Check if the result is not empty
                for line in result[0]:
                    text = line[1][0]  # Extracting the text from the detection result
                    page_text_lines.append(text)
            page_text = "\n".join(page_text_lines)
            all_results.append((Path(img_path).name, page_text))
        except Exception as e:
            print(f"⚠️ 第 {idx} 张图片处理失败: {e}")
            import traceback
            traceback.print_exc()  # Print full exception stack trace for debugging
            all_results.append((Path(img_path).name, "[OCR 失败]"))

    with open(output_txt, "w", encoding="utf-8") as f:
        for img_name, text in all_results:
            f.write(f"=== {img_name} ===\n")
            f.write(text + "\n\n")

    print(f"\n🎉 全部完成！结果已保存到: {os.path.abspath(output_txt)}")


if __name__ == "__main__":
    IMAGE_FOLDER = os.path.abspath("../rag_qa/samples/images2")  # 使用绝对路径确保正确性
    OUTPUT_FILE = "ocr_all_pages.txt"
    ocr_all_png_in_folder(IMAGE_FOLDER, OUTPUT_FILE)