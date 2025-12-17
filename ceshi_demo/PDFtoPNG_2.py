# ceshi15.py
from pdf2image import convert_from_path
from PIL import Image
import os
import sys

# === 配置区 ===
pdf_path = r"../rag_qa/samples/001.pdf"
output_dir = r"../rag_qa/samples/images2"
poppler_path = r"F:\PDFtoPNG\poppler-25.07.0\Library\bin"

# 转为绝对路径，避免相对路径歧义
pdf_path = os.path.abspath(pdf_path)
output_dir = os.path.abspath(output_dir)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

print(f"📄 正在将 PDF 转换为 PNG 图像...")
print(f"📥 PDF 路径: {pdf_path}")
print(f"📤 输出目录: {output_dir}")
print(f"⚙️  Poppler 路径: {poppler_path}")

# 检查 PDF 是否存在
if not os.path.isfile(pdf_path):
    print(f"❌ 错误: PDF 文件不存在！请检查路径:\n{pdf_path}")
    sys.exit(1)

try:
    # 转换 PDF 为图像列表（默认返回 PIL Image 对象）
    images = convert_from_path(
        pdf_path,
        dpi=200,
        fmt="png",
        poppler_path=poppler_path,
        thread_count=2,  # 加速（可选）
        grayscale=False  # 保留彩色（若 PDF 是彩色）
    )

    print(f"🖼️  成功加载 {len(images)} 页，正在保存为标准 RGB PNG...")

    for i, image in enumerate(images):
        # === 关键修复：强制转为 RGB（白底） ===
        if image.mode in ("RGBA", "LA", "P"):
            # 创建白色背景
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode in ("RGBA", "LA"):
                background.paste(image, mask=image.split()[-1])  # 使用 alpha 通道
            else:  # mode == "P" (调色板)
                image = image.convert("RGBA")
                background.paste(image, mask=image.split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # 保存
        output_file = os.path.join(output_dir, f"page_{i+1:03d}.png")
        image.save(output_file, "PNG", optimize=True)
        print(f"✅ 已保存: {os.path.basename(output_file)} (模式: {image.mode}, 尺寸: {image.size})")

    print(f"\n🎉 转换成功！共 {len(images)} 页，保存在:\n{output_dir}")

except FileNotFoundError as e:
    print(f"❌ Poppler 未找到: {e}")
    print("请确认 poppler_path 是否指向包含 'pdftoppm.exe' 的 bin 目录")
    sys.exit(1)

except Exception as e:
    print(f"❌ 转换失败: {e}")
    import traceback
    traceback.print_exc()
    print("\n请检查：")
    print("1. PDF 文件是否损坏？")
    print("2. Poppler 路径是否正确？（应包含 pdftoppm.exe）")
    print("3. 路径是否含中文或特殊字符？建议全程使用英文路径")
    sys.exit(1)