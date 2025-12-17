# ceshi15.py
from pdf2image import convert_from_path
from PIL import Image
import os
import sys

# === 配置区 ===
pdf_path = r"../rag_qa/samples/001.pdf"
output_dir = r"../rag_qa/samples/images3"
poppler_path = r"F:\PDFtoPNG\poppler-25.07.0\Library\bin"

# OCR 友好尺寸：长边不超过此值（RapidOCR 推荐 960~1200）
MAX_OCR_SIZE = 1200

# 转为绝对路径
pdf_path = os.path.abspath(pdf_path)
output_dir = os.path.abspath(output_dir)
os.makedirs(output_dir, exist_ok=True)

print(f"📄 正在将 PDF 转换为 PNG 图像...")
print(f"📥 PDF 路径: {pdf_path}")
print(f"📤 输出目录: {output_dir}")
print(f"⚙️  Poppler 路径: {poppler_path}")
print(f"📏 OCR 友好最大边长: {MAX_OCR_SIZE}")

if not os.path.isfile(pdf_path):
    print(f"❌ 错误: PDF 文件不存在！请检查路径:\n{pdf_path}")
    sys.exit(1)

try:
    images = convert_from_path(
        pdf_path,
        dpi=200,
        fmt="png",
        poppler_path=poppler_path,
        thread_count=2,
        grayscale=False
    )

    print(f"🖼️  成功加载 {len(images)} 页，正在处理并保存为 OCR 友好 PNG...")

    for i, image in enumerate(images):
        # === 步骤1: 转为 RGB + 白底 ===
        if image.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", image.size, (255, 255, 255))
            if image.mode in ("RGBA", "LA"):
                background.paste(image, mask=image.split()[-1])
            else:  # "P"
                image_rgba = image.convert("RGBA")
                background.paste(image_rgba, mask=image_rgba.split()[-1])
            image = background
        elif image.mode != "RGB":
            image = image.convert("RGB")

        # === 步骤2: 智能缩放（仅当图像太大时）===
        w, h = image.size
        if max(w, h) > MAX_OCR_SIZE:
            ratio = MAX_OCR_SIZE / max(w, h)
            new_w = int(w * ratio)
            new_h = int(h * ratio)
            # 使用高质量重采样
            image = image.resize((new_w, new_h), Image.LANCZOS)
            print(f"   ↪ 已缩放: {w}x{h} → {new_w}x{new_h}")

        # === 步骤3: 保存 ===
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
    sys.exit(1)