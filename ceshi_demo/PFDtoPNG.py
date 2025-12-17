# 这个程序生成转换的图片PNG 极有可能是“伪图像”或格式异常，导致 PaddleOCR 无法正确读取，从而全部识别失败。
"""
PDF 转 PNG 常见问题：

问题类型	                      表现	                            对 OCR 的影响
透明背景（RGBA）	          图像模式为 RGBA	         PaddleOCR 某些版本无法处理透明通道，返回空结果
单色/二值图像（1-bit）	      模式为 1（黑白）	                    检测模型可能无法提取特征
分辨率过低	                DPI < 150	                     文字像素太少，识别失败
嵌入式字体未渲染	        看似有文字，实为矢量轮廓	            转 PNG 时未正确光栅化，图像为空或乱码
损坏的 PNG 头	            文件能打开，但底层数据异常	        PaddleOCR 底层 CV 模块读取失败
"""#
# 用PDFtoPNG_2转

# ceshi15.py
from pdf2image import convert_from_path
import os

# === 配置区 ===
pdf_path = r"../rag_qa/samples/001.pdf"          # 输入 PDF 路径（相对或绝对）
output_dir = r"../rag_qa/samples/images"         # 输出 PNG 目录
poppler_path = r"F:\PDFtoPNG\poppler-25.07.0\Library\bin"  # ←←← 关键！你的 Poppler 路径

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

print(f"📄 正在将 {pdf_path} 转换为 PNG 图像...")
print(f"🔍 使用 Poppler 路径: {poppler_path}")

# 执行转换
try:
    images = convert_from_path(
        pdf_path,
        dpi=200,                # 分辨率：200 DPI 足够清晰
        fmt="png",              # 输出格式
        poppler_path=poppler_path  # 显式指定路径，绕过 PATH 问题
    )

    # 保存每一页
    for i, image in enumerate(images):
        output_file = os.path.join(output_dir, f"page_{i+1:03d}.png")
        image.save(output_file, "PNG")
        print(f"✅ 已保存: {output_file}")

    print(f"\n🎉 转换成功！共 {len(images)} 页，保存在: {os.path.abspath(output_dir)}")

except Exception as e:
    print(f"❌ 转换失败: {e}")
    print("\n请检查：")
    print("1. PDF 文件路径是否存在？")
    print("2. Poppler 路径是否正确？")
    print("3. 是否有中文/空格路径？建议用英文路径测试")