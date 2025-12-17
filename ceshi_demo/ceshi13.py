# test_ocr_full.py
from paddleocr import PaddleOCR
from pdf2image import convert_from_path

# 1. 测试 PaddleOCR 是否能初始化
print("正在加载 PaddleOCR 模型...")
ocr = PaddleOCR(use_angle_cls=False, lang="ch", use_dilation=False)
print("✅ PaddleOCR 模型加载成功！")

# 2. （可选）测试 PDF 转图（先准备一个 test.pdf）
# images = convert_from_path("test.pdf", dpi=150)
# print(f"✅ PDF 转图成功，共 {len(images)} 页")