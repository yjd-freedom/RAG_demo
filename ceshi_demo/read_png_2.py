import os
import sys
from pathlib import Path

# 设置环境变量
os.environ['PPX_HOME'] = r'F:\paddle_cache'
print(f"当前PPX_HOME: {os.environ.get('PPX_HOME')}")


def ocr_folder_to_txt(image_folder: str, output_txt: str):
    """
    对文件夹内所有 PNG/JPG 图片进行 OCR，并合并为一个 TXT 文件
    """
    image_folder = Path(image_folder)
    output_txt = Path(output_txt)

    # 支持的图片格式
    image_extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = sorted([
        f for f in image_folder.iterdir()
        if f.is_file() and f.suffix.lower() in image_extensions
    ])

    if not image_files:
        print("❌ 未找到任何图片文件！")
        return

    print(f"📁 找到 {len(image_files)} 张图片，开始 OCR...")

    # 手动指定模型路径（F盘的实际模型目录）
    # 注意：检测模型和识别模型是两个不同的目录，需要分别指定
    det_model_dir = r"F:\paddle_cache\det"  # 检测模型目录
    rec_model_dir = r"F:\paddle_cache\rec"  # 识别模型目录
    print(f"📌 检测模型路径: {det_model_dir}")
    print(f"📌 识别模型路径: {rec_model_dir}")

    # 导入并初始化OCR引擎
    from paddleocr import PaddleOCR

    try:
        # 适配最新版本参数（移除show_log，分离检测和识别模型）
        ocr_engine = PaddleOCR(
            text_detection_model_dir=det_model_dir,       # 检测模型（单独目录）
            text_recognition_model_dir=rec_model_dir,     # 识别模型（单独目录）
            use_textline_orientation=False                # 不使用角度分类
            # 不指定lang，避免与自定义模型路径冲突
        )
        print(f"✅ OCR 引擎初始化成功: {type(ocr_engine)}")
    except Exception as e:
        print(f"❌ OCR 引擎初始化失败: {e}")
        # 检查模型目录是否存在
        for path in [det_model_dir, rec_model_dir]:
            if not os.path.exists(path):
                print(f"⚠️  模型目录不存在: {path}")
            else:
                # 检查关键文件是否存在
                required_files = ["inference.json", "model.pdiparams", "model.pdmodel"]
                missing = [f for f in required_files if not os.path.exists(os.path.join(path, f))]
                if missing:
                    print(f"⚠️  {path} 缺失文件: {missing}")
        return

    all_text = []

    for i, img_path in enumerate(image_files, 1):
        print(f"📄 正在处理: {img_path.name} ({i}/{len(image_files)})")
        try:
            result = ocr_engine.ocr(str(img_path), cls=False)
            if result and result[0]:
                # 提取所有识别出的文本（按顺序）
                page_text = "".join([line[1][0] for line in result[0]])
                all_text.append(page_text)
                print(f"✅ 识别到 {len(result[0])} 行文本")
            else:
                all_text.append("")  # 空页
                print("⚠️  未识别到文本")
        except Exception as e:
            print(f"⚠️  处理 {img_path.name} 时出错: {e}")
            all_text.append("")

    # 保存结果
    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("\n\n--- 分页分隔符 ---\n\n".join(all_text))

    print(f"✅ 全部完成！结果已保存到: {output_txt}")


if __name__ == "__main__":
    # 可根据需要修改图片文件夹和输出文件路径
    IMAGE_FOLDER = r"../rag_qa/samples/images3"
    OUTPUT_TXT = r"ocr_output2.txt"
    ocr_folder_to_txt(IMAGE_FOLDER, OUTPUT_TXT)