# 该py文件可以实现读取某个文件夹得全部图片内容并把读取的内容写进一个txt文件


import os,sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,current_dir)
project_root = os.path.dirname(current_dir)
sys.path.insert(0,project_root)
from typing import Iterator
from rag_qa.edu_document_loaders import get_ocr
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader


class OCRIMGLoader(BaseLoader):
    """图片OCR识别加载器，支持单张图片识别"""

    def __init__(self, img_path: str) -> None:
        """初始化加载器

        Args:
            img_path: 图片路径
        """
        self.img_path = img_path

    def lazy_load(self) -> Iterator[Document]:
        """延迟加载文档"""
        text = self.img2text()
        yield Document(page_content=text, metadata={"source": self.img_path})

    def img2text(self) -> str:
        """将图片转换为文本"""
        ocr = get_ocr()
        result, _ = ocr(self.img_path)
        if result:
            ocr_result = [line[1] for line in result]
            return "\n".join(ocr_result)
        return ""


def process_image_folder(folder_path: str, output_file: str) -> None:
    """处理文件夹中的所有图片并将结果写入文件

    Args:
        folder_path: 图片文件夹路径
        output_file: 输出文本文件路径
    """
    # 支持的图片格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    # 获取文件夹中所有图片文件
    image_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(image_extensions) and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not image_files:
        print(f"在文件夹 {folder_path} 中未找到任何图片文件")
        return

    # 打开输出文件
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # 遍历所有图片文件
        for img_path in image_files:
            print(f"正在处理: {img_path}")
            try:
                # 创建加载器并获取识别结果
                loader = OCRIMGLoader(img_path)
                documents = loader.load()

                # 写入文件，包含图片路径作为标识
                f_out.write(f"=== 图片: {img_path} ===\n")
                f_out.write(documents[0].page_content)
                f_out.write("\n\n")  # 不同图片内容之间空两行分隔

            except Exception as e:
                print(f"处理 {img_path} 时出错: {str(e)}")

    print(f"所有图片处理完成，结果已保存至 {output_file}")


if __name__ == '__main__':
    # 示例：处理samples/images2文件夹中的所有图片，结果保存到ocr_results.txt
    image_folder = '../rag_qa/samples/images2/'  # 图片文件夹路径
    output_txt = 'ocr_results.txt'  # 输出文本文件路径

    # 执行处理
    process_image_folder(image_folder, output_txt)