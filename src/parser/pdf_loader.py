# -*- coding: utf-8 -*-
import fitz  # PyMuPDF
import os
import sys

# 【关键】把项目根目录加入系统路径，确保能跨目录导入 config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 从配置中心引入绝对路径，告别硬编码！
from config.settings import SOURCE_PDF_FILE, IMAGE_SAVE_DIR

class TeslaPDFParser:
    def __init__(self, pdf_path=SOURCE_PDF_FILE, output_image_dir=IMAGE_SAVE_DIR):
        """
        初始化解析器，默认参数直接从 settings.py 中读取
        """
        self.pdf_path = pdf_path
        self.output_image_dir = output_image_dir
        os.makedirs(self.output_image_dir, exist_ok=True)

    def parse(self):
        print(f"🚀 开始解析 PDF: {self.pdf_path}")
        doc = fitz.open(self.pdf_path)
        extracted_data = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            image_list = page.get_images(full=True)
            saved_images = []

            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                image_name = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
                image_path = os.path.join(self.output_image_dir, image_name)

                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                saved_images.append(image_path)

            extracted_data.append({
                "page": page_num + 1,
                "text": text.strip(),
                "images": saved_images
            })

        print(f"✅ 解析完成！共处理了 {len(doc)} 页，提取的图片已妥善安放在 {self.output_image_dir}")
        return extracted_data

# ================= 测试代码 =================
if __name__ == "__main__":
    if not os.path.exists(SOURCE_PDF_FILE):
        print(f"⚠️ 找不到文件！请检查 {SOURCE_PDF_FILE} 是否存在！")
    else:
        parser = TeslaPDFParser()
        result = parser.parse()
        print("\n--- 第一页解析结果预览 ---")
        print(f"图片路径: {result[0]['images']}")
        print(f"文本内容前 150 字: \n{result[0]['text'][:150]}...")