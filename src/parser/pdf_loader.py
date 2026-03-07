import fitz  # 这是 PyMuPDF 的官方包名
import os

class TeslaPDFParser:
    def __init__(self, pdf_path, output_image_dir="data/extracted_images"):
        """
        初始化解析器
        :param pdf_path: PDF 文件的路径
        :param output_image_dir: 提取出的图片保存的目录
        """
        self.pdf_path = pdf_path
        self.output_image_dir = output_image_dir
        # 如果保存图片的文件夹不存在，自动创建一个
        os.makedirs(self.output_image_dir, exist_ok=True)

    def parse(self):
        print(f"🚀 开始解析 PDF: {self.pdf_path}")
        # 打开 PDF 文档
        doc = fitz.open(self.pdf_path)
        extracted_data = []

        # 逐页遍历文档
        for page_num in range(len(doc)):
            page = doc[page_num]

            # 1. 提取当页纯文本
            text = page.get_text()

            # 2. 提取当页所有图片
            image_list = page.get_images(full=True)
            saved_images = []

            for img_index, img in enumerate(image_list):
                # 获取图片的交叉引用号 (xref)
                xref = img[0]
                # 提取图片的基础数据
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # 构造图片文件名：页码_图片序号.扩展名 (例如: page_1_img_1.png)
                image_name = f"page_{page_num+1}_img_{img_index+1}.{image_ext}"
                image_path = os.path.join(self.output_image_dir, image_name)

                # 将图片写入本地硬盘
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                saved_images.append(image_path)

            # 把这页的文本和图片关联起来存入字典
            extracted_data.append({
                "page": page_num + 1,
                "text": text.strip(),
                "images": saved_images
            })

        print(f"✅ 解析完成！共处理了 {len(doc)} 页，提取的图片已妥善安放在 {self.output_image_dir}")
        return extracted_data

# ================= 测试代码 =================
if __name__ == "__main__":
    # 假设你的 PDF 已经放到了 data 目录下
    pdf_file = "data/Tesla_Manual.pdf" 
    
    # 稍微做个防御性检查
    if not os.path.exists(pdf_file):
        print(f"⚠️ 找不到文件！请先确保你把特斯拉手册重命名为 Tesla_Manual.pdf 并放在了 data/ 目录下哦！")
    else:
        # 实例化解析器并运行
        parser = TeslaPDFParser(pdf_file)
        result = parser.parse()
        
        # 打印第一页的结果预览一下
        print("\n--- 第一页解析结果预览 ---")
        print(f"该页提取到的图片路径: {result[0]['images']}")
        print(f"该页文本内容前 150 字: \n{result[0]['text'][:150]}...")