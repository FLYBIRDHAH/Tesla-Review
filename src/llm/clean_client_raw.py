# -*- coding: utf-8 -*-
import re
import sys
import os

# 确保能跨目录导入 config 和 parser
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.settings import CLEAN_HEADER_FOOTER_PATTERN, CLEANED_DATA_DIR
from src.parser.pdf_loader import TeslaPDFParser

class TextCleaner:
    def __init__(self):
        """初始化清洗器，加载 settings.py 中的正则规则"""
        self.header_footer_pattern = CLEAN_HEADER_FOOTER_PATTERN

    def clean_text(self, raw_text: str) -> str:
        """
        核心清洗逻辑：对单页的脏文本进行 4 步深度清洗
        """
        if not raw_text:
            return ""

        # 第 1 步：脱狗皮膏药 (剔除页眉、页脚、孤立的页码)
        # 使用配置里的正则，把匹配到的行直接替换为空
        text = self.header_footer_pattern.sub('', raw_text)

        # 第 2 步：缝合断句 (解决 PDF 强行换行的问题)
        # 逻辑：如果一个换行符旁边没有句号/问号/叹号等结束符，说明它是被排版强行切断的，我们要把它连起来。
        # 先把真正的段落标记（双换行）保护起来，替换成特殊的占位符
        text = text.replace('\n\n', '<PARAGRAPH_BREAK>')
        # 把剩下的单换行全部替换成一个空格，让句子连贯
        text = text.replace('\n', ' ')
        # 把保护起来的段落标记还原回去
        text = text.replace('<PARAGRAPH_BREAK>', '\n\n')

        # 第 3 步：压缩空间 (清理多余的空格和制表符)
        # \s+ 代表一个或多个空白字符，把它们统统变成一个单空格
        text = re.sub(r'\s+', ' ', text)

        # 第 4 步：首尾大扫除
        return text.strip()

    def process_document(self, parsed_data: list) -> list:
        """
        处理 pdf_loader 传过来的整本手册数据
        """
        print("🧽 净水器启动：开始清洗文档文本...")
        cleaned_data = []
        
        for page_data in parsed_data:
            original_text = page_data.get("text", "")
            cleaned_text = self.clean_text(original_text)
            
            # 只保留清洗后还有内容的页面
            if cleaned_text:
                cleaned_data.append({
                    "page": page_data["page"],
                    "text": cleaned_text,
                    "images": page_data.get("images", [])
                })
                
        print(f"✨ 清洗完毕！共输出 {len(cleaned_data)} 页有效干净文本。")
        return cleaned_data


# ================= 测试流水线 =================
if __name__ == "__main__":
    # 1. 启动收割机 (解析 PDF)
    print("--- 阶段 1：解析 PDF ---")
    parser = TeslaPDFParser()
    raw_data = parser.parse()

    # 2. 启动净水器 (清洗文本)
    print("\n--- 阶段 2：深度清洗 ---")
    cleaner = TextCleaner()
    clean_data = cleaner.process_document(raw_data)

    # 3. 抽查检验成果！
    if clean_data:
        test_page = clean_data[0] # 取第一页看看
        print(f"\n✅ 抽查第 {test_page['page']} 页清洗后的纯净文本 (前200字):")
        print("-" * 40)
        print(test_page['text'][:200])
        print("-" * 40)
        
        # 可选：把洗干净的数据保存到文本文件里，方便以后直接用
        """
        output_file = CLEANED_DATA_DIR / "cleaned_manual.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            for page in clean_data:
                f.write(page['text'] + "\n\n")
        print(f"💾 洗干净的全文已存档至: {output_file}")
        """