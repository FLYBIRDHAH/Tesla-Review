# -*- coding: utf-8 -*-
"""Prompt模板集中管理模块"""

from typing import Dict

class PromptTemplate:
    """Prompt模板类，支持变量替换"""
    def __init__(self, template: str, description: str = ""):
        self.template = template
        self.description = description

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

class Prompts:
    """Prompt模板集合"""
    
    # 文档清洗整理的专属 Prompt
    CLEAN = PromptTemplate(
 """你是一个专业的文档整理助手，负责对汽车用户手册中的内容进行整理和总结。请根据以下要求对文档进行处理：

1. **让句子变得更加通顺**：重新整合句子、段落，去除一些不必要的符号，例如换行符等。
2. **按标题归类整理**：按照文档的语义关系，把属于同一个标题下的文档做归类合并, 记住标题要用markdown的形式加粗，例如###。
3.**不要输出你的思考过程**:只保留整理后的文档
{doc_content}
整理后的输出：
""",
        "文档清洗整理"
    )

# 便捷访问函数
def get_prompt(name: str, **kwargs) -> str:
    prompt_map: Dict[str, PromptTemplate] = {
        'clean': Prompts.CLEAN,
    }
    if name not in prompt_map:
        raise ValueError(f"找不到对应的 Prompt: {name}")
    return prompt_map[name].format(**kwargs)