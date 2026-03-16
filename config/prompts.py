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
        """
        你是一个专业的文档整理助手，负责对汽车用户手册中的内容进行整理和总结。请根据以下要求对文档进行处理：

1. **让句子变得更加通顺**：重新整合句子、段落，去除一些不必要的符号，例如换行符等。
2. **按标题归类整理**：按照文档的语义关系，把属于同一个标题下的文档做归类合并, 记住标题要用markdown的形式加粗，例如###。
3.**不要输出你的思考过程**:只保留整理后的文档
{doc_content}
整理后的输出：
""",
        "文档清洗整理"
    )

    # HyDE 假设文档生成
    HYDE = PromptTemplate(
        """你是一位特斯拉Model 3的资深工程师。
为了帮助检索引擎在《用户手册》中找到最精准的段落，请你根据用户的提问，凭你的经验“伪造”一段可能出现在官方手册里的标准段落。
要求：只输出假设的说明书内容，使用客观严谨的说明文语气，包含尽可能多的相关专业术语，不要加任何废话解释，字数在100字以内。

【示例 1】
用户提问：方向盘怎么加热？
假设段落：要开启或关闭方向盘加热，请在触摸屏底部点击温度控制图标，然后点击方向盘图标。此功能可将方向盘保持在舒适的温度。

【示例 2】
用户提问：车要是没电了，怎么从里面开门？
假设段落：如果车辆断电，可以通过机械释放装置打开车门。在前排车门车窗开关前方，向上拉起机械式车门释放把手即可解锁并打开车门。

【当前任务】
用户提问：{query}
假设段落：""",
        "HyDE假设文档生成"
    )

    # RAG 问答复 prompt
    CHAT = PromptTemplate(
        """### 信息
{context}

### 任务
你是特斯拉电动汽车Model 3车型的用户手册问答系统，你具备{{信息}}中的知识。
请回答问题"{query}"，答案需要精准，语句通顺，并严格按照以下格式输出

{{答案}}【{{引用编号1}}, {{引用编号2}}, ...】
如果无法从中得到答案，请说 "无答案" ，不允许在答案中添加编造成分。
""",
        "RAG问答复prompt"
    )

# 便捷访问函数
def get_prompt(name: str, **kwargs) -> str:
    prompt_map: Dict[str, PromptTemplate] = {
        'clean': Prompts.CLEAN,
    }
    if name not in prompt_map:
        raise ValueError(f"找不到对应的 Prompt: {name}")
    return prompt_map[name].format(**kwargs)