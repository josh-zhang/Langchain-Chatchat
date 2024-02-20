import re
import unicodedata
from typing import List

from server.chat.utils import History


def language_detect(text: str) -> str:
    text = re.sub(r"([ ■◼•＊…— �●⚫]+|[·\.~•、—'}\n\t]{1,})", '', text.strip())
    stats = {
        "zh": 0,
        "ja": 0,
        "ko": 0,
        "en": 0,
        "th": 0,
        "other": 0
    }
    char_count = 0
    for char in text:
        try:
            code_name = unicodedata.name(char)
        except Exception:
            continue
        char_count += 1
        # 判断是否为中文
        if 'CJK' in code_name:
            stats["zh"] += 1
        # 判断是否为日文
        elif 'HIRAGANA' in code_name or 'KATAKANA' in code_name:
            stats["ja"] += 1
        # 判断是否为泰文
        elif "THAI" in code_name:
            stats["th"] += 1
        # 判断是否为韩文
        elif 'HANGUL' in code_name:
            stats["ko"] += 1
        # 判断是否为英文
        elif 'LA' in code_name:
            stats["en"] += 1
        else:
            stats["other"] += 1

    lang = ""
    ratio = 0.0
    for lan in stats:
        if lan == "other":
            continue
        # trick: 英文按字母统计不准确，除以4大致表示word个数
        if lan == "en":
            stats[lan] /= 4.0
        lan_r = float(stats[lan]) / char_count
        if ratio < lan_r:
            lang = lan
            ratio = lan_r

    return lang


def language_prompt(lan: str) -> str:
    _ZH_LANGUAGE_MAP = {
        "zh": "中文",
        "en": "英文",
        "other": "中文",
        "ja": "中文",
        "zh_gd": "中文",
        "ko": "韩文",
        "th": "泰文"
    }
    return _ZH_LANGUAGE_MAP.get(lan.lower(), "中文")


def document_prompt_template():
    return """["Source_id": {doc_id}, "Content": "{page_content}"]"""


def get_prompt(question: str, fallback: str, history: List[History], context: str, lan='') -> str:
    chat_history = ""
    for his in history:
        chat_history += f"'{his.role}': '{his.content}'\n"

    prompt_template = "你是AI助手。你可以根据下面给出的参考资料和聊天历史来回答用户问题。"

    if context:
        prompt_template += """
### 参考资料 ###
{{ context }}
"""

    if chat_history:
        prompt_template += f"""
### 聊天历史 ###
{chat_history}
"""

    prompt_template += """
### 用户问题 ###
{{ question }}
"""

    answer_prompts = ["1. 你只能根据上面参考资料中给出的事实信息来回答用户问题，不要胡编乱造。",
                      "2. 如果向用户提出澄清问题有助于回答问题，可以尝试提问。"]
    index = 3
    if len(fallback) > 0:
        answer_prompts.append(
            str(index) + ". " + """如果参考资料中的信息不足以回答用户问题，请直接回答下面三个双引号中的内容：\"\"\"{fallback}\"\"\"。""".format(
                fallback=fallback))
        index += 1

    citation_prompt = "如果你给出的答案里引用了参考资料中的内容，请在答案的结尾处添加你引用的Source_id，引用的Source_id值来自于参考资料中，并用两个方括号括起来。示例：[[出处1]]、[[出处2]]"
    answer_prompts.append(str(index) + ". " + citation_prompt)
    index += 1

    if not lan:
        lan = language_detect(question)

    style_prompt = """请你以第一人称并且用严谨的风格来回答问题，一定要用{language}来回答，并且基于事实详细阐述。""".format(
        language=language_prompt(lan),
    )
    answer_prompts.append(str(index) + ". " + style_prompt)
    answer_prompts = "\n".join(answer_prompts)
    # prompt = prompt_template.replace('{requirement}', answer_prompts)

    prompt_template += f"""
### 回答要求 ###
{answer_prompts}
"""

    return prompt_template


def generate_doc_qa(query: str, history: List[History], docs: List[str], fallback: str):
    """Generates chat responses according to the input text, history and page content."""
    # handle input params
    print(f"query: {query}, docs: {docs}, fallback: {fallback}")

    # iterate over all documents
    context = ""
    for inum, doc in enumerate(docs):
        if not doc:
            continue
        source_id = inum + 1
        context += document_prompt_template().format(doc_id=f"出处{source_id}", page_content=doc) + "\n\n"

    prompt_template = get_prompt(query, fallback, history, context, lan='zh')

    print(f"docQA prompt_template: {prompt_template}")

    return prompt_template, context
