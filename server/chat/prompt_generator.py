from typing import List

from server.chat.utils import History


def document_prompt_template():
    return """"Source_id": {doc_id}\n"参考正文": {page_content}"""


def get_prompt(fallback: str, history: List[History], context: str) -> str:
    chat_history = ""
    for his in history:
        chat_history += f"{his.role}: {his.content}\n"

    if context and chat_history:
        prompt_template = """你用严谨风格，根据下面的参考信息和聊天历史，严格按照下面的回答要求回答用户问题。

### 参考信息 ###
{{ context }}"""

    elif context and not chat_history:
        prompt_template = """你用严谨风格，根据下面的参考信息，严格按照下面的回答要求回答用户问题。

### 参考信息 ###
{{ context }}"""

    elif not context and chat_history:
        prompt_template = "你用严谨风格，根据下面的聊天历史，严格按照下面的回答要求回答用户问题。"
    else:
        prompt_template = "你用严谨风格，严格按照下面的回答要求回答用户问题。"

    if context:
        answer_prompts = ["1. 你只能根据上面参考信息中给出的事实来回答用户问题，不要胡编乱造。",
                          "2. 如果向用户提出澄清问题有助于回答问题，可以尝试提问。"]
        index = 3
        if len(fallback) > 0:
            answer_prompts.append(
                str(index) + ". " + """如果参考信息不足以回答用户问题，请直接回答："{fallback}"，并给出简单解释。""".format(
                    fallback=fallback))
            index += 1

        citation_prompt = "如果你给出的答案里引用了上面参考信息中的内容，请在答案结尾处添加你引用的Source_id，引用的Source_id值来自于参考信息中，并用两个方括号括起来。示例：[[出处1]]、[[出处2]]"
        answer_prompts.append(str(index) + ". " + citation_prompt)
        index += 1
    else:
        answer_prompts = ["1. 你基于事实详细阐述。",
                          "2. 如果向用户提出澄清问题有助于回答问题，可以尝试提问。"]

    answer_prompts = "\n".join(answer_prompts)

    prompt_template += f"""### 回答要求 ###
{answer_prompts}
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

    return prompt_template


def generate_doc_qa(query: str, history: List[History], docs: List[str], fallback: str, context: str = ""):
    print(f"query: {query}, docs: {docs}, fallback: {fallback}")

    # iterate over all documents
    if not context:
        for inum, doc in enumerate(docs):
            if not doc:
                continue
            source_id = inum + 1
            context += document_prompt_template().format(doc_id=f"出处{source_id}", page_content=doc) + "\n\n"

    prompt_template = get_prompt(fallback, history, context)

    print(f"docQA prompt_template: {prompt_template}")

    return prompt_template, context
