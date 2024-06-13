from typing import List

from server.chat.utils import History
from server.knowledge_base.utils import huggingface_tokenizer_length, truncate_string_by_token_limit
from configs import logger


def document_prompt_template():
    return "## 参考编号: [[{doc_id}]] 参考正文: {page_content}\n"


def get_prompt(fallback: str, history: List[History], has_context: bool) -> str:
    chat_history = ""
    for his in history:
        chat_history += f"{his.role}: {his.content}\n"

    if has_context and chat_history:
        prompt_template = """你根据下面的参考信息和聊天历史，严格按照回答要求回答用户问题。

# 参考信息

{{ context }}
"""

    elif has_context and not chat_history:
        prompt_template = """你根据下面的参考信息，严格按照回答要求回答用户问题。

# 参考信息

{{ context }}
"""

    elif not has_context and chat_history:
        prompt_template = "你根据下面的聊天历史，严格按照回答要求回答用户问题。"
    else:
        prompt_template = "你严格按照回答要求回答用户问题。"

    if has_context:
        answer_prompts = ["1. 你只能根据上面参考信息中给出的事实来回答用户问题，不要胡编乱造。"]
        index = 2
        if len(fallback) > 0:
            answer_prompts.append(
                str(index) + ". " + """如果参考信息不足以回答用户问题，请直接回答："{fallback}"，并给出简单解释。""".format(
                    fallback=fallback))
            index += 1

        citation_prompt = "如果你给出的答案里引用了上面参考信息中的内容，请在答案结尾处添加你引用的参考编号，并用两个方括号括起来。示例：[[引用1]]、[[引用2]]"
        answer_prompts.append(str(index) + ". " + citation_prompt)
        index += 1
    else:
        answer_prompts = ["1. 你基于事实详细阐述。",
                          "2. 如果向用户提出澄清问题有助于回答问题，可以尝试提问。"]

    answer_prompts = "\n".join(answer_prompts)

    prompt_template += f"""# 回答要求

{answer_prompts}
"""

    if chat_history:
        prompt_template += f"""
# 聊天历史

{chat_history}
"""

    prompt_template += """
# 用户问题

{{ question }}
"""

    return prompt_template


def generate_doc_qa(query: str, history: List[History], docs: List[str], fallback: str, max_tokens, context: str = ""):
    has_context = len(context) > 0 or len(docs) > 0

    prompt_template = get_prompt(fallback, history, has_context)

    current_prompt = prompt_template.replace("{{ question }}", query)

    # logger.info(f"current_prompt: {current_prompt}")

    current_token_length = huggingface_tokenizer_length(current_prompt)

    max_tokens_for_context = max_tokens - current_token_length - 500

    # iterate over all documents
    if context:
        context = truncate_string_by_token_limit(context, max_tokens_for_context)
    else:
        for inum, doc in enumerate(docs):
            if not doc:
                continue
            source_id = inum + 1
            source_content = document_prompt_template().format(doc_id=f"引用{source_id}", page_content=doc) + "\n\n"
            context += source_content
            context, need_stop = truncate_string_by_token_limit(context, max_tokens_for_context)
            if need_stop:
                break

    current_prompt = prompt_template.replace("{{ question }}", query).replace("{{ context }}", context)

    # logger.info(f"current_prompt: {current_prompt}")

    current_token_length = huggingface_tokenizer_length(current_prompt)

    max_tokens_remain = max_tokens - current_token_length - 100

    logger.info(f"docQA max_tokens_remain {max_tokens_remain} prompt_template: {prompt_template}")

    max_tokens_remain = max(50, max_tokens_remain)

    return prompt_template, context, max_tokens_remain, current_token_length
