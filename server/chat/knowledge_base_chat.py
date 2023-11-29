import requests
from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs import (LLM_MODELS, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE)
from server.utils import wrap_done
from server.utils import BaseResponse, get_prompt_template
from langchain.callbacks import AsyncIteratorCallbackHandler
from typing import AsyncIterable, List, Optional
import asyncio
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.utils import get_doc_path
import json
from pathlib import Path
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(SCORE_THRESHOLD,
                                                            description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                                            ge=0, le=2),
                              history: List[History] = Body([],
                                                            description="历史对话",
                                                            examples=[[
                                                                {"role": "user",
                                                                 "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                                {"role": "assistant",
                                                                 "content": "虎头虎脑"}]]
                                                            ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: Optional[int] = Body(None,
                                                               description="限制LLM生成Token数量，默认None代表模型最大值"),
                              prompt_name: str = Body("default",
                                                      description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                              request: Request = None,
                              ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    history = [History.from_data(h) for h in history]

    docs = search_docs(query, knowledge_base_name, top_k, score_threshold)
    context = "\n".join([doc.page_content for doc in docs])
    if len(docs) == 0:  ## 如果没有找到相关文档，使用Empty模板
        prompt_template = get_prompt_template("knowledge_base_chat", "Empty")
    else:
        prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
    input_msg = History(role="user", content=prompt_template).to_msg_template(False)

    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])

    print(f"chat_prompt\n{chat_prompt}")

    payload = {
        'model': model_name,
        'key': '小明',
        'messages': [{'role': 'system', 'content': f"请阅读以下文章然后回答问题。"},
                     {'role': 'user', 'content': f"\n{context}\n问题：{query}"}],
        'top_p': 0.7,
        'max_tokens': max_tokens,
        'max_tokens_single_turn': 500,
        'min_tokens_single_turn': 100,
        'temperature': temperature,
    }
    header = {
        "Content-Type": "application/json"
    }

    response = requests.post("http://127.0.0.1:8000/v2/kbchat/completions", json=payload, headers=header)
    ans = json.loads(response.content.decode("utf-8"))
    answer = ans['answer'] if isinstance(ans, dict) else ""

    def output():
        source_documents = []
        doc_path = get_doc_path(knowledge_base_name)
        for inum, doc in enumerate(docs):
            filename = Path(doc.metadata["source"]).resolve().relative_to(doc_path)
            parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
            base_url = request.base_url
            url = f"{base_url}knowledge_base/download_doc?" + parameters
            text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
            source_documents.append(text)

        if len(source_documents) == 0:  # 没有找到相关文档
            source_documents.append(f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")

        yield json.dumps({"answer": answer,
                          "docs": source_documents},
                         ensure_ascii=False)

    return StreamingResponse(output(), media_type="text/event-stream")
