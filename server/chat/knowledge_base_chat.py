import requests
from fastapi import Body, Request
from fastapi.responses import StreamingResponse
from configs import LLM_MODELS, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, TEMPERATURE, SAVE_CHAT_HISTORY, TOP_P
from server.utils import BaseResponse, get_prompt_template
from typing import List, Optional
from langchain.prompts.chat import ChatPromptTemplate
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.utils import get_doc_path
import json
from pathlib import Path
from urllib.parse import urlencode
from server.knowledge_base.kb_doc_api import search_docs
from server.db.repository import add_chat_history_to_db, update_chat_history
import sseclient


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
                              top_p: float = Body(TOP_P, description="LLM 核采样", gt=0.0, lt=1.0),
                              max_tokens: Optional[int] = Body(None,
                                                               description="限制LLM生成Token数量，默认None代表模型最大值"),
                              prompt_name: str = Body("default",
                                                      description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
                              request: Request = None,
                              ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    docs = search_docs(query, knowledge_base_name, top_k, score_threshold)

    if len(docs) == 0:  ## 如果没有找到相关文档，使用Empty模板
        prompt_template = get_prompt_template("knowledge_base_chat", "Empty")
    else:
        prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)
    history = [History.from_data(h) for h in history]

    print(f"prompt_template {prompt_template}")
    print(f"history {history}")

    input_msg = History(role="user", content=prompt_template).to_msg_template(False)
    chat_prompt = ChatPromptTemplate.from_messages(
        [i.to_msg_template() for i in history] + [input_msg])

    print(f"chat_prompt {chat_prompt.messages}")
    print(f"chat_prompt {chat_prompt.input_variables}")

    context = "\n".join([doc.page_content for doc in docs])

    messages = []
    for chatMessagePromptTemplate in chat_prompt.messages:
        role = chatMessagePromptTemplate.role
        prompt = chatMessagePromptTemplate.prompt
        template = prompt.template
        template = template.replace("{{ question }}", query)
        template = template.replace("{{ context }}", context)
        messages.append({'role': role, 'content': template})

    print(f"messages\n{messages}")

    def output():
        payload = {
            'model': model_name,
            'key': 'kbqa',
            'messages': messages,
            'top_p': top_p,
            'max_tokens': max_tokens,
            'max_tokens_single_turn': 500,
            'min_tokens_single_turn': 100,
            'temperature': temperature,
        }
        headers = {
            "Content-Type": "application/json",
            'Accept': 'text/event-stream'
        }
        url = "http://127.0.0.1:8000/v2/kbchat/completions"

        def with_urllib3(url, headers):
            """Get a streaming response for the given event feed using urllib3."""
            import urllib3
            http = urllib3.PoolManager()
            return http.request('GET', url, preload_content=False, headers=headers)

        def with_requests(url, headers):
            """Get a streaming response for the given event feed using requests."""
            import requests
            return requests.post(url, stream=True, headers=headers)

        def with_httpx(url, headers, payload):
            """Get a streaming response for the given event feed using httpx."""
            import httpx
            with httpx.stream('POST', url, headers=headers, json=payload) as s:
                # Note: 'yield from' is Python >= 3.3. Use for/yield instead if you
                # are using an earlier version.
                yield from s.iter_bytes()

        response = with_httpx(url, headers, payload)  # or with_requests(url, headers)
        client = sseclient.SSEClient(response)

        answer = ""
        source_documents = []

        for event in client.events():
            chat_history_id = add_chat_history_to_db(chat_type="llm_chat", query=query)

            data = event.data

            if not data:
                yield json.dumps({"answer": answer, "chat_history_id": chat_history_id}, ensure_ascii=False)
                break

            data = eval(data)

            answer = data['answer'] if isinstance(data, dict) else ""

            doc_path = get_doc_path(knowledge_base_name)
            for inum, doc in enumerate(docs):
                filename = Path(doc.metadata["source"]).resolve().relative_to(doc_path)
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                base_url = request.base_url
                url = f"{base_url}knowledge_base/download_doc?" + parameters
                text = f"""出处 [{inum + 1}] [{filename}]({url}) \n\n{doc.page_content}\n\n"""
                source_documents.append(text)
            if len(source_documents) == 0:  # 没有找到相关文档
                source_documents.append(
                    f"""<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>""")

            if SAVE_CHAT_HISTORY and len(chat_history_id) > 0:
                # 后续可以加入一些其他信息，比如真实的prompt等
                update_chat_history(chat_history_id, response=answer)

        yield json.dumps({"docs": source_documents}, ensure_ascii=False)

    return StreamingResponse(output(), media_type="text/event-stream")
