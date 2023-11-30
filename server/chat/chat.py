import requests
from fastapi import Body
from fastapi.responses import StreamingResponse
from configs import LLM_MODELS, TEMPERATURE, SAVE_CHAT_HISTORY, TOP_P
from server.utils import wrap_done
from typing import AsyncIterable
import asyncio
import json
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Optional
from server.chat.utils import History
from server.utils import get_prompt_template
from server.db.repository import add_chat_history_to_db, update_chat_history
import sseclient


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               history: List[History] = Body([],
                                             description="历史对话",
                                             examples=[[
                                                 {"role": "user", "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                 {"role": "assistant", "content": "虎头虎脑"}]]
                                             ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM生成Token数量，默认None代表模型最大值"),
               top_p: float = Body(TOP_P, description="LLM 核采样", gt=0.0, lt=1.0),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):

    prompt_template = get_prompt_template("llm_chat", prompt_name)
    history = [History.from_data(h) for h in history]

    print(f"prompt_template {prompt_template}")
    print(f"history {history}")

    context = ""

    messages = [{'role': 'system', 'content': f"请阅读以下文章然后回答问题。"},
                {'role': 'user', 'content': f"\n{context}\n问题：{query}"}]

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
        url = "http://127.0.0.1:8000/v2/chat/completions"

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
        for event in client.events():
            chat_history_id = add_chat_history_to_db(chat_type="llm_chat", query=query)

            data = event.data

            if not data:
                yield json.dumps({"answer": answer,
                                  "chat_history_id": chat_history_id},
                                 ensure_ascii=False)
                break

            data = eval(data)

            answer = data['answer'] if isinstance(data, dict) else ""

            if SAVE_CHAT_HISTORY and len(chat_history_id) > 0:
                # 后续可以加入一些其他信息，比如真实的prompt等
                update_chat_history(chat_history_id, response=answer)

    return StreamingResponse(output(), media_type="text/event-stream")