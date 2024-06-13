import json
import asyncio
from typing import AsyncIterable, List, Optional

from fastapi import Body, Request
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from langchain.prompts.chat import ChatPromptTemplate
from urllib.parse import urlencode
from urllib.parse import urlparse
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler

from server.utils import wrap_done, get_ChatOpenAI
from server.chat.utils import History
from server.chat.prompt_generator import generate_doc_qa
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.kb_doc_api import search_docs
from server.knowledge_base.utils import huggingface_tokenizer_length
from server.utils import BaseResponse
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from server.db.repository import add_message_to_db
from configs import (LLM_MODEL,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,
                     BM_25_FACTOR,
                     TEMPERATURE,
                     API_SERVER_HOST_MAPPING,
                     API_SERVER_PORT_MAPPING, USE_BM25, USE_RERANK, USE_MERGE, DENSE_FACTOR, SPARSE_FACTOR)


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
                              conversation_id: str = Body("", description="对话框ID"),
                              knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
                              search_type: str = Body(..., description="搜索问答方式", examples=["重新搜索"]),
                              top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
                              score_threshold: float = Body(
                                  SCORE_THRESHOLD,
                                  description="知识库匹配相关度阈值，取值范围在0-1之间，SCORE越小，相关度越高，取到1相当于不筛选，建议设置在0.5左右",
                                  ge=0,
                                  le=2
                              ),
                              history: List[History] = Body(
                                  [],
                                  description="历史对话",
                                  examples=[[
                                      {"role": "user",
                                       "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                      {"role": "assistant",
                                       "content": "虎头虎脑"}]]
                              ),
                              source: List = Body(
                                  [],
                                  description="历史引用",
                                  examples=["引用正文1", "引用正文2"]
                              ),
                              stream: bool = Body(False, description="流式输出"),
                              model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: Optional[int] = Body(
                                  None,
                                  description="限制LLM总token数量"
                              ),
                              prompt_name: str = Body(
                                  "default",
                                  description="使用的prompt模板名称(在configs/prompt_config.py中配置)"
                              ),
                              request: Request = None,
                              ):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    is_search_enhance = kb.search_enhance
    rescale_factor = 1 + BM_25_FACTOR if is_search_enhance else 1
    score_threshold = score_threshold * rescale_factor

    if request:
        base_url = request.base_url.__str__()
        base_url_parsed = urlparse(base_url)
        hostname_altered = API_SERVER_HOST_MAPPING.get(base_url_parsed.hostname, base_url_parsed.hostname)
        port_altered = str(API_SERVER_PORT_MAPPING.get(base_url_parsed.port, base_url_parsed.port))
        base_url_altered = base_url_parsed.scheme + "://" + hostname_altered + ":" + port_altered + "/"
    else:
        base_url_altered = ""

    history = [History.from_data(h) for h in history]

    async def knowledge_base_chat_iterator(
            query: str,
            search_type: str,
            top_k: int,
            history: Optional[List[History]],
            source: Optional[List],
            model_name: str = model_name,
    ) -> AsyncIterable[str]:
        final_search_type = "继续问答" if search_type == "继续问答" and source else "重新搜索"

        if final_search_type == "重新搜索":
            docs = await run_in_threadpool(search_docs,
                                           query=query,
                                           knowledge_base_name=knowledge_base_name,
                                           top_k=top_k,
                                           max_tokens=max_tokens,
                                           score_threshold=score_threshold,
                                           use_bm25=USE_BM25,
                                           use_rerank=USE_RERANK,
                                           use_merge=USE_MERGE,
                                           dense_top_k_factor=DENSE_FACTOR,
                                           sparse_top_k_factor=SPARSE_FACTOR,
                                           sparse_factor=BM_25_FACTOR)
            docs = docs[:top_k]
            text_docs = [doc.page_content for doc in docs]
        else:
            docs = source
            text_docs = docs

        prompt_template, context, max_tokens_remain, input_token_counts = generate_doc_qa(query, history, text_docs,
                                                                                          "根据已知信息无法回答该问题",
                                                                                          max_tokens)

        if "总行" in model_name:
            callback = None
            streaming = False
            callbacks = []
        else:
            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]
            streaming = stream

        # 负责保存llm response到message db
        message_id = add_message_to_db(chat_type="knowledge_base_chat", query=query, conversation_id=conversation_id)
        conversation_callback = ConversationCallbackHandler(conversation_id=conversation_id, message_id=message_id,
                                                            chat_type="knowledge_base_chat", query=query)
        callbacks.append(conversation_callback)

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens_remain,
            callbacks=callbacks,
            streaming=streaming
        )

        input_msg = History(role="user", content=prompt_template).to_msg_template(False)

        chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        source_documents = []
        source_documents_content = []

        if final_search_type == "重新搜索":
            for inum, doc in enumerate(docs):
                filename = doc.metadata.get("source")
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                url = f"{base_url_altered}knowledge_base/download_doc?" + parameters
                text = ""
                if inum > 0:
                    text = "--------\n"
                text += f"""出处 [{inum + 1}] [{filename}]({url})  匹配度 {int(doc.scores["total"] / rescale_factor * 100)}%\n\n"""
                source_documents.append(text)
                source_documents_content.append(doc.page_content)

        if streaming:
            # Begin a task that runs in the background.
            task = asyncio.create_task(wrap_done(
                chain.acall({"context": context, "question": query}),
                callback.done),
            )
            outputs = ""
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                outputs = token
                yield json.dumps({"answer": token, "message_id": message_id}, ensure_ascii=False)

            if outputs:
                token_counts = huggingface_tokenizer_length(outputs) + input_token_counts
            else:
                token_counts = input_token_counts

            yield json.dumps({"docs": source_documents,
                              "docs_content": source_documents_content,
                              "search_type": final_search_type,
                              "token_counts": token_counts}, ensure_ascii=False)

            await task
        else:
            answer = await chain.ainvoke({"context": context, "question": query})
            outputs = answer["text"]

            if outputs:
                token_counts = huggingface_tokenizer_length(outputs) + input_token_counts
            else:
                token_counts = input_token_counts

            yield json.dumps({"answer": outputs,
                              "docs": source_documents,
                              "docs_content": source_documents_content,
                              "search_type": final_search_type,
                              "token_counts": token_counts,
                              "message_id": message_id},
                             ensure_ascii=False)

    return EventSourceResponse(
        knowledge_base_chat_iterator(query, search_type, top_k, history, source, model_name))
