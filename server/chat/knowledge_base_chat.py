import json
import asyncio
import requests
from typing import AsyncIterable, List, Optional

import torch
from fastapi import Body, Request
from fastapi.concurrency import run_in_threadpool
from sse_starlette.sse import EventSourceResponse
from langchain.prompts.chat import ChatPromptTemplate
from urllib.parse import urlencode
from urllib.parse import urlparse
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler

from server.utils import BaseResponse, get_prompt_template, wrap_done, get_ChatOpenAI
from server.chat.utils import History
from server.chat.prompt_generator import generate_doc_qa
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.kb_doc_api import search_docs
from server.knowledge_base.kb_cache.base import reranker_pool
from server.utils import BaseResponse, xinference_supervisor_address
from configs import (LLM_MODELS,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,
                     BM_25_FACTOR,
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     API_SERVER_HOST_MAPPING,
                     API_SERVER_PORT_MAPPING,
                     RERANKER_MAX_LENGTH)


def do_rerank(
        documents: List[str],
        query: str,
        top_n: Optional[int] = None,
        max_chunks_per_doc: Optional[int] = None,
        return_documents: Optional[bool] = None,
        model_name: str = RERANKER_MODEL,
):
    model_uid = model_name[:-4] if model_name.endswith("-api") else model_name
    url = f"{xinference_supervisor_address()}/v1/rerank"
    request_body = {
        "model": model_uid,
        "documents": documents,
        "query": query,
        "top_n": top_n,
        "max_chunks_per_doc": max_chunks_per_doc,
        "return_documents": return_documents,
    }
    response = requests.post(url, json=request_body)
    if response.status_code != 200:
        return []
    response_data = response.json()['results']
    return response_data


async def knowledge_base_chat(query: str = Body(..., description="用户输入", examples=["你好"]),
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
                              model_name: str = Body(LLM_MODELS[0], description="LLM 模型名称。"),
                              temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=1.0),
                              max_tokens: Optional[int] = Body(
                                  None,
                                  description="限制LLM生成Token数量，默认None代表模型最大值"
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
            prompt_name: str = prompt_name,
    ) -> AsyncIterable[str]:

        nonlocal max_tokens
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        if "总行" in model_name:
            streaming = False
            callbacks = []
        else:
            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]
            streaming = stream

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=callbacks,
            streaming=streaming
        )

        final_search_type = "继续问答" if search_type == "继续问答" and source else "重新搜索"

        if final_search_type == "重新搜索":
            docs = await run_in_threadpool(search_docs,
                                           query=query,
                                           knowledge_base_name=knowledge_base_name,
                                           top_k=top_k,
                                           score_threshold=score_threshold)

            # 加入reranker
            if USE_RERANKER and len(docs) > 1:
                doc_list = list(docs)
                remain_length = RERANKER_MAX_LENGTH - len(query)
                _docs = [d.page_content[:remain_length] for d in doc_list]

                final_results = []

                # results = do_rerank(_docs, query)
                # for i in results:
                #     idx = i['index']
                #     value = i['relevance_score']
                #     doc = doc_list[idx]
                #     doc.metadata["relevance_score"] = value
                #     final_results.append(doc)

                sentence_pairs = [[query, _doc] for _doc in _docs]
                scores = reranker_pool.get_score(sentence_pairs, RERANKER_MODEL)
                sorted_tuples = sorted([(value, index) for index, value in enumerate(scores)], key=lambda x: x[0], reverse=True)
                for value, index in sorted_tuples:
                    doc = doc_list[index]
                    doc.metadata["relevance_score"] = value
                    final_results.append(doc)

                docs = final_results

            docs = docs[:top_k]
            text_docs = [doc.page_content for doc in docs]
        else:
            docs = source
            text_docs = docs

        prompt_template, context = generate_doc_qa(query, history, text_docs, "根据已知信息无法回答该问题")

        input_msg = History(role="user", content=prompt_template).to_msg_template(False)

        chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model)

        source_documents = []
        source_documents_content = []

        if final_search_type == "重新搜索":
            for inum, doc in enumerate(docs):
                filename = doc.metadata.get("source")
                parameters = urlencode({"knowledge_base_name": knowledge_base_name, "file_name": filename})
                # base_url = request.base_url
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

            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents,
                              "docs_content": source_documents_content,
                              "search_type": final_search_type}, ensure_ascii=False)

            await task
        else:
            answer = await chain.ainvoke({"context": context, "question": query})

            yield json.dumps({"answer": answer,
                              "docs": source_documents,
                              "docs_content": source_documents_content,
                              "search_type": final_search_type},
                             ensure_ascii=False)

    return EventSourceResponse(
        knowledge_base_chat_iterator(query, search_type, top_k, history, source, model_name, prompt_name))
