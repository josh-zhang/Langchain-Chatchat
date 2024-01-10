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

from server.utils import BaseResponse, get_prompt_template, wrap_done, get_ChatOpenAI
from server.chat.utils import History
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.kb_doc_api import search_docs
from server.reranker.reranker import LangchainReranker
from server.utils import embedding_device
from configs import (LLM_MODELS,
                     VECTOR_SEARCH_TOP_K,
                     SCORE_THRESHOLD,
                     BM_25_FACTOR,
                     TEMPERATURE,
                     USE_RERANKER,
                     RERANKER_MODEL,
                     RERANKER_MAX_LENGTH,
                     API_SERVER_HOST_MAPPING,
                     API_SERVER_PORT_MAPPING,
                     MODEL_PATH)


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

    base_url = request.base_url.__str__()
    base_url_parsed = urlparse(base_url)
    hostname_altered = API_SERVER_HOST_MAPPING.get(base_url_parsed.hostname, base_url_parsed.hostname)
    port_altered = str(API_SERVER_PORT_MAPPING.get(base_url_parsed.port, base_url_parsed.port))
    base_url_altered = base_url_parsed.scheme + "://" + hostname_altered + ":" + port_altered + "/"

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
        callback = AsyncIteratorCallbackHandler()
        if isinstance(max_tokens, int) and max_tokens <= 0:
            max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=[callback],
        )

        final_search_type = "继续问答" if search_type == "继续问答" and source else "重新搜索"

        if final_search_type == "重新搜索":
            docs = await run_in_threadpool(search_docs,
                                           query=query,
                                           knowledge_base_name=knowledge_base_name,
                                           top_k=top_k,
                                           score_threshold=score_threshold)

            # 加入reranker
            if USE_RERANKER:
                reranker_model_path = MODEL_PATH["reranker"].get(RERANKER_MODEL, "BAAI/bge-reranker-large")
                print("-----------------model path------------------")
                print(reranker_model_path)
                reranker_model = LangchainReranker(top_n=top_k,
                                                   device=embedding_device(),
                                                   max_length=RERANKER_MAX_LENGTH,
                                                   model_name_or_path=reranker_model_path
                                                   )
                # print(docs)
                # docs = reranker_model.compress_documents(documents=docs, query=query)

                doc_list = list(docs)
                _docs = [d.page_content[:512] for d in doc_list]
                sentence_pairs = [[query, _doc] for _doc in _docs]
                results = reranker_model._model.predict(sentences=sentence_pairs,
                                              batch_size=reranker_model.batch_size,
                                              #  show_progress_bar=self.show_progress_bar,
                                              num_workers=reranker_model.num_workers,
                                              #  activation_fct=self.activation_fct,
                                              #  apply_softmax=self.apply_softmax,
                                              convert_to_tensor=True
                                              )
                top_k = reranker_model.top_n if reranker_model.top_n < len(results) else len(results)

                values, indices = results.topk(top_k)
                final_results = []
                for value, index in zip(values, indices):
                    doc = doc_list[index]
                    doc.metadata["relevance_score"] = value
                    final_results.append(doc)
                docs = final_results

                print("---------after rerank------------------")
                print(docs)
        else:
            docs = source

        if len(docs) == 0:  # 如果没有找到相关文档，使用empty模板
            prompt_template = get_prompt_template("knowledge_base_chat", "empty")
        else:
            prompt_template = get_prompt_template("knowledge_base_chat", prompt_name)

        input_msg = History(role="user", content=prompt_template).to_msg_template(False)
        chat_prompt = ChatPromptTemplate.from_messages(
            [i.to_msg_template() for i in history] + [input_msg])
        chain = LLMChain(prompt=chat_prompt, llm=model)

        header = "已知信息"
        if prompt_name == "faq":
            header = "常见问答"
        context = ""
        index = 1

        if final_search_type == "重新搜索":
            for doc in docs:
                context += f"\n##{header}{index}##\n{doc.page_content}\n"
                index += 1
        else:
            for doc in docs:
                context += f"\n##{header}{index}##\n{doc}\n"
                index += 1

        # Begin a task that runs in the background.
        task = asyncio.create_task(wrap_done(
            chain.acall({"context": context, "question": query}),
            callback.done),
        )

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

        # if len(source_documents) == 0:  # 没有找到相关文档
        #     source_documents.append(f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>")

        if stream:
            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps({"answer": token}, ensure_ascii=False)
            yield json.dumps({"docs": source_documents,
                              "docs_content": source_documents_content,
                              "search_type": final_search_type}, ensure_ascii=False)
        else:
            answer = ""
            async for token in callback.aiter():
                answer += token
            yield json.dumps({"answer": answer,
                              "docs": source_documents,
                              "docs_content": source_documents_content,
                              "search_type": final_search_type},
                             ensure_ascii=False)
        await task

    return EventSourceResponse(
        knowledge_base_chat_iterator(query, search_type, top_k, history, source, model_name, prompt_name))
