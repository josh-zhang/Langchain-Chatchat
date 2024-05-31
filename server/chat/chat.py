import json
import asyncio
from typing import List, Optional, Union, AsyncIterable

from fastapi import Body
from sse_starlette.sse import EventSourceResponse
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts import PromptTemplate

from server.chat.utils import History
from server.utils import get_prompt_template, get_ChatOpenAI, wrap_done
from server.db.repository import add_message_to_db
from configs import LLM_MODEL, TEMPERATURE
from server.callback_handler.conversation_callback_handler import ConversationCallbackHandler
from server.memory.conversation_db_buffer_memory import ConversationBufferDBMemory


async def chat(query: str = Body(..., description="用户输入", examples=["恼羞成怒"]),
               conversation_id: str = Body("", description="对话框ID"),
               history_len: int = Body(-1, description="从数据库中取历史消息的数量"),
               history: Union[int, List[History]] = Body([],
                                                         description="历史对话，设为一个整数可以从数据库中读取历史消息",
                                                         examples=[[
                                                             {"role": "user",
                                                              "content": "我们来玩成语接龙，我先来，生龙活虎"},
                                                             {"role": "assistant", "content": "虎头虎脑"}]]
                                                         ),
               stream: bool = Body(False, description="流式输出"),
               model_name: str = Body(LLM_MODEL, description="LLM 模型名称。"),
               temperature: float = Body(TEMPERATURE, description="LLM 采样温度", ge=0.0, le=2.0),
               max_tokens: Optional[int] = Body(None, description="限制LLM总token数量"),
               # top_p: float = Body(TOP_P, description="LLM 核采样。勿与temperature同时设置", gt=0.0, lt=1.0),
               prompt_name: str = Body("default", description="使用的prompt模板名称(在configs/prompt_config.py中配置)"),
               ):
    async def chat_iterator() -> AsyncIterable[str]:
        nonlocal history

        if "总行" in model_name:
            callback = None
            streaming = False
            callbacks = []
        else:
            callback = AsyncIteratorCallbackHandler()
            callbacks = [callback]
            streaming = stream

        memory = None

        # 负责保存llm response到message db
        message_id = add_message_to_db(chat_type="llm_chat", query=query, conversation_id=conversation_id)
        conversation_callback = ConversationCallbackHandler(conversation_id=conversation_id, message_id=message_id,
                                                            chat_type="llm_chat",
                                                            query=query)
        callbacks.append(conversation_callback)
        # if isinstance(max_tokens, int) and max_tokens <= 0:
        #     max_tokens = None

        model = get_ChatOpenAI(
            model_name=model_name,
            temperature=temperature,
            max_tokens=None,
            callbacks=callbacks,
            streaming=streaming
        )

        if history:  # 优先使用前端传入的历史消息
            history = [History.from_data(h) for h in history]
            prompt_template = get_prompt_template("llm_chat", prompt_name)[1]
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages(
                [i.to_msg_template() for i in history] + [input_msg])
        elif conversation_id and history_len > 0:  # 前端要求从数据库取历史消息
            # 使用memory 时必须 prompt 必须含有memory.memory_key 对应的变量
            prompt = get_prompt_template("llm_chat", "with_history")[1]
            chat_prompt = PromptTemplate.from_template(prompt)
            # 根据conversation_id 获取message 列表进而拼凑 memory
            memory = ConversationBufferDBMemory(conversation_id=conversation_id,
                                                llm=model,
                                                message_limit=history_len)
        else:
            prompt_template = get_prompt_template("llm_chat", prompt_name)[1]
            input_msg = History(role="user", content=prompt_template).to_msg_template(False)
            chat_prompt = ChatPromptTemplate.from_messages([input_msg])

        chain = LLMChain(prompt=chat_prompt, llm=model, memory=memory)

        if streaming:
            # Begin a task that runs in the background.
            task = asyncio.create_task(wrap_done(
                chain.acall({"input": query}),
                callback.done),
            )

            async for token in callback.aiter():
                # Use server-sent-events to stream the response
                yield json.dumps(
                    {"text": token, "message_id": message_id},
                    ensure_ascii=False)

            await task
        else:
            answer = await chain.ainvoke({"input": query})

            yield json.dumps(
                {"text": answer["text"], "message_id": message_id},
                ensure_ascii=False)

    return EventSourceResponse(chat_iterator())
