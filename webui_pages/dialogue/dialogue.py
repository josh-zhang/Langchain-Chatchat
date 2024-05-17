import uuid
import base64
from datetime import datetime

import streamlit as st
from streamlit_chatbox import *
from streamlit_javascript import st_javascript

from configs import HISTORY_LEN
from server.knowledge_base.utils import LOADER_DICT
from server.utils import get_prompts
from webui_pages.utils import *

chat_box = ChatBox(
    assistant_avatar=os.path.join(
        "img",
        "chatchat_icon_blue_square_v2.png"
    )
)


def get_messages_history(history_len: int, content_in_expander: bool = False) -> List[Dict]:
    '''
    返回消息历史。
    content_in_expander控制是否返回expander元素中的内容，一般导出的时候可以选上，传入LLM的history不需要
    '''

    def filter(msg):
        content = [x for x in msg["elements"] if x._output_method in ["markdown", "text"]]
        if not content_in_expander:
            content = [x for x in content if not x._in_expander]
        content = [x.content for x in content]

        return {
            "role": msg["role"],
            "content": "\n\n".join(content),
        }

    return chat_box.filter_history(history_len=history_len, filter=filter)


@st.cache_data
def upload_temp_docs(files, document_loader_name, _api: ApiRequest) -> str:
    '''
    将文件上传到临时目录，用于文件问答
    返回临时向量库ID
    '''
    return _api.upload_temp_docs(files, document_loader_name).get("data", {}).get("id")


@st.cache_data(ttl=60)
def get_api_running_models(_api):
    available_models = _api.list_api_running_models()
    return available_models


@st.cache_data(ttl=60)
def get_knowledge_bases(_api):
    return _api.list_knowledge_bases()


def support_iframe(ftype):
    return ftype in ["text/plain", "application/pdf", "text/html"]


def dialogue_page(api: ApiRequest):
    running_model_dict = {k: v for k, v in get_api_running_models(api)}
    running_models = list(running_model_dict.keys())
    default_model = LLM_MODEL

    if not running_models:
        st.info("对话系统异常，暂时无法访问对话功能")
        return

    st.session_state.dialogue_mode = "闲聊"

    st.session_state.setdefault("conversation_ids", {})
    st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)

    if not chat_box.chat_inited:
        st.toast(
            f"欢迎使用智能对话系统! \n\n"
            f"当前运行的模型`{default_model}`, 您可以开始提问了."
        )
        chat_box.init_session()

    with st.sidebar:
        # 多会话
        conv_names = list(st.session_state["conversation_ids"].keys())
        index = 0
        if st.session_state.get("cur_conv_name") in conv_names:
            index = conv_names.index(st.session_state.get("cur_conv_name"))
        # conversation_name = st.selectbox("当前会话：", conv_names, index=index)
        conversation_name = conv_names[index]
        chat_box.use_chat_name(conversation_name)
        conversation_id = st.session_state["conversation_ids"][conversation_name]

        def on_llm_change():
            if llm_model:
                # config = api.get_model_config(llm_model)
                # if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                #     st.session_state["prev_llm_model"] = llm_model
                st.session_state["cur_llm_model"] = st.session_state.llm_model

        cur_llm_model = st.session_state.get("cur_llm_model", default_model)
        if cur_llm_model in running_models:
            index = running_models.index(cur_llm_model)
        else:
            index = 0

        llm_model = st.selectbox("选择对话模型：",
                                 running_models,
                                 index,
                                 # format_func=llm_model_format_func,
                                 on_change=on_llm_change,
                                 key="llm_model")

        temperature = st.slider("生成温度：", 0.0, 1.0, TEMPERATURE, 0.05)
        history_len = st.number_input("历史对话轮数：", 0, 20, HISTORY_LEN)

        prompt_dict = get_prompts("llm_chat")
        prompt_templates_kb_list = list(prompt_dict.keys())

        if "prompt_template_select" not in st.session_state:
            st.session_state.prompt_template_select = prompt_templates_kb_list[0]

        def prompt_change():
            st.toast(f"已切换为 {prompt_dict[st.session_state.prompt_template_select][0]} 模板。")

        def prompt_format_func(key):
            return prompt_dict[key][0]

        st.selectbox(
            "选择提示词模板：",
            prompt_templates_kb_list,
            index=0,
            on_change=prompt_change,
            format_func=prompt_format_func,
            key="prompt_template_select",
        )
        prompt_template_name = st.session_state.prompt_template_select

        with st.expander("当前提示词", False):
            st.text(f"{prompt_dict[prompt_template_name][1]}")

    # Display chat messages from history on app rerun
    chat_box.output_messages()

    def on_feedback(
            feedback,
            message_id: str = "",
            history_index: int = -1,
    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        api.chat_feedback(message_id=message_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "欢迎反馈您打分的理由",
    }

    if prompt := st.chat_input("请输入对话内容，换行请使用Shift+Enter。输入/help查看自定义命令 ", key="prompt",
                               max_chars=4000):
        history = get_messages_history(history_len)

        chat_box.user_say(prompt)
        chat_box.ai_say("正在思考...")
        text = ""
        message_id = ""
        r = api.chat_chat(prompt,
                          history=history,
                          conversation_id=conversation_id,
                          model=llm_model,
                          prompt_name=prompt_template_name,
                          temperature=temperature,
                          max_chars=running_model_dict[llm_model])
        for t in r:
            if error_msg := check_error_msg(t):  # check whether error occured
                st.error(error_msg)
                break
            text += t.get("text", "")
            chat_box.update_msg(text)
            message_id = t.get("message_id", "")

        metadata = {
            "message_id": message_id,
        }
        chat_box.update_msg(text, streaming=False, metadata=metadata)  # 更新最终的字符串，去除光标
        chat_box.show_feedback(**feedback_kwargs,
                               key=message_id,
                               on_submit=on_feedback,
                               kwargs={"message_id": message_id, "history_index": len(chat_box.history) - 1})

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

    now = datetime.now()
    with st.sidebar:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            chat_box.reset_history()
            st.rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )


def file_dialogue_page(api: ApiRequest):
    running_model_dict = {k: v for k, v in get_api_running_models(api)}
    running_models = list(running_model_dict.keys())
    default_model = LLM_MODEL

    if not running_models:
        st.info("对话系统异常，暂时无法访问对话功能")
        return

    st.session_state.dialogue_mode = "文件问答"

    st.session_state.setdefault("conversation_ids", {})
    st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
    st.session_state.setdefault("file_chat_id", None)
    st.session_state.setdefault("file_chat_value", None)
    st.session_state.setdefault("file_chat_content", "")
    st.session_state.setdefault("file_chat_type", None)

    if not chat_box.chat_inited:
        st.toast(
            f"欢迎使用文件问答系统! \n\n"
            f"当前运行的模型`{default_model}`, 您可以开始提问了."
        )
        chat_box.init_session()

    with st.sidebar:
        # 多会话
        conv_names = list(st.session_state["conversation_ids"].keys())
        index = 0
        if st.session_state.get("cur_conv_name") in conv_names:
            index = conv_names.index(st.session_state.get("cur_conv_name"))
        # conversation_name = st.selectbox("当前会话：", conv_names, index=index)
        conversation_name = conv_names[index]
        chat_box.use_chat_name(conversation_name)
        conversation_id = st.session_state["conversation_ids"][conversation_name]

        with st.expander("文件问答配置", True):
            doc_type = st.selectbox(
                "上传文件类型",
                ["普通文件", "知识库网页文件"],
                key="doc_type")

            if doc_type == "知识库网页文件":
                support_types = ["html"]
                document_loader_name = "CustomHTMLLoader"
            else:
                support_types = [i for ls in LOADER_DICT.values() for i in ls]
                document_loader_name = "default"

            single_file = st.file_uploader("将上传文件直接拖拽到下方区域（支持文件格式如下）：",
                                           support_types,
                                           help=f"支持文件格式包含 {support_types}",
                                           accept_multiple_files=False)

            if single_file:
                file_type = single_file.type.lower()
                bytes_data = single_file.getvalue()
                base64_pdf = base64.b64encode(bytes_data).decode("utf-8", 'ignore')

                st.session_state["file_chat_value"] = base64_pdf
                st.session_state["file_chat_type"] = file_type

            if st.button("开始上传", disabled=not single_file):
                st.session_state["file_chat_id"] = upload_temp_docs([single_file], document_loader_name, api)

            if st.session_state["file_chat_id"]:
                kb_top_k = st.number_input("搜索知识条数：", 1, 20, VECTOR_SEARCH_TOP_K)

                ## Bge 模型会超过1
                score_threshold = st.slider(f"搜索门槛 (门槛越高相似度要求越高，默认为{SCORE_THRESHOLD})：", 0.0, 1.0,
                                            float(SCORE_THRESHOLD), 0.01)
            else:
                kb_top_k = VECTOR_SEARCH_TOP_K
                score_threshold = float(SCORE_THRESHOLD)

        def on_llm_change():
            if llm_model:
                # config = api.get_model_config(llm_model)
                # if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                #     st.session_state["prev_llm_model"] = llm_model
                st.session_state["cur_llm_model"] = st.session_state.llm_model

        # def llm_model_format_func(x):
        #     if x in running_models:
        #         return f"{x} (运行中)"
        #     return x

        cur_llm_model = st.session_state.get("cur_llm_model", default_model)
        if cur_llm_model in running_models:
            index = running_models.index(cur_llm_model)
        else:
            index = 0

        llm_model = st.selectbox("选择对话模型：",
                                 running_models,
                                 index,
                                 # format_func=llm_model_format_func,
                                 on_change=on_llm_change,
                                 key="llm_model")

        temperature = st.slider("生成温度：", 0.0, 1.0, TEMPERATURE, 0.05)
        history_len = st.number_input("历史对话轮数：", 0, 20, HISTORY_LEN)

    col1, col2 = st.columns(spec=[1, 1], gap="small")

    with col1:
        inner_width = st_javascript("window.innerWidth")
        if inner_width:
            ui_width = max(inner_width - 10, 10)
        else:
            ui_width = 10

        file_type = st.session_state["file_chat_type"]
        file_value = st.session_state["file_chat_value"]
        if file_type and file_value and support_iframe(file_type):
            st.session_state["file_chat_content"] = st.text_area("请输入参考信息",
                                                                 value=st.session_state["file_chat_content"],
                                                                 height=200).strip()

            html_template = f'<iframe src="data:{file_type};base64,{file_value}" type"{file_type}" width={str(ui_width)} height={str(ui_width * 4 / 3)}></iframe>'
            st.markdown(html_template, unsafe_allow_html=True)
        else:
            st.session_state["file_chat_content"] = st.text_area("请输入参考信息",
                                                                 value=st.session_state["file_chat_content"],
                                                                 height=int(ui_width)).strip()

    prompt = st.chat_input("请输入对话内容，换行请使用Shift+Enter。输入/help查看自定义命令 ", key="prompt",
                           max_chars=2000)

    with col2:
        if prompt:
            chat_box.output_messages()

            history = get_messages_history(history_len)

            file_chat_id = st.session_state["file_chat_id"]
            knowledge_content = st.session_state["file_chat_content"]

            if file_chat_id is None and not knowledge_content:
                st.error("请先上传文件，或输入参考信息后，再进行对话")
                st.stop()

            if knowledge_content:
                file_chat_id = ""
                ai_say = f"正在阅读参考信息..."
            else:
                ai_say = f"正在查询文件 `{file_chat_id}` ..."

            chat_box.user_say(prompt)
            chat_box.ai_say([
                ai_say,
                Markdown("...", in_expander=True, title="文件搜索结果", state="complete"),
            ])
            text = ""
            d = None
            for d in api.file_chat(prompt,
                                   knowledge_id=file_chat_id,
                                   knowledge_content=knowledge_content,
                                   top_k=kb_top_k,
                                   score_threshold=score_threshold,
                                   history=history,
                                   model=llm_model,
                                   # prompt_name=prompt_template_name,
                                   temperature=temperature,
                                   max_chars=running_model_dict[llm_model]):
                if error_msg := check_error_msg(d):  # check whether error occured
                    st.error(error_msg)
                elif chunk := d.get("answer"):
                    text += chunk
                    chat_box.update_msg(text, element_index=0)
            chat_box.update_msg(text, element_index=0, streaming=False)
            if d:
                chat_box.update_msg("\n\n".join(d.get("docs", [])), element_index=1, streaming=False)

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

    now = datetime.now()
    with st.sidebar:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            # st.session_state.setdefault("conversation_ids", {})
            # st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
            st.session_state["file_chat_id"] = None
            st.session_state["file_chat_value"] = None
            st.session_state["file_chat_content"] = ""
            st.session_state["file_chat_type"] = None
            chat_box.reset_history()
            st.rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )


def kb_dialogue_page(api: ApiRequest):
    running_model_dict = {k: v for k, v in get_api_running_models(api)}
    running_models = list(running_model_dict.keys())
    default_model = LLM_MODEL

    if not running_models:
        st.info("对话系统异常，暂时无法访问对话功能")
        return

    st.session_state.dialogue_mode = "知识库问答"

    st.session_state.setdefault("conversation_ids", {})
    st.session_state["conversation_ids"].setdefault(chat_box.cur_chat_name, uuid.uuid4().hex)
    st.session_state.setdefault("cur_source_docs", [])

    if not chat_box.chat_inited:
        st.toast(
            f"欢迎使用知识库问答系统! \n\n"
            f"当前运行的模型`{default_model}`, 您可以开始提问了."
        )
        chat_box.init_session()

    with st.sidebar:
        # 多会话
        conv_names = list(st.session_state["conversation_ids"].keys())
        index = 0
        if st.session_state.get("cur_conv_name") in conv_names:
            index = conv_names.index(st.session_state.get("cur_conv_name"))
        # conversation_name = st.selectbox("当前会话：", conv_names, index=index)
        conversation_name = conv_names[index]
        chat_box.use_chat_name(conversation_name)
        conversation_id = st.session_state["conversation_ids"][conversation_name]

        with st.expander("知识库配置", True):
            kb_list = get_knowledge_bases(api)
            if kb_list is None:
                kb_list = []
            kb_dict = {kb[0]: kb[1] for kb in kb_list}
            kb_name_list = [kb[0] for kb in kb_list]

            def format_func(option):
                return kb_dict[option]

            def on_kb_change():
                st.toast(f"已加载知识库： {kb_dict[st.session_state.selected_kb]}")

            selected_kb = st.selectbox(
                "选择知识库：",
                kb_name_list,
                on_change=on_kb_change,
                format_func=format_func,
                key="selected_kb",
            )
            kb_top_k = st.number_input("搜索知识条数：", 1, 20, VECTOR_SEARCH_TOP_K)

            ## Bge 模型会超过1
            score_threshold = st.slider(f"搜索门槛 (门槛越高相似度要求越高，默认为{SCORE_THRESHOLD})：", 0.0, 1.0,
                                        float(SCORE_THRESHOLD), 0.01)

            has_source = 0 if st.session_state["cur_source_docs"] else 1
            kb_search_type = st.radio('问答搜索方式', ['继续问答', '重新搜索'],
                                      index=has_source,
                                      captions=["AI根据新的输入重新搜索知识库进行问答",
                                                "AI根据上方搜索结果进行问答"])

        def on_llm_change():
            if llm_model:
                # config = api.get_model_config(llm_model)
                # if not config.get("online_api"):  # 只有本地model_worker可以切换模型
                #     st.session_state["prev_llm_model"] = llm_model
                st.session_state["cur_llm_model"] = st.session_state.llm_model

        # def llm_model_format_func(x):
        #     if x in running_models:
        #         return f"{x} (运行中)"
        #     return x

        cur_llm_model = st.session_state.get("cur_llm_model", default_model)
        if cur_llm_model in running_models:
            index = running_models.index(cur_llm_model)
        else:
            index = 0

        llm_model = st.selectbox("选择对话模型：",
                                 running_models,
                                 index,
                                 # format_func=llm_model_format_func,
                                 on_change=on_llm_change,
                                 key="llm_model")

        temperature = st.slider("生成温度：", 0.0, 1.0, TEMPERATURE, 0.05)
        history_len = st.number_input("历史对话轮数：", 0, 20, HISTORY_LEN)

    # Display chat messages from history on app rerun
    chat_box.output_messages()

    def on_feedback(
            feedback,
            message_id: str = "",
            history_index: int = -1,
    ):
        reason = feedback["text"]
        score_int = chat_box.set_feedback(feedback=feedback, history_index=history_index)
        api.chat_feedback(message_id=message_id,
                          score=score_int,
                          reason=reason)
        st.session_state["need_rerun"] = True

    feedback_kwargs = {
        "feedback_type": "thumbs",
        "optional_text_label": "欢迎反馈您打分的理由",
    }

    if prompt := st.chat_input("请输入对话内容，换行请使用Shift+Enter。输入/help查看自定义命令 ", key="prompt",
                               max_chars=2000):
        history = get_messages_history(history_len)

        chat_box.user_say(prompt)
        chat_box.ai_say([
            f"正在查询知识库 `{kb_dict[selected_kb]}` ...",
            Markdown("...", in_expander=True, title="知识库搜索结果", state="complete"),
        ])
        text = ""
        d = None
        for d in api.knowledge_base_chat(prompt,
                                         knowledge_base_name=selected_kb,
                                         search_type=kb_search_type,
                                         top_k=kb_top_k,
                                         score_threshold=score_threshold,
                                         history=history,
                                         source=st.session_state[
                                             "cur_source_docs"] if kb_search_type == '继续问答' else [],
                                         model=llm_model,
                                         # prompt_name=prompt_template_name,
                                         temperature=temperature,
                                         max_chars=running_model_dict[llm_model]):
            if error_msg := check_error_msg(d):  # check whether error occured
                st.error(error_msg)
            elif chunk := d.get("answer"):
                text += chunk
                chat_box.update_msg(text, element_index=0)
        chat_box.update_msg(text, element_index=0, streaming=False)
        if d:
            this_search_type = d.get("search_type", "重新搜索")

            if this_search_type == "重新搜索":
                source_documents = d.get("docs", [])
                source_documents_content = d.get("docs_content", [])

                if source_documents:
                    source = ""
                    for i, j in zip(source_documents, source_documents_content):
                        source += i + j + "\n\n"

                    st.session_state["cur_source_docs"] = source_documents_content
                else:
                    source = f"<span style='color:red'>未找到相关文档,该回答为大模型自身能力解答！</span>"

                chat_box.update_msg(source, element_index=1, streaming=False)
            else:
                chat_box.update_msg(f"<span style='color:red'>继续利用上方搜索结果进行问答</span>",
                                    element_index=1, streaming=False)

    if st.session_state.get("need_rerun"):
        st.session_state["need_rerun"] = False
        st.rerun()

    now = datetime.now()
    with st.sidebar:
        cols = st.columns(2)
        export_btn = cols[0]
        if cols[1].button(
                "清空对话",
                use_container_width=True,
        ):
            st.session_state["cur_source_docs"] = []
            chat_box.reset_history()
            st.rerun()

    export_btn.download_button(
        "导出记录",
        "".join(chat_box.export2md()),
        file_name=f"{now:%Y-%m-%d %H.%M}_对话记录.md",
        mime="text/markdown",
        use_container_width=True,
    )
