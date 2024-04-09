import time
import datetime
import re

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

from server.knowledge_base.utils import get_file_path, LOADER_DICT
from server.knowledge_base.kb_service.base import get_kb_details, get_kb_file_details
from configs import kbs_config
from webui_pages.utils import *

# SENTENCE_SIZE = 100

cell_renderer = JsCode("""function(params) {if(params.value==true){return '✓'}else{return '×'}}""")


# Function to validate and clean the input
def validate_and_clean_input(user_input):
    # Regular expression for matching allowed characters
    pattern = re.compile('^[a-zA-Z0-9_]+$')
    if pattern.match(user_input):
        return user_input, True  # Input is valid
    else:
        # Replace disallowed characters with an empty string
        cleaned_input = re.sub('[^a-zA-Z0-9_]', '', user_input)
        return cleaned_input, False  # Input was invalid, but cleaned


def config_aggrid(
        df: pd.DataFrame,
        columns: Dict[Tuple[str, str], Dict] = {},
        selection_mode: Literal["single", "multiple", "disabled"] = "single",
        use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        # pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    gb.configure_pagination(
        enabled=True,
        paginationAutoPageSize=False,
        paginationPageSize=30
    )
    return gb


def file_exists(kb_name: str, selected_rows: List) -> Tuple[str, str]:
    """
    check whether a doc file exists in local knowledge base folder.
    return the file's name and path if it exists.
    """
    if selected_rows:
        file_name = selected_rows[0]["file_name"]
        file_path = get_file_path(kb_name, file_name)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""


@st.cache_data(ttl=300)
def get_embed_models(_api):
    return _api.list_embed_models()


def knowledge_base_page(api: ApiRequest, is_lite: bool = None):
    selected_kb_index = 0
    exist_kb_names = []
    exist_kb_infos = []
    kb_dict = {}

    kb_details = get_kb_details()

    try:
        kb_dict = {x["kb_name"]: x for x in kb_details}
        exist_kb_names = list(kb_dict.keys())
        exist_kb_infos = [x["kb_info"] for x in kb_details]
        if "selected_kb_name" in st.session_state and st.session_state["selected_kb_name"] in exist_kb_names:
            selected_kb_index = exist_kb_names.index(st.session_state["selected_kb_name"])
    except Exception as e:
        st.error(
            "获取知识库信息错误，请检查是否已按照 `README.md` 中 `4 知识库初始化与迁移` 步骤完成初始化或迁移，或是否为数据库连接错误。")
        st.stop()

    # if "selected_kb_info" not in st.session_state:
    #     st.session_state["selected_kb_info"] = ""

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_dict.get(kb_name):
            return f"{kb['kb_info']} ({kb['vs_type']} @ {kb['embed_model']})"
        else:
            return kb_name

    selected_kb = st.selectbox(
        "请选择或新建知识库：",
        exist_kb_names + ["新建知识库"],
        format_func=format_selected_kb,
        index=selected_kb_index
    )

    if selected_kb == "新建知识库":
        embed_models = get_embed_models(api)

        if not embed_models:
            st.info("系统异常，暂时无法新建知识库")
            return

        with st.form("新建知识库"):

            now = datetime.datetime.now()
            now_str = now.strftime("%Y%m%d")
            suggested_id = f"{now_str}_html"
            suggested_name = f"{now_str}源文件"

            new_kb_name = st.text_input(
                "新建知识库ID (仅支持英文字母、数字和下划线)",
                placeholder="新知识库ID (仅支持英文字母、数字和下划线)",
                key="kb_name",
                value=suggested_id,
                max_chars=50,
            )

            # Validate the input
            _, is_valid = validate_and_clean_input(new_kb_name)

            if not is_valid:
                st.error("新建知识库ID中仅支持包含英文字母、数字和下划线")

            new_kb_info = st.text_input(
                "知识库名称",
                placeholder="知识库名称",
                key="kb_info",
                value=suggested_name,
                max_chars=200,
            )
            new_kb_agent_guide = st.text_area(
                "知识库介绍",
                placeholder="知识库介绍",
                key="kb_agent_guide",
                value="",
                max_chars=200,
            )

            cols = st.columns(2)

            vs_types = list(kbs_config.keys())
            vs_type = cols[0].selectbox(
                "向量库类型",
                vs_types,
                index=vs_types.index(DEFAULT_VS_TYPE),
                key="vs_type",
            )

            embed_model = cols[1].selectbox(
                "Embedding 模型",
                embed_models,
                index=0,
                key="embed_model",
            )

            search_enhance = st.checkbox("开启检索加强", True)

            submit_create_kb = st.form_submit_button(
                "新建",
                # disabled=not is_valid,
                use_container_width=True,
            )

        if submit_create_kb and is_valid:
            _, is_valid = validate_and_clean_input(new_kb_name)
            if not new_kb_name or not new_kb_name.strip():
                st.error(f"知识库ID不能为空！")
            elif not is_valid:
                st.error("新建知识库ID中仅支持包含英文字母、数字和下划线")
            elif new_kb_name in kb_dict:
                st.error(f"ID为 {new_kb_name} 的知识库已经存在，请直接使用。如需重新创建，请先删除现有同ID知识库！")
            else:
                if not new_kb_info or not new_kb_info.strip():
                    st.error(f"知识库名称不能为空！")
                elif new_kb_info in exist_kb_infos:
                    st.error(f"名称为 {new_kb_info} 的知识库已经存在，请直接使用。如需重新创建，请先删除现有同名称知识库！")
                else:
                    ret = api.create_knowledge_base(
                        knowledge_base_name=new_kb_name,
                        knowledge_base_info=new_kb_info,
                        knowledge_base_agent_guide=new_kb_agent_guide,
                        vector_store_type=vs_type,
                        embed_model=embed_model,
                        search_enhance=search_enhance,
                    )
                    st.toast(ret.get("msg", " "))
                    st.session_state["selected_kb_name"] = new_kb_name
                    # st.session_state["selected_kb_info"] = kb_info
                    st.rerun()

    elif selected_kb:
        this_kb_name = selected_kb
        # st.session_state["selected_kb_info"] = kb_dict[kb]['kb_info']
        # 上传文件
        this_kb_info = st.text_area("知识库名称", value=kb_dict[this_kb_name]['kb_info'], max_chars=None,
                                    key=None, help=None, on_change=None, args=None, kwargs=None, disabled=True)
        this_kb_agent_guide = st.text_area("知识库介绍", value=kb_dict[this_kb_name]['kb_agent_guide'], max_chars=None,
                                           key=None, help=None, on_change=None, args=None, kwargs=None, disabled=True)

        doc_type = st.selectbox(
            "上传文件类型",
            ["普通文件", "知识库网页文件", "FAQ表格文件"],
            key="doc_type")

        if doc_type == "FAQ表格文件":
            support_types = ["xlsx"]
            document_loader_name = "CustomExcelLoader"
        elif doc_type == "知识库网页文件":
            support_types = ["html"]
            document_loader_name = "CustomHTMLLoader"
        else:
            support_types = [i for ls in LOADER_DICT.values() for i in ls]
            document_loader_name = "default"

        files = st.file_uploader("上传文件：",
                                 support_types,
                                 accept_multiple_files=True)

        if doc_type == "FAQ表格文件":
            chunk_size = 0
            chunk_overlap = 0
            zh_title_enhance = False
        else:
            with st.expander(
                    "文件处理配置",
                    expanded=False,
            ):
                cols = st.columns(3)
                chunk_size = cols[0].number_input("单段文本最大长度：", 1, 8000, CHUNK_SIZE)
                chunk_overlap = cols[1].number_input("相邻文本重合长度：", 0, chunk_size, OVERLAP_SIZE)
                cols[2].write("")
                cols[2].write("")
                zh_title_enhance = cols[2].checkbox("开启中文标题加强", ZH_TITLE_ENHANCE)

        if st.button(
                "添加文件到知识库",
                disabled=len(files) == 0,
        ):
            ret = api.upload_kb_docs(files,
                                     knowledge_base_name=this_kb_name,
                                     document_loader_name=document_loader_name,
                                     override=True,
                                     chunk_size=chunk_size,
                                     chunk_overlap=chunk_overlap,
                                     zh_title_enhance=zh_title_enhance)
            if msg := check_success_msg(ret):
                st.toast(msg, icon="✔")
            elif msg := check_error_msg(ret):
                st.toast(msg, icon="✖")

        st.divider()

        # 知识库详情
        # st.info("请选择文件，点击按钮进行操作。")
        kb_file_details = get_kb_file_details(this_kb_name)

        for kb_file_detail in kb_file_details:
            loader = kb_file_detail['document_loader']
            if loader == "CustomExcelLoader":
                kb_file_detail['file_type'] = "FAQ表格文件"
            elif loader == "CustomHTMLLoader":
                kb_file_detail['file_type'] = "知识库网页文件"
            else:
                kb_file_detail['file_type'] = "普通文件"

        doc_details = pd.DataFrame(kb_file_details)

        if not len(doc_details):
            st.info(f"知识库【`{this_kb_info}`】 中暂无文件")
        else:
            st.write(f"知识库【`{this_kb_info}`】中已有文件:")
            st.info("知识库中包含源文件与向量库，请从下表中选择文件后操作")
            doc_details.drop(columns=["kb_name"], inplace=True)
            doc_details = doc_details[[
                "No", "file_name", "document_loader", "text_splitter", "docs_count", "in_folder", "in_db",
            ]]
            # doc_details["in_folder"] = doc_details["in_folder"].replace(True, "✓").replace(False, "×")
            # doc_details["in_db"] = doc_details["in_db"].replace(True, "✓").replace(False, "×")
            gb = config_aggrid(
                doc_details,
                {
                    ("No", "序号"): {},
                    ("file_name", "文档名称"): {},
                    ("file_type", "文档类型"): {},
                    # ("file_ext", "文档类型"): {},
                    # ("file_version", "文档版本"): {},
                    ("docs_count", "文档数量"): {},
                    ("create_time", "创建时间"): {},
                    ("document_loader", "文档加载器"): {},
                    ("text_splitter", "分词器"): {},
                    ("in_folder", "源文件"): {"cellRenderer": cell_renderer},
                    ("in_db", "向量库"): {"cellRenderer": cell_renderer},
                },
                "multiple",
            )

            doc_grid = AgGrid(
                doc_details,
                gb.build(),
                columns_auto_size_mode="FIT_CONTENTS",
                theme="alpine",
                custom_css={
                    "#gridToolBar": {"display": "none"},
                },
                allow_unsafe_jscode=True,
                enable_enterprise_modules=False
            )

            selected_rows = doc_grid.get("selected_rows", [])

            cols = st.columns(3)

            # file_name, file_path = file_exists(this_kb_name, selected_rows)
            # if file_path:
            #     with open(file_path, "rb") as fp:
            #         cols[0].download_button(
            #             "下载选中文档",
            #             fp,
            #             file_name=file_name,
            #             use_container_width=True, )
            # else:
            #     cols[0].download_button(
            #         "下载选中文档",
            #         "",
            #         disabled=True,
            #         use_container_width=True, )
            if not selected_rows:
                cols[0].link_button(
                    "下载选中文档",
                    "",
                    use_container_width=True,
                    disabled=True,
                )

                # # 将文件分词并加载到向量库中
                # if cols[1].button(
                #         "添加至向量库",
                #         disabled=True,
                #         use_container_width=True,
                # ):
                #     pass
            else:
                selected_file_name = selected_rows[0]["file_name"]
                cols[0].link_button(
                    "下载选中文档",
                    f"{get_api_address_from_client()}/knowledge_base/download_doc?knowledge_base_name={this_kb_name}&file_name={selected_file_name}",
                    use_container_width=True,
                    disabled=False,
                )

                # # 将文件分词并加载到向量库中
                # if cols[1].button(
                #         "重新添加至向量库" if selected_rows and (
                #                 pd.DataFrame(selected_rows)["in_db"]).any() else "添加至向量库",
                #         disabled=False,
                #         use_container_width=True,
                # ):
                #     file_names = [row["file_name"] for row in selected_rows]
                #     api.update_kb_docs(this_kb_name,
                #                        file_names=file_names,
                #                        chunk_size=chunk_size,
                #                        chunk_overlap=chunk_overlap,
                #                        zh_title_enhance=zh_title_enhance)
                #     st.rerun()

            # 将文件从向量库中删除，但不删除文件本身。
            if cols[1].button(
                    "从向量库删除",
                    disabled=not (selected_rows and selected_rows[0]["in_db"]),
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                document_loaders = [row["document_loader"] for row in selected_rows]
                api.delete_kb_docs(this_kb_name, file_names=file_names, document_loaders=document_loaders)
                st.rerun()

            if cols[2].button(
                    "从知识库中删除",
                    type="primary",
                    use_container_width=True,
            ):
                file_names = [row["file_name"] for row in selected_rows]
                document_loaders = [row["document_loader"] for row in selected_rows]
                api.delete_kb_docs(this_kb_name, file_names=file_names, document_loaders=document_loaders,
                                   delete_content=True)
                st.rerun()

        st.divider()

        cols = st.columns(3)

        cols[0].link_button(
            "下载知识库中所有文档",
            f"{get_api_address_from_client()}/knowledge_base/download_knowledge_base_files?knowledge_base_name={this_kb_name}",
            use_container_width=True,
            type="primary",
        )

        if cols[1].button(
                "为知识库中文档生成问答",
                use_container_width=True,
        ):
            # st.toast(f"为知识库{this_kb_name}生成问答")
            ret = api.gen_qa_for_knowledge_base(this_kb_name, LLM_MODELS[0])
            st.toast(ret.get("msg", " "))
            time.sleep(1)
            st.rerun()

        # if cols[2].button(
        #         "依据源文件重建向量库",
        #         use_container_width=True,
        # ):
        #     with st.spinner("向量库重构中，请耐心等待，勿刷新或关闭页面。"):
        #         empty = st.empty()
        #         empty.progress(0.0, "")
        #         for d in api.recreate_vector_store(this_kb_name,
        #                                            this_kb_info,
        #                                            this_kb_agent_guide,
        #                                            chunk_size=chunk_size,
        #                                            chunk_overlap=chunk_overlap,
        #                                            zh_title_enhance=zh_title_enhance):
        #             if msg := check_error_msg(d):
        #                 st.toast(msg)
        #             else:
        #                 empty.progress(d["finished"] / d["total"], d["msg"])
        #         st.rerun()

        if cols[2].button(
                "删除知识库",
                use_container_width=True,
        ):
            ret = api.delete_knowledge_base(this_kb_name)
            st.toast(ret.get("msg", " "))
            time.sleep(1)
            st.rerun()
