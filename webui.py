import sys

import streamlit as st
from auth.widgets import __login__

from webui_pages.utils import *
from streamlit_option_menu import option_menu
from webui_pages.dialogue.dialogue import dialogue_page
from webui_pages.dialogue.dialogue import file_dialogue_page
from webui_pages.dialogue.dialogue import kb_dialogue_page
from webui_pages.knowledge_base.knowledge_base import knowledge_base_page
from configs import VERSION
from server.utils import api_address

api = ApiRequest(base_url=api_address())

if __name__ == "__main__":
    st.set_page_config(
        "知识库问答系统",
        os.path.join("img", "chatchat_icon_blue_square_v2.png"),
        initial_sidebar_state="expanded",
        layout='wide',
        menu_items={
            'About': f"""欢迎使用 知识库问答系统 {VERSION}！"""
        }
    )

    __login__obj = __login__(width=200, height=250, hide_menu_bool=False, hide_footer_bool=False)

    LOGGED_IN = __login__obj.build_login_ui()

    if LOGGED_IN == True:
        is_lite = "lite" in sys.argv

        pages = {
            "知识库对话": {
                "icon": "chat",
                "func": kb_dialogue_page,
            },
            "文件对话": {
                "icon": "chat",
                "func": file_dialogue_page,
            },
            "闲聊对话": {
                "icon": "chat",
                "func": dialogue_page,
            },
            "知识库管理": {
                "icon": "hdd-stack",
                "func": knowledge_base_page,
            },
        }

        with st.sidebar:
            # st.image(
            #     os.path.join(
            #         "img",
            #         "logo-long-chatchat-trans-v2.png"
            #     ),
            #     use_column_width=True
            # )
            # st.caption(
            #     f"""<p align="right">当前版本：{VERSION}</p>""",
            #     unsafe_allow_html=True,
            # )
            options = list(pages)
            icons = [x["icon"] for x in pages.values()]

            default_index = 0
            selected_page = option_menu(
                "",
                options=options,
                icons=icons,
                # menu_icon="chat-quote",
                default_index=default_index,
            )

        if selected_page in pages:
            pages[selected_page]["func"](api=api)

        __login__obj.logout_widget()
