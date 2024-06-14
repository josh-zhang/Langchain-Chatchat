import json
import os

import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_cookies_manager import EncryptedCookieManager

from .utils import check_usr_pass
from .utils import check_valid_name
from .utils import check_unique_usr
from .utils import register_new_usr


class __login__:
    """
    Builds the UI for the Login/ Sign Up page.
    """

    def __init__(self, width, height, logout_button_name: str = '退出登录',
                 hide_menu_bool: bool = False, hide_footer_bool: bool = False):
        """
        Arguments:
        -----------
        1. self
        4. width : Width of the animation on the login page.
        5. height : Height of the animation on the login page.
        6. logout_button_name : The logout button name.
        7. hide_menu_bool : Pass True if the streamlit menu should be hidden.
        8. hide_footer_bool : Pass True if the 'made with streamlit' footer should be hidden.
        """
        self.width = width
        self.height = height
        self.logout_button_name = logout_button_name
        self.hide_menu_bool = hide_menu_bool
        self.hide_footer_bool = hide_footer_bool

        self.cookies = EncryptedCookieManager(
            prefix="streamlit_login_cookies",
            password='9d68d6f2-4258-45c9-96eb-2d6bc74ddbb5-d8f49cab-edbb-404a-94d0-b25b1d4a564b')

        if not self.cookies.ready():
            st.stop()

    def check_auth_json_file_exists(self, auth_filename: str) -> bool:
        """
        Checks if the auth file (where the user info is stored) already exists.
        """
        file_names = []
        for path in os.listdir('./'):
            if os.path.isfile(os.path.join('./', path)):
                file_names.append(path)

        present_files = []
        for file_name in file_names:
            if auth_filename in file_name:
                present_files.append(file_name)

            present_files = sorted(present_files)
            if len(present_files) > 0:
                return True
        return False

    def login_widget(self) -> None:
        """
        Creates the login widget, checks and sets cookies, authenticates the users.
        """

        # Checks if cookie exists.
        if st.session_state['LOGGED_IN'] == False:
            if st.session_state['LOGOUT_BUTTON_HIT'] == False:
                fetched_cookies = self.cookies
                if '__streamlit_login_signup_ui_username__' in fetched_cookies.keys():
                    un = fetched_cookies['__streamlit_login_signup_ui_username__']
                    if un != '1c9a923f-fb21-4a91-b3f3-5f18e3f01182':
                        st.session_state['LOGGED_IN'] = True
                        st.session_state['LOGGED_USERNAME'] = un

        if st.session_state['LOGGED_IN'] == False:
            st.session_state['LOGOUT_BUTTON_HIT'] = False

            del_login = st.empty()
            with del_login.form("Login Form"):
                username = st.text_input("登录名", placeholder='请输入您的登录名')
                password = st.text_input("登录密码", placeholder='请输入您的登录密码', type='password')

                st.markdown("###")
                login_submit_button = st.form_submit_button(label='登录')

                if login_submit_button == True:
                    authenticate_user_check = check_usr_pass(username, password)

                    if authenticate_user_check == False:
                        st.error("登录名或登录密码错误")

                    else:
                        st.session_state['LOGGED_IN'] = True
                        st.session_state['LOGGED_USERNAME'] = username
                        self.cookies['__streamlit_login_signup_ui_username__'] = username
                        self.cookies.save()
                        del_login.empty()
                        st.rerun()

    def animation(self) -> None:
        pass

    def sign_up_widget(self) -> None:
        """
        Creates the sign-up widget and stores the user info in a secure way in the _secret_auth_.json file.
        """
        with st.form("Sign Up Form"):
            name_sign_up = st.text_input("用户昵称 *", placeholder='请输入您的昵称')
            valid_name_check = check_valid_name(name_sign_up)

            username_sign_up = st.text_input("登录名 *", placeholder='请输入您的登录名')
            unique_username_check = check_unique_usr(username_sign_up)

            password_sign_up = st.text_input("登录密码 *", placeholder='请输入您的登录密码', type='password')

            st.markdown("###")
            sign_up_submit_button = st.form_submit_button(label='注册')

            if sign_up_submit_button:
                if valid_name_check == False:
                    st.error("请输入您的昵称")

                elif unique_username_check == False:
                    st.error(f'抱歉, 登录名 {username_sign_up} 已经被注册')

                elif unique_username_check == None:
                    st.error('登录名不能为空')

                if valid_name_check == True:
                    if unique_username_check == True:
                        register_new_usr(name_sign_up, username_sign_up, password_sign_up)
                        st.success("注册成功")

    def logout_widget(self, cur_un:str) -> None:
        """
        Creates the logout widget in the sidebar only if the user is logged in.
        """
        if st.session_state['LOGGED_IN'] == True:
            del_logout = st.sidebar.empty()
            del_logout.markdown("#")
            logout_click_check = del_logout.button(f"用户 - {cur_un} | {self.logout_button_name}", use_container_width=True)

            if logout_click_check == True:
                st.session_state['LOGOUT_BUTTON_HIT'] = True
                st.session_state['LOGGED_IN'] = False
                st.session_state['LOGGED_USERNAME'] = ""
                self.cookies['__streamlit_login_signup_ui_username__'] = '1c9a923f-fb21-4a91-b3f3-5f18e3f01182'
                del_logout.empty()
                st.rerun()

    def nav_sidebar(self):
        """
        Creates the side navigaton bar
        """
        main_page_sidebar = st.sidebar.empty()
        with main_page_sidebar:
            selected_option = option_menu(
                menu_title='',
                # menu_icon='list-columns-reverse',
                icons=['box-arrow-in-right', 'person-plus'],
                options=['登录', '创建账户'],
                styles={
                    "container": {"padding": "5px"},
                    "nav-link": {"font-size": "14px", "text-align": "left", "margin": "0px"}})
        return main_page_sidebar, selected_option

    def hide_menu(self) -> None:
        """
        Hides the streamlit menu situated in the top right.
        """
        st.markdown(""" <style>
        #MainMenu {visibility: hidden;}
        </style> """, unsafe_allow_html=True)

    def hide_footer(self) -> None:
        """
        Hides the 'made with streamlit' footer.
        """
        st.markdown(""" <style>
        footer {visibility: hidden;}
        </style> """, unsafe_allow_html=True)

    def build_login_ui(self):
        """
        Brings everything together, calls important functions.
        """
        if 'LOGGED_IN' not in st.session_state:
            st.session_state['LOGGED_IN'] = False

        if 'LOGGED_USERNAME' not in st.session_state:
            st.session_state['LOGGED_USERNAME'] = ""

        if 'LOGOUT_BUTTON_HIT' not in st.session_state:
            st.session_state['LOGOUT_BUTTON_HIT'] = False

        auth_json_exists_bool = self.check_auth_json_file_exists('_secret_auth_.json')

        if auth_json_exists_bool == False:
            with open("_secret_auth_.json", "w") as auth_json:
                json.dump([], auth_json)

        main_page_sidebar, selected_option = self.nav_sidebar()

        if selected_option == '登录':
            c1, c2 = st.columns([7, 3])
            with c1:
                self.login_widget()
            with c2:
                if st.session_state['LOGGED_IN'] == False:
                    self.animation()

        if selected_option == '创建账户':
            self.sign_up_widget()

        # self.logout_widget()

        if st.session_state['LOGGED_IN'] == True:
            main_page_sidebar.empty()

        if self.hide_menu_bool == True:
            self.hide_menu()

        if self.hide_footer_bool == True:
            self.hide_footer()

        return st.session_state['LOGGED_IN'], st.session_state['LOGGED_USERNAME']
