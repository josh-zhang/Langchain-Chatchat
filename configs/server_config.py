import sys

# httpx 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。
HTTPX_DEFAULT_TIMEOUT = 300.0

# API 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False

# 各服务器默认绑定host。如改为"0.0.0.0"需要修改下方所有XX_SERVER的host
DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"

# webui.py server
WEBUI_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 8080,
}

LITELLM_SERVER = "198.203.120.5:40782"

# api.py server
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 6006,
}

API_SERVER_HOST_MAPPING = {
    "127.0.0.1": "198.203.120.8",
    "localhost": "198.203.120.8",
    "0.0.0.0": "198.203.120.8",
}

API_SERVER_PORT_MAPPING = {
    80: 40781,
    8000: 40782,
    8080: 40783,
    4141: 40784,
    6006: 40785,
    30000: 40786,
    30001: 40787,
    30002: 40788
}

# xinference server
XINFERENCE_API = {
    "host": "198.203.120.8",
    "port": 40778,
}

QA_JOB_SCRIPT_PATH = "/opt/projects/qabot-be/deploy/main.py"
