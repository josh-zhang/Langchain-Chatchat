import os
from typing import Literal, Optional, Callable, Generator, Dict, Any, Awaitable, Union, Tuple, List
from pathlib import Path

import httpx

from configs import (API_SERVER, EMBEDDING_DEVICE, MODEL_PATH, MODEL_ROOT_PATH, logger, log_verbose,
                     HTTPX_DEFAULT_TIMEOUT, prompt_config, LITELLM_SERVER, KB_ROOT_PATH)

LOADER_DICT = {
    "UnstructuredHTMLLoader": ['.html', '.htm'],
    "UnstructuredMarkdownLoader": ['.md'],
    "JSONLoader": [".json"],
    "JSONLinesLoader": [".jsonl"],
    "CSVLoader": [".csv"],
    "RapidOCRPDFLoader": [".pdf"],
    "RapidOCRDocLoader": ['.docx'],
    "RapidOCRPPTLoader": ['.pptx', ],
    "RapidOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
    "UnstructuredFileLoader": ['.txt'],
    "UnstructuredExcelLoader": ['.xlsx', '.xls'],
    "UnstructuredTSVLoader": ['.tsv'],
    "UnstructuredXMLLoader": ['.xml'],
    # "UnstructuredWordDocumentLoader": ['.doc'],
    # "CustomHTMLLoader": ['.html'],
    # "MHTMLLoader": ['.mhtml'],
    # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
    # "UnstructuredEmailLoader": ['.eml', '.msg'],
    # "UnstructuredEPubLoader": ['.epub'],
    # "NotebookLoader": ['.ipynb'],
    # "UnstructuredODTLoader": ['.odt'],
    # "PythonLoader": ['.py'],
    # "UnstructuredRSTLoader": ['.rst'],
    # "UnstructuredRTFLoader": ['.rtf'],
    # "SRTLoader": ['.srt'],
    # "TomlLoader": ['.toml'],
    # "UnstructuredPowerPointLoader": ['.ppt', '.pptx'],
    # "TextLoader": ['.txt'],
    # "EverNoteLoader": ['.enex'],
}


def api_address() -> str:
    host = API_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = API_SERVER["port"]
    return f"http://{host}:{port}"


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_file_path(knowledge_base_name: str, doc_name: str):
    doc_path = Path(get_doc_path(knowledge_base_name)).resolve()
    file_path = (doc_path / doc_name).resolve()
    if str(file_path).startswith(str(doc_path)):
        return str(file_path)


def get_prompts(type: str) -> Optional[Dict]:
    '''
    从prompt_config中加载模板内容
    type: "llm_chat","agent_chat","knowledge_base_chat","search_engine_chat"的其中一种，如果有新功能，应该进行加入。
    '''
    return prompt_config.PROMPT_TEMPLATES.get(type)


def set_httpx_config(
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        proxy: Union[str, Dict] = None,
):
    '''
    设置httpx默认timeout。httpx默认timeout是5秒，在请求LLM回答时不够用。
    将本项目相关服务加入无代理列表，避免fastchat的服务器请求错误。(windows下无效)
    对于chatgpt等在线API，如要使用代理需要手动配置。搜索引擎的代理如何处置还需考虑。
    '''

    import httpx
    import os

    httpx._config.DEFAULT_TIMEOUT_CONFIG.connect = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.read = timeout
    httpx._config.DEFAULT_TIMEOUT_CONFIG.write = timeout

    # 在进程范围内设置系统级代理
    proxies = {}
    if isinstance(proxy, str):
        for n in ["http", "https", "all"]:
            proxies[n + "_proxy"] = proxy
    elif isinstance(proxy, dict):
        for n in ["http", "https", "all"]:
            if p := proxy.get(n):
                proxies[n + "_proxy"] = p
            elif p := proxy.get(n + "_proxy"):
                proxies[n + "_proxy"] = p

    for k, v in proxies.items():
        os.environ[k] = v

    # set host to bypass proxy
    no_proxy = [x.strip() for x in os.environ.get("no_proxy", "").split(",") if x.strip()]
    no_proxy += [
        # do not use proxy for locahost
        "http://127.0.0.1",
        "http://localhost",
    ]
    # do not use proxy for user deployed fastchat servers
    # for x in [
    #     fschat_controller_address(),
    #     # fschat_model_worker_address(),
    #     fschat_openai_api_address(),
    # ]:
    #     host = ":".join(x.split(":")[:2])
    #     if host not in no_proxy:
    #         no_proxy.append(host)
    os.environ["NO_PROXY"] = ",".join(no_proxy)

    # TODO: 简单的清除系统代理不是个好的选择，影响太多。似乎修改代理服务器的bypass列表更好。
    # patch requests to use custom proxies instead of system settings
    def _get_proxies():
        return proxies

    import urllib.request
    urllib.request.getproxies = _get_proxies

    # 自动检查torch可用的设备。分布式部署时，不运行LLM的机器上可以不装torch


def api_address() -> str:
    from configs.server_config import API_SERVER

    host = API_SERVER["host"]
    if host == "0.0.0.0":
        host = "127.0.0.1"
    port = API_SERVER["port"]
    return f"http://{host}:{port}"


def get_httpx_client(
        use_async: bool = False,
        proxies: Union[str, Dict] = None,
        timeout: float = HTTPX_DEFAULT_TIMEOUT,
        **kwargs,
) -> Union[httpx.Client, httpx.AsyncClient]:
    '''
    helper to get httpx client with default proxies that bypass local addesses.
    '''
    default_proxies = {
        # do not use proxy for locahost
        "all://127.0.0.1": None,
        "all://localhost": None,
    }
    # do not use proxy for user deployed fastchat servers
    # for x in [
    #     fschat_controller_address(),
    #     # fschat_model_worker_address(),
    #     fschat_openai_api_address(),
    # ]:
    #     host = ":".join(x.split(":")[:2])
    #     default_proxies.update({host: None})

    # get proxies from system envionrent
    # proxy not str empty string, None, False, 0, [] or {}
    default_proxies.update({
        "http://": (os.environ.get("http_proxy")
                    if os.environ.get("http_proxy") and len(os.environ.get("http_proxy").strip())
                    else None),
        "https://": (os.environ.get("https_proxy")
                     if os.environ.get("https_proxy") and len(os.environ.get("https_proxy").strip())
                     else None),
        "all://": (os.environ.get("all_proxy")
                   if os.environ.get("all_proxy") and len(os.environ.get("all_proxy").strip())
                   else None),
    })
    for host in os.environ.get("no_proxy", "").split(","):
        if host := host.strip():
            # default_proxies.update({host: None}) # Origin code
            default_proxies.update({'all://' + host: None})  # PR 1838 fix, if not add 'all://', httpx will raise error

    # merge default proxies with user provided proxies
    if isinstance(proxies, str):
        proxies = {"all://": proxies}

    if isinstance(proxies, dict):
        default_proxies.update(proxies)

    # construct Client
    kwargs.update(timeout=timeout, proxies=default_proxies)

    if log_verbose:
        logger.info(f'{get_httpx_client.__class__.__name__}:kwargs: {kwargs}')

    if use_async:
        return httpx.AsyncClient(**kwargs)
    else:
        return httpx.Client(**kwargs)
