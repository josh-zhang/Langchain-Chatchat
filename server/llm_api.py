from fastapi import Body
from configs import logger, log_verbose, LLM_MODELS, HTTPX_DEFAULT_TIMEOUT
from server.utils import (BaseResponse, list_config_llm_models, get_httpx_client)
from copy import deepcopy


def list_running_models(
    controller_address: str = Body(None, description="Fastchat controller服务器地址"),
    placeholder: str = Body(None, description="该参数未使用，占位用"),
) -> BaseResponse:
    '''
    从fastchat controller获取已加载模型列表及其配置项
    '''
    try:
        # controller_address = controller_address or fschat_controller_address()
        controller_address = "http://127.0.0.1:8000"
        with get_httpx_client() as client:
            r = client.post(controller_address + "/v1/models")
            # models = r.json()["models"]
            # data = {m: get_model_config(m).data for m in models}
            data = r.json()['data']
            return BaseResponse(data=data)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get available models from controller: {controller_address}。错误信息是： {e}")


def list_config_models() -> BaseResponse:
    '''
    从本地获取configs中配置的模型列表
    '''
    configs = {}
    # 删除ONLINE_MODEL配置中的敏感信息
    for name, config in list_config_llm_models()["online"].items():
        configs[name] = {}
        for k, v in config.items():
            if not (k == "worker_class"
                or "key" in k.lower()
                or "secret" in k.lower()
                or k.lower().endswith("id")):
                configs[name][k] = v
    return BaseResponse(data=configs)


# def get_model_config(
#     model_name: str = Body(description="配置中LLM模型的名称"),
#     placeholder: str = Body(description="占位用，无实际效果")
# ) -> BaseResponse:
#     '''
#     获取LLM模型配置项（合并后的）
#     '''
#     config = {}
#     # 删除ONLINE_MODEL配置中的敏感信息
#     for k, v in get_model_worker_config(model_name=model_name).items():
#         if not (k == "worker_class"
#             or "key" in k.lower()
#             or "secret" in k.lower()
#             or k.lower().endswith("id")):
#             config[k] = v
#
#     return BaseResponse(data=config)


def stop_llm_model(
    model_name: str = Body(..., description="要停止的LLM模型名称", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址")
) -> BaseResponse:
    '''
    向fastchat controller请求停止某个LLM模型。
    注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
    '''
    try:
        controller_address = controller_address
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name},
            )
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"failed to stop LLM model {model_name} from controller: {controller_address}。错误信息是： {e}")


def change_llm_model(
    model_name: str = Body(..., description="当前运行模型", examples=[LLM_MODELS[0]]),
    new_model_name: str = Body(..., description="要切换的新模型", examples=[LLM_MODELS[0]]),
    controller_address: str = Body(None, description="Fastchat controller服务器地址")
):
    '''
    向fastchat controller请求切换LLM模型。
    '''
    try:
        controller_address = controller_address
        with get_httpx_client() as client:
            r = client.post(
                controller_address + "/release_worker",
                json={"model_name": model_name, "new_model_name": new_model_name},
                timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
            )
            return r.json()
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                        exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            msg=f"failed to switch LLM model from controller: {controller_address}。错误信息是： {e}")


def list_search_engines() -> BaseResponse:
    from server.chat.search_engine_chat import SEARCH_ENGINES

    return BaseResponse(data=list(SEARCH_ENGINES))
