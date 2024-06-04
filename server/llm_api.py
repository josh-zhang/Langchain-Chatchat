from configs import logger, log_verbose, LITELLM_SERVER
from server.utils import BaseResponse, get_httpx_client


def list_api_running_models() -> BaseResponse:
    '''
    从LITELLM_SERVER获取已加载模型列表及其配置项
    '''
    try:
        with get_httpx_client() as client:
            r = client.get("http://" + LITELLM_SERVER + "/model/info")
            res = r.json()['data']
            model_list = [(i['model_name'], int(i["litellm_params"]["extra_headers"]["max_tokens"])) for i in res if
                          'mode' not in i['model_info'] or (
                                      'mode' in i['model_info'] and i['model_info']['mode'] != 'embedding')]
            model_list = list(set(model_list))
            return BaseResponse(data=model_list)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(
            code=500,
            data={},
            msg=f"failed to get available models from LLM_SERVER: {LITELLM_SERVER}。错误信息是： {e}")

# def list_config_models(
#         types: List[str] = Body(["local", "online"], description="模型配置项类别，如local, online, worker"),
#         placeholder: str = Body(None, description="占位用，无实际效果")
# ) -> BaseResponse:
#     '''
#     从本地获取configs中配置的模型列表
#     '''
#     data = {}
#     for type, models in list_config_llm_models().items():
#         if type in types:
#             data[type] = models
#     return BaseResponse(data=data)

# def get_model_config(
#         model_name: str = Body(description="配置中LLM模型的名称"),
#         placeholder: str = Body(None, description="占位用，无实际效果")
# ) -> BaseResponse:
#     '''
#     获取LLM模型配置项（合并后的）
#     '''
#     config = {}
#     # 删除ONLINE_MODEL配置中的敏感信息
#     for k, v in get_model_worker_config(model_name=model_name).items():
#         if not (k == "worker_class"
#                 or "key" in k.lower()
#                 or "secret" in k.lower()
#                 or k.lower().endswith("id")):
#             config[k] = v
#
#     return BaseResponse(data=config)

# def stop_llm_model(
#     model_name: str = Body(..., description="要停止的LLM模型名称", examples=[LLM_MODELS[0]]),
#     controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[])
# ) -> BaseResponse:
#     '''
#     向fastchat controller请求停止某个LLM模型。
#     注意：由于Fastchat的实现方式，实际上是把LLM模型所在的model_worker停掉。
#     '''
#     try:
#         controller_address = controller_address
#         with get_httpx_client() as client:
#             r = client.post(
#                 controller_address + "/release_worker",
#                 json={"model_name": model_name},
#             )
#             return r.json()
#     except Exception as e:
#         logger.error(f'{e.__class__.__name__}: {e}',
#                         exc_info=e if log_verbose else None)
#         return BaseResponse(
#             code=500,
#             msg=f"failed to stop LLM model {model_name} from controller: {controller_address}。错误信息是： {e}")


# def change_llm_model(
#     model_name: str = Body(..., description="当前运行模型", examples=[LLM_MODELS[0]]),
#     new_model_name: str = Body(..., description="要切换的新模型", examples=[LLM_MODELS[0]]),
#     controller_address: str = Body(None, description="Fastchat controller服务器地址", examples=[fschat_controller_address()])
# ):
#     '''
#     向fastchat controller请求切换LLM模型。
#     '''
#     try:
#         controller_address = controller_address or fschat_controller_address()
#         with get_httpx_client() as client:
#             r = client.post(
#                 controller_address + "/release_worker",
#                 json={"model_name": model_name, "new_model_name": new_model_name},
#                 timeout=HTTPX_DEFAULT_TIMEOUT, # wait for new worker_model
#             )
#             return r.json()
#     except Exception as e:
#         logger.error(f'{e.__class__.__name__}: {e}',
#                         exc_info=e if log_verbose else None)
#         return BaseResponse(
#             code=500,
#             msg=f"failed to switch LLM model from controller: {controller_address}。错误信息是： {e}")

# def list_running_models_v1(
#         controller_address: str = Body(None, description="Fastchat controller服务器地址"),
#         placeholder: str = Body(None, description="该参数未使用，占位用"),
# ) -> BaseResponse:
#     '''
#     从fastchat controller获取已加载模型列表及其配置项
#     '''
#     try:
#         host = LLM_SERVER["host"]
#         port = LLM_SERVER["port"]
#         controller_address = f"http://{host}:{port}"
#         with get_httpx_client() as client:
#             r = client.post(controller_address + "/v1/models")
#             data = r.json()['data']
#             return BaseResponse(data=data)
#     except Exception as e:
#         logger.error(f'{e.__class__.__name__}: {e}',
#                      exc_info=e if log_verbose else None)
#         return BaseResponse(
#             code=500,
#             data={},
#             msg=f"failed to get available models from controller: {controller_address}。错误信息是： {e}")
