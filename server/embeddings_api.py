from typing import Dict, List

import requests
from langchain.docstore.document import Document
from fastapi import Body

from configs import EMBEDDING_MODEL, logger, log_verbose, MODEL_PATH, LITELLM_SERVER
from server.utils import BaseResponse


def list_embed_models(
        supervisor_address: str = Body(LITELLM_SERVER, description="litellm服务器地址"),
        placeholder: str = Body(None, description="该参数未使用，占位用"),
) -> BaseResponse:
    '''
    从fastchat controller获取已加载模型列表及其配置项
    '''
    local_embed_models = list(MODEL_PATH["embed_model"])
    supervisor_address = supervisor_address or LITELLM_SERVER
    supervisor_address = f"{supervisor_address}/model/info" if supervisor_address.startswith(
        "http") else f"http://{supervisor_address}/model/info"

    try:
        # from xinference_client import RESTfulClient
        # client = RESTfulClient(supervisor_address)
        # models = client.list_models()
        # models = [k for k, v in models.items() if v["model_type"] == 'embedding']
        res = requests.get(supervisor_address, headers={"Content-Type": "application/json"})
        res = res.json()['data']
        model_list = [i['model_name'] for i in res if
                      'mode' in i['model_info'] and i['model_info']['mode'] == 'embedding']
        model_list = [f"{i}-api" for i in model_list]

        model_list += local_embed_models

        return BaseResponse(data=model_list)
    except Exception as e:
        logger.error(f'{e.__class__.__name__}: {e}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(data=local_embed_models)


def embed_texts(
        texts: List[str],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> BaseResponse:
    '''
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    TODO: 也许需要加入缓存机制，减少 token 消耗
    '''
    try:
        if embed_model.endswith("-api"):
            supervisor_address = LITELLM_SERVER
            supervisor_address = f"{supervisor_address}/model/info" if supervisor_address.startswith(
                "http") else f"http://{supervisor_address}/model/info"

            response = requests.post(f"{supervisor_address}/v1/embeddings",
                                     json={"model": embed_model[:-4], "input": texts})
            data = [i["embedding"] for i in response.json()["data"]]
            return BaseResponse(data=data)
        else:
            from server.utils import load_local_embeddings

            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=embeddings.embed_documents(texts))
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")


async def aembed_texts(
        texts: List[str],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> BaseResponse:
    '''
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    '''
    try:
        if embed_model.endswith("-api"):
            supervisor_address = LITELLM_SERVER
            supervisor_address = f"{supervisor_address}/model/info" if supervisor_address.startswith(
                "http") else f"http://{supervisor_address}/model/info"

            response = requests.post(f"{supervisor_address}/v1/embeddings",
                                     json={"model": embed_model[:-4], "input": texts})
            data = [i["embedding"] for i in response.json()["data"]]
            return BaseResponse(data=data)
        else:
            from server.utils import load_local_embeddings

            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=await embeddings.aembed_documents(texts))
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")


# def embed_texts_endpoint(
#         texts: List[str] = Body(..., description="要嵌入的文本列表", examples=[["hello", "world"]]),
#         embed_model: str = Body(EMBEDDING_MODEL, description=f"使用的嵌入模型，除了本地部署的Embedding模型。"),
#         to_query: bool = Body(False, description="向量是否用于查询。有些模型如Minimax对存储/查询的向量进行了区分优化。"),
# ) -> BaseResponse:
#     '''
#     对文本进行向量化，返回 BaseResponse(data=List[List[float]])
#     '''
#     return embed_texts(texts=texts, embed_model=embed_model, to_query=to_query)


def embed_documents(
        docs: List[Document],
        embed_model: str = EMBEDDING_MODEL,
        to_query: bool = False,
) -> Dict:
    """
    将 List[Document] 向量化，转化为 VectorStore.add_embeddings 可以接受的参数
    """
    texts = [x.page_content for x in docs]
    metadatas = [x.metadata for x in docs]
    embeddings = embed_texts(texts=texts, embed_model=embed_model, to_query=to_query).data
    if embeddings is not None:
        return {
            "texts": texts,
            "embeddings": embeddings,
            "metadatas": metadatas,
        }
