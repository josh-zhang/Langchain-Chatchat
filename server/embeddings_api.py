from typing import Dict, List

import numpy as np
from langchain.docstore.document import Document
from fastapi import Body

from configs import EMBEDDING_MODEL, logger
from server.utils import BaseResponse


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
        from server.utils import load_local_embeddings

        embeddings = load_local_embeddings(model=embed_model)
        return BaseResponse(data=embeddings.embed_documents(texts))
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")


def embed_texts_endpoint(
        texts: List[str] = Body(..., description="要嵌入的文本列表", examples=[["hello", "world"]]),
        embed_model: str = Body(EMBEDDING_MODEL, description=f"使用的嵌入模型，除了本地部署的Embedding模型。"),
        to_query: bool = Body(False, description="向量是否用于查询。有些模型如Minimax对存储/查询的向量进行了区分优化。"),
) -> BaseResponse:
    '''
    对文本进行向量化，返回 BaseResponse(data=List[List[float]])
    '''
    return embed_texts(texts=texts, embed_model=embed_model, to_query=to_query)


def embed_texts_simi_endpoint(
        text_a: str = Body(..., description="要计算的文本一", examples=["hello world"]),
        text_b: str = Body(..., description="要计算的文本二", examples=["hi world"]),
) -> BaseResponse:
    '''
    计算两个文本相似度，返回 BaseResponse(data=List[float]])
    '''
    from server.utils import load_local_embeddings

    embeddings = load_local_embeddings(model="bge-base-zh", normalize_embeddings=True)
    embedding_a = embeddings.embed_documents([text_a])[0]
    embedding_b = embeddings.embed_documents([text_b])[0]
    query_embed_2d_a = np.reshape(embedding_a, (1, -1))
    query_embed_2d_b = np.reshape(embedding_b, (1, -1))

    similarity = (query_embed_2d_a @ query_embed_2d_b.T).tolist()[0]

    return BaseResponse(data=similarity)


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
