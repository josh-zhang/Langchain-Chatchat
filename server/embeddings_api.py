from langchain.docstore.document import Document
from configs import EMBEDDING_MODEL, logger
from server.utils import BaseResponse, list_embed_models
from fastapi import Body
from typing import Dict, List
from server.utils import load_local_embeddings


def embed_texts(
        texts: List[str],
        embed_model: str = EMBEDDING_MODEL,
) -> BaseResponse:
    '''
    对文本进行向量化。返回数据格式：BaseResponse(data=List[List[float]])
    TODO: 也许需要加入缓存机制，减少 token 消耗
    '''
    try:
        if embed_model in list_embed_models():
            embeddings = load_local_embeddings(model=embed_model)
            return BaseResponse(data=embeddings.embed_documents(texts))
        return BaseResponse(code=500, msg=f"指定的模型 {embed_model} 不支持 Embeddings 功能。")
    except Exception as e:
        logger.error(e)
        return BaseResponse(code=500, msg=f"文本向量化过程中出现错误：{e}")


def embed_texts_endpoint(
        texts: List[str] = Body(..., description="要嵌入的文本列表", examples=[["hello", "world"]]),
        embed_model: str = Body(EMBEDDING_MODEL, description=f"使用的嵌入模型，除了本地部署的Embedding模型"),
) -> BaseResponse:
    '''
    对文本进行向量化，返回 BaseResponse(data=List[List[float]])
    '''
    return embed_texts(texts=texts, embed_model=embed_model)


def embed_documents(
        docs: List[Document],
        embed_model: str = EMBEDDING_MODEL,
) -> Dict:
    """
    将 List[Document] 向量化，转化为 VectorStore.add_embeddings 可以接受的参数
    """
    texts = [x.page_content for x in docs]
    print(f"sample text[0] {texts[0]}")

    metadatas = [x.metadata for x in docs]
    embedding_model = load_local_embeddings(model=embed_model)
    embeddings = embedding_model.embed_documents(texts)

    return {
        "texts": texts,
        "embeddings": embeddings,
        "metadatas": metadatas,
    }
