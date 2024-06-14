import urllib
from server.utils import BaseResponse, ListListResponse
from server.knowledge_base.utils import validate_kb_name, validate_kb_info
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_base_repository import list_kbs_from_db
from configs import EMBEDDING_MODEL, logger, log_verbose, USE_BM25
from fastapi import Body, Query


def list_kbs(kb_owner: str = Query("")):
    # Get List of Knowledge Base
    return ListListResponse(data=list_kbs_from_db(kb_owner))


def create_kb(kb_owner: str = Body(..., examples=""),
              knowledge_base_name: str = Body(..., examples=["samples"]),
              kb_info: str = Body(..., examples=["samples_introduction"]),
              kb_agent_guide: str = Body(..., examples=["samples_introduction_for_agent"]),
              vector_store_type: str = Body("faiss"),
              embed_model: str = Body(EMBEDDING_MODEL),
              search_enhance: bool = Body(USE_BM25),
              ) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=404, msg="知识库编号不能为空，请重新填写知识库编号")

    if not validate_kb_info(kb_info):
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is not None:
        return BaseResponse(code=404, msg=f"已存在同名知识库 {knowledge_base_name}")

    kb = KBServiceFactory.get_service(kb_owner, knowledge_base_name, kb_info, kb_agent_guide, vector_store_type,
                                      embed_model, search_enhance)
    try:
        kb.create_kb()
    except Exception as e:
        msg = f"创建知识库出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    msg = f"已新增 {kb_owner} 私有知识库 {knowledge_base_name}" if kb_owner else f"已新增公开知识库 {knowledge_base_name}"

    return BaseResponse(code=200, msg=msg)


def delete_kb(
        knowledge_base_name: str = Body(..., examples=["samples"])
) -> BaseResponse:
    # Delete selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    try:
        status = kb.clear_vs()
        status = kb.drop_kb()
        if status:
            return BaseResponse(code=200, msg=f"成功删除知识库 {knowledge_base_name}")
    except Exception as e:
        msg = f"删除知识库时出现意外： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"删除知识库失败 {knowledge_base_name}")
