import urllib
from server.utils import BaseResponse, ListListResponse
from server.knowledge_base.utils import validate_kb_name, validate_kb_info, validate_kb_owner
from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_base_repository import list_kbs_from_db
from configs import EMBEDDING_MODEL, logger, log_verbose, USE_BM25
from fastapi import Body, Query


def list_kbs(kb_owner: str = Query("")):
    # Get List of Knowledge Base
    return ListListResponse(data=list_kbs_from_db(kb_owner))


def create_kb(kb_owner: str = Body(..., examples=""),
              kb_viewer: str = Body(..., examples=""),
              kb_name: str = Body(..., examples=["samples"]),
              kb_info: str = Body(..., examples=["samples_introduction"]),
              kb_agent_guide: str = Body(..., examples=["samples_introduction_for_agent"]),
              vector_store_type: str = Body("milvus"),
              embed_model: str = Body(EMBEDDING_MODEL),
              search_enhance: bool = Body(USE_BM25),
              ) -> BaseResponse:
    if not validate_kb_name(kb_name):
        return BaseResponse(code=404, msg="知识库编号不能为空，请重新填写知识库编号")

    if not validate_kb_info(kb_info):
        return BaseResponse(code=404, msg="知识库名称不能为空，请重新填写知识库名称")

    if not validate_kb_owner(kb_owner):
        return BaseResponse(code=404, msg="知识库创建者不能为空")

    kb = KBServiceFactory.get_service_by_name(kb_name)
    if kb is not None:
        return BaseResponse(code=404, msg=f"已存在同编号知识库 {kb_name}")

    kb = KBServiceFactory.get_service(kb_owner, kb_viewer, kb_name, kb_info, kb_agent_guide, embed_model,
                                      search_enhance, vector_store_type)
    try:
        kb.create_kb()
    except Exception as e:
        msg = f"创建知识库出错： {e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    msg = f"已新增 {kb_viewer} 私有知识库 {kb_name}" if kb_viewer else f"已新增公开知识库 {kb_name}"

    return BaseResponse(code=200, msg=msg)


def delete_kb(
        operator: str = Body(..., examples=["admin"]),
        knowledge_base_name: str = Body(..., examples=["samples"])
) -> BaseResponse:
    # Delete selected knowledge base
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    if kb.kb_owner != operator:
        return BaseResponse(code=404, msg=f"只有 {knowledge_base_name} 创建者 {kb.kb_owner} 可以操作")

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
