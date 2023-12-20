from server.db.models.knowledge_base_model import KnowledgeBaseModel
from server.db.session import with_session


@with_session
def add_kb_to_db(session, kb_name, kb_info, kb_summary, vs_type, embed_model, search_enhance):
    # 创建知识库实例
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_name).first()
    if not kb:
        kb = KnowledgeBaseModel(kb_name=kb_name, kb_info=kb_info, kb_summary=kb_summary, vs_type=vs_type,
                                embed_model=embed_model, search_enhance=search_enhance)
        session.add(kb)
    else:  # update kb with new vs_type and embed_model
        kb.kb_info = kb_info
        kb.kb_summary = kb_summary
        kb.vs_type = vs_type
        kb.embed_model = embed_model
        kb.search_enhance = search_enhance
    return True


@with_session
def list_kbs_from_db(session, min_file_count: int = -1):
    kbs = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.file_count > min_file_count).all()
    kbs = [[kb.kb_name, kb.info] for kb in kbs]
    return kbs


@with_session
def kb_exists(session, kb_name):
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_name).first()
    status = True if kb else False
    return status


@with_session
def load_kb_from_db(session, kb_name):
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_name).first()
    if kb:
        kb_name, kb_info, kb_summary, vs_type, embed_model, search_enhance = kb.kb_name, kb.kb_info, kb.kb_summary, kb.vs_type, kb.embed_model, kb.search_enhance
    else:
        kb_name, kb_info, kb_summary, vs_type, embed_model, search_enhance = None, None, None, None, None, None
    return kb_name, kb_info, kb_summary, vs_type, embed_model, search_enhance


@with_session
def delete_kb_from_db(session, kb_name):
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_name).first()
    if kb:
        session.delete(kb)
    return True


@with_session
def get_kb_detail(session, kb_name: str) -> dict:
    kb: KnowledgeBaseModel = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_name).first()
    if kb:
        return {
            "kb_name": kb.kb_name,
            "kb_info": kb.kb_info,
            "kb_summary": kb.kb_summary,
            "vs_type": kb.vs_type,
            "embed_model": kb.embed_model,
            "file_count": kb.file_count,
            "create_time": kb.create_time,
        }
    else:
        return {}
