from server.db.models.knowledge_base_model import KnowledgeBaseModel
from server.db.models.knowledge_file_model import KnowledgeFileModel, FileDocModel, FileAnswerModel, \
    AnswerQuestionModel
from server.db.session import with_session
from server.knowledge_base.utils import KnowledgeFile
from typing import List, Dict


@with_session
def list_file_num_docs_id_by_kb_name_and_file_name(session,
                                                   kb_name: str,
                                                   file_name: str,
                                                   ) -> List[int]:
    '''
    列出某知识库某文件对应的所有Document的id。
    返回形式：[str, ...]
    '''
    doc_ids = session.query(FileDocModel.doc_id).filter_by(kb_name=kb_name, file_name=file_name).all()
    return [int(_id[0]) for _id in doc_ids]


@with_session
def list_file_num_question_id_by_kb_name_and_file_name(session,
                                                       kb_name: str,
                                                       file_name: str,
                                                       ) -> List[int]:
    '''
    列出某知识库某文件对应的所有Document的id。
    返回形式：[str, ...]
    '''
    doc_ids = session.query(AnswerQuestionModel.doc_id).filter_by(kb_name=kb_name, file_name=file_name).all()
    return [int(_id[0]) for _id in doc_ids]


@with_session
def list_file_num_answer_id_by_kb_name_and_file_name(session,
                                                     kb_name: str,
                                                     file_name: str,
                                                     ) -> List[int]:
    '''
    列出某知识库某文件对应的所有Document的id。
    返回形式：[str, ...]
    '''
    doc_ids = session.query(FileAnswerModel.doc_id).filter_by(kb_name=kb_name, file_name=file_name).all()
    return [int(_id[0]) for _id in doc_ids]


@with_session
def list_docs_from_db(session,
                      kb_name: str,
                      file_name: str = None,
                      metadata: Dict = {},
                      ) -> List[Dict]:
    '''
    列出某知识库某文件对应的所有Document。
    返回形式：[{"id": str, "metadata": dict}, ...]
    '''

    if file_name:
        docs = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name),
                                                  FileDocModel.file_name.ilike(file_name))
    else:
        docs = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))

    for k, v in metadata.items():
        docs = docs.filter(FileDocModel.meta_data[k].as_string() == str(v))

    return [{"id": x.doc_id, "metadata": x.metadata} for x in docs.all()]


@with_session
def list_answer_from_db(session,
                        kb_name: str,
                        file_name: str = None,
                        metadata: Dict = {},
                        ) -> List[Dict]:
    '''
    列出某知识库某文件对应的所有Answer。
    返回形式：[{"id": str, "metadata": dict}, ...]
    '''

    if file_name:
        docs = session.query(FileAnswerModel).filter(FileAnswerModel.kb_name.ilike(kb_name),
                                                     FileAnswerModel.file_name.ilike(file_name))
    else:
        docs = session.query(FileAnswerModel).filter(FileAnswerModel.kb_name.ilike(kb_name))

    for k, v in metadata.items():
        docs = docs.filter(FileAnswerModel.meta_data[k].as_string() == str(v))

    return [{"id": x.doc_id, "metadata": x.metadata} for x in docs.all()]


@with_session
def list_question_from_db(session,
                          kb_name: str,
                          file_name: str = None,
                          metadata: Dict = {},
                          ) -> List[Dict]:
    '''
    列出某知识库某文件对应的所有question。
    返回形式：[{"id": str, "metadata": dict}, ...]
    '''
    if file_name:
        docs = session.query(AnswerQuestionModel).filter(AnswerQuestionModel.kb_name.ilike(kb_name),
                                                         AnswerQuestionModel.file_name.ilike(file_name))
    else:
        docs = session.query(AnswerQuestionModel).filter(AnswerQuestionModel.kb_name.ilike(kb_name))

    for k, v in metadata.items():
        docs = docs.filter(FileDocModel.meta_data[k].as_string() == str(v))

    return [{"id": x.doc_id, "metadata": x.metadata} for x in docs.all()]


@with_session
def delete_docs_from_db(session,
                        kb_name: str,
                        file_name: str = None,
                        ):
    '''
    删除某知识库某文件对应的所有Document
    '''
    if file_name:
        query = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name),
                                                   FileDocModel.file_name.ilike(file_name))
    else:
        query = session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(kb_name))

    query.delete()
    session.commit()


@with_session
def delete_answer_from_db(session,
                          kb_name: str,
                          file_name: str = None,
                          ):
    '''
    删除某知识库某文件对应的所有Answer
    '''
    if file_name:
        query = session.query(FileAnswerModel).filter(FileAnswerModel.kb_name.ilike(kb_name),
                                                      FileAnswerModel.file_name.ilike(file_name))
    else:
        query = session.query(FileAnswerModel).filter(FileAnswerModel.kb_name.ilike(kb_name))

    query.delete()
    session.commit()


@with_session
def delete_question_from_db(session,
                            kb_name: str,
                            file_name: str = None,
                            ):
    '''
    删除某知识库某文件对应的所有question
    '''

    if file_name:
        query = session.query(AnswerQuestionModel).filter(AnswerQuestionModel.kb_name.ilike(kb_name),
                                                          AnswerQuestionModel.file_name.ilike(file_name))
    else:
        query = session.query(AnswerQuestionModel).filter(AnswerQuestionModel.kb_name.ilike(kb_name))

    query.delete()
    session.commit()


@with_session
def delete_file_from_db(session, kb_name: str, file_name: str):
    existing_file = (session.query(KnowledgeFileModel)
                     .filter(KnowledgeFileModel.file_name.ilike(file_name), KnowledgeFileModel.kb_name.ilike(kb_name))
                     .first())
    if existing_file:
        session.delete(existing_file)
        session.commit()

        kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(kb_name)).first()
        if kb:
            kb.file_count -= 1
            session.commit()
        return True
    else:
        return False


@with_session
def delete_files_from_db(session, knowledge_base_name: str):
    session.query(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(knowledge_base_name)).delete(
        synchronize_session='fetch')
    session.query(FileDocModel).filter(FileDocModel.kb_name.ilike(knowledge_base_name)).delete(
        synchronize_session='fetch')
    session.query(FileAnswerModel).filter(FileAnswerModel.kb_name.ilike(knowledge_base_name)).delete(
        synchronize_session='fetch')
    session.query(AnswerQuestionModel).filter(AnswerQuestionModel.kb_name.ilike(knowledge_base_name)).delete(
        synchronize_session='fetch')
    session.commit()

    kb = session.query(KnowledgeBaseModel).filter(KnowledgeBaseModel.kb_name.ilike(knowledge_base_name)).first()
    if kb:
        kb.file_count = 0
    session.commit()

    return True


@with_session
def add_docs_to_db(session,
                   kb_name: str,
                   file_name: str,
                   doc_infos: List[Dict]):
    '''
    将某知识库某文件对应的所有Document信息添加到数据库。
    doc_infos形式：[{"id": str, "metadata": dict}, ...]
    '''
    # ! 这里会出现doc_infos为None的情况，需要进一步排查
    if doc_infos is None:
        print("输入的server.db.repository.knowledge_file_repository.add_docs_to_db的doc_infos参数为None")
        return False
    for d in doc_infos:
        obj = FileDocModel(
            kb_name=kb_name,
            file_name=file_name,
            doc_id=d["id"],
            meta_data=d["metadata"],
        )
        session.add(obj)
    session.commit()
    return True


@with_session
def add_answer_to_db(session,
                     kb_name: str,
                     file_name_list,
                     answer_id_list,
                     doc_infos: List[Dict]):
    '''
    将某知识库某文件对应的所有Answer信息添加到数据库。
    doc_infos形式：[{"id": str, "metadata": dict}, ...]
    '''
    # ! 这里会出现doc_infos为None的情况，需要进一步排查
    if doc_infos is None:
        print("输入的server.db.repository.knowledge_file_repository.add_answer_to_db的doc_infos参数为None")
        return False
    for d, answer_id, file_name in zip(doc_infos, answer_id_list, file_name_list):
        obj = FileAnswerModel(
            kb_name=kb_name,
            file_name=file_name,
            answer_id=answer_id,
            doc_id=d["id"],
            meta_data=d["metadata"],
        )
        session.add(obj)
    session.commit()
    return True


@with_session
def add_question_to_db(session,
                       kb_name: str,
                       file_name_list,
                       answer_id_list,
                       question_id_list,
                       doc_infos: List[Dict]):
    '''
    将某知识库某文件对应的所有question信息添加到数据库。
    doc_infos形式：[{"id": str, "metadata": dict}, ...]
    '''
    # ! 这里会出现doc_infos为None的情况，需要进一步排查
    if doc_infos is None:
        print("输入的server.db.repository.knowledge_file_repository.add_question_to_db的doc_infos参数为None")
        return False
    for d, file_name, question_id, answer_id in zip(doc_infos, file_name_list, question_id_list, answer_id_list):
        obj = AnswerQuestionModel(
            kb_name=kb_name,
            file_name=file_name,
            answer_id=answer_id,
            question_id=question_id,
            doc_id=d["id"],
            meta_data=d["metadata"],
        )
        session.add(obj)
    session.commit()
    return True


@with_session
def get_answer_id_by_question_raw_id_from_db(session,
                                             kb_name: str,
                                             raw_id: str,
                                             ):
    question = session.query(AnswerQuestionModel).filter(AnswerQuestionModel.kb_name.ilike(kb_name)).filter_by(
        question_id=raw_id).first()

    if question:
        return question.answer_id
    else:
        return ""


@with_session
def get_answer_doc_id_by_answer_id_from_db(session,
                                           kb_name: str,
                                           raw_id: str,
                                           ):
    answer = session.query(FileAnswerModel).filter(FileAnswerModel.kb_name.ilike(kb_name)).filter_by(
        answer_id=raw_id).first()

    if answer:
        return answer.doc_id
    else:
        return ""


@with_session
def count_files_from_db(session, kb_name: str) -> int:
    return session.query(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(kb_name)).count()


@with_session
def list_files_from_db(session, kb_name):
    files = session.query(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(kb_name)).all()
    docs = [(f.file_name, f.create_time) for f in files]
    docs = sorted(docs, key=lambda element: element[1], reverse=True)
    docs = [doc[0] for doc in docs]
    return docs


@with_session
def list_files_info_from_db(session, kb_name):
    files = session.query(KnowledgeFileModel).filter(KnowledgeFileModel.kb_name.ilike(kb_name)).all()
    docs = [(f.file_name, f.document_loader_name, f.create_time) for f in files]
    docs = sorted(docs, key=lambda element: element[2], reverse=True)
    docs = [(doc[0], doc[1]) for doc in docs]
    return docs


@with_session
def add_file_to_db(session,
                   kb_file: KnowledgeFile,
                   docs_count: int = 0,
                   custom_docs: bool = False,
                   doc_infos: List[str] = [],  # 形式：[{"id": str, "metadata": dict}, ...]
                   ):
    kb = session.query(KnowledgeBaseModel).filter_by(kb_name=kb_file.kb_name).first()
    if kb:
        # 如果已经存在该文件，则更新文件信息与版本号
        existing_file: KnowledgeFileModel = (session.query(KnowledgeFileModel)
                                             .filter(KnowledgeFileModel.kb_name.ilike(kb_file.kb_name),
                                                     KnowledgeFileModel.file_name.ilike(kb_file.filename))
                                             .first())
        mtime = kb_file.get_mtime()
        size = kb_file.get_size()

        if existing_file:
            existing_file.file_mtime = mtime
            existing_file.file_size = size
            existing_file.docs_count = docs_count
            existing_file.custom_docs = custom_docs
            existing_file.file_version += 1
        # 否则，添加新文件
        else:
            new_file = KnowledgeFileModel(
                file_name=kb_file.filename,
                file_ext=kb_file.ext,
                kb_name=kb_file.kb_name,
                document_loader_name=kb_file.document_loader_name,
                text_splitter_name=kb_file.text_splitter_name or "ChineseRecursiveTextSplitter",
                file_mtime=mtime,
                file_size=size,
                docs_count=docs_count,
                custom_docs=custom_docs,
            )
            kb.file_count += 1
            session.add(new_file)
            session.commit()
        status = add_docs_to_db(kb_name=kb_file.kb_name, file_name=kb_file.filename, doc_infos=doc_infos)
    else:
        status = False
    return status


@with_session
def file_exists_in_db(session, kb_file: KnowledgeFile):
    existing_file = (session.query(KnowledgeFileModel)
                     .filter(KnowledgeFileModel.file_name.ilike(kb_file.filename),
                             KnowledgeFileModel.kb_name.ilike(kb_file.kb_name))
                     .first())
    return True if existing_file else False


@with_session
def get_file_detail(session, kb_name: str, filename: str) -> dict:
    file: KnowledgeFileModel = (session.query(KnowledgeFileModel)
                                .filter(KnowledgeFileModel.file_name.ilike(filename),
                                        KnowledgeFileModel.kb_name.ilike(kb_name))
                                .first())
    if file:
        return {
            "kb_name": file.kb_name,
            "file_name": file.file_name,
            "file_ext": file.file_ext,
            "file_version": file.file_version,
            "document_loader": file.document_loader_name,
            "text_splitter": file.text_splitter_name,
            "create_time": file.create_time,
            "file_mtime": file.file_mtime,
            "file_size": file.file_size,
            "custom_docs": file.custom_docs,
            "docs_count": file.docs_count,
        }
    else:
        return {}
