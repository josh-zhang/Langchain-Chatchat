import os
import urllib
from typing import List

from fastapi.responses import FileResponse
from fastapi import File, Form, Body, Query, UploadFile
from langchain.docstore.document import Document
from pydantic import Json

from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_file_repository import get_file_detail, list_docs_from_db, list_answer_from_db, \
    list_question_from_db
from server.utils import BaseResponse, ListResponse, run_in_thread_pool
from server.knowledge_base.kb_job.gen_qa import gen_qa_task, JobExecutor, JobFutures, FuturesAtomic
from server.knowledge_base.utils import (validate_kb_name, get_file_path, files2docs_in_thread,
                                         KnowledgeFile, DocumentWithScores, get_doc_path, create_compressed_archive)
from configs import (VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, BM_25_FACTOR, LITELLM_SERVER,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE, logger, log_verbose, BASE_TEMP_DIR)


def search_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
        max_tokens: int = Body(3000, description="最大参考字数"),
        score_threshold: float = Body(SCORE_THRESHOLD,
                                      description="知识库匹配相关度阈值，取值范围在0-1之间，"
                                                  "SCORE越小，相关度越高，"
                                                  "取到1相当于不筛选，建议设置在0.5左右",
                                      ge=0, le=1),
        use_bm25: bool = Body(True, description="是否添加BM25召回"),
        use_rerank: bool = Body(True, description="是否重拍"),
        use_merge: bool = Body(True, description="是否合并临近段落"),
        dense_top_k_factor: float = Body(3.0, description="密集匹配向量数"),
        sparse_top_k_factor: float = Body(1.0, description="稀疏匹配向量数"),
        sparse_factor: float = Body(BM_25_FACTOR, description="稀疏匹配系数"),
        # limit_before_rerank: bool = Body(False, description="是否在重排前限制token总数"),
        # file_name: str = Body("", description="文件名称，支持 sql 通配符"),
        # metadata: dict = Body({}, description="根据 metadata 进行过滤，仅支持一级键"),
) -> List[DocumentWithScores]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return []

    leave_for_prompt = 1000
    leave_for_context_min = 500
    max_tokens_for_context = max_tokens - leave_for_prompt
    max_tokens_for_context = max(leave_for_context_min, max_tokens_for_context)
    max_tokens_for_context_reank = 18000 # for single instance of bge rerank v2

    dense_topk = int(top_k * dense_top_k_factor)
    sparse_topk = int(top_k * sparse_top_k_factor)

    # logger.info(f"top_k {top_k}")
    # logger.info(f"dense_topk {dense_topk}")
    # logger.info(f"sparse_topk {sparse_topk}")
    # logger.info(f"max_tokens {max_tokens}")
    # logger.info(f"score_threshold {score_threshold}")

    # if query:
    docs = kb.search_all(query, dense_topk, sparse_topk, sparse_factor, use_bm25, score_threshold)

    logger.info(f"{len(docs)} docs after search")

    # if limit_before_rerank:
    docs, count_tokens = kb.limit_tokens_rerank(docs, max_tokens_for_context_reank)

    logger.info(f"rerank tokens {count_tokens}, {len(docs)} docs after token filter")

    if use_rerank and len(docs) > top_k:
        _docs = [d.page_content for d in docs]

        rerank_results = list()
        results = kb.do_rerank(_docs, query)
        for i in results:
            idx = i['index']
            value = i['relevance_score']
            doc = docs[idx]
            doc.metadata["relevance_score"] = value
            rerank_results.append(doc)
        docs = rerank_results

    docs, count_tokens = kb.limit_tokens(docs, max_tokens_for_context)

    logger.info(f"llm tokens {count_tokens}, {len(docs)} docs after token filter")

    if use_merge and docs:
        docs = kb.combine_docs(docs)

    logger.info(f"{len(docs)} docs total searched")

    # elif file_name or metadata:
    #     docs = kb.list_docs(file_name=file_name, metadata=metadata)
    # else:
    #     docs = []

    return docs


def list_files(
        knowledge_base_name: str
) -> ListResponse:
    if not validate_kb_name(knowledge_base_name):
        return ListResponse(code=403, msg="Don't attack me", data=[])

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return ListResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
    else:
        all_doc_names = kb.list_files()
        return ListResponse(data=all_doc_names)


# def list_docs(
#         knowledge_base_name: str,
#         file_name: str
# ) -> ListResponse:
#     if not validate_kb_name(knowledge_base_name):
#         return ListResponse(code=403, msg="Don't attack me", data=[])
#
#     knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
#     kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
#     if kb is None:
#         return ListResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}", data=[])
#     else:
#         all_doc_names = kb.list_docs("docs", file_name)
#         all_question_names = kb.list_docs("question", file_name)
#         all_answer_names = kb.list_docs("answer", file_name)
#
#         all_doc_names = [str(all_doc_names), str(all_question_names), str(all_answer_names)]
#
#         return ListResponse(data=all_doc_names)


def count_docs(
        knowledge_base_name: str,
        vector_name: str,
        file_name: str
) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)

    if vector_name == "docs":
        doc_infos = list_docs_from_db(kb_name=knowledge_base_name, file_name=file_name)
    elif vector_name == "question":
        doc_infos = list_question_from_db(kb_name=knowledge_base_name, file_name=file_name)
    elif vector_name == "answer":
        doc_infos = list_answer_from_db(kb_name=knowledge_base_name, file_name=file_name)
    else:
        assert False

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    else:
        count = kb.count_docs(vector_name, file_name)

        return BaseResponse(code=200, data={"vector_count": count, "db_count": len(doc_infos), "file_name": file_name,
                                            "vector_name": vector_name})


def _save_files_in_thread(files: List[UploadFile],
                          knowledge_base_name: str,
                          override: bool):
    """
    通过多线程将上传的文件保存到对应知识库目录内。
    生成器返回保存结果：{"code":200, "msg": "xxx", "data": {"knowledge_base_name":"xxx", "file_name": "xxx"}}
    """

    def save_file(file: UploadFile, knowledge_base_name: str, override: bool) -> dict:
        '''
        保存单个文件。
        '''
        filename = file.filename
        data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

        try:
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            file_content = file.file.read()  # 读取上传文件的内容
            if (os.path.isfile(file_path)
                    and not override
                    and os.path.getsize(file_path) == len(file_content)
            ):
                # TODO: filesize 不同后的处理
                file_status = f"文件 {filename} 已存在。"
                logger.warn(file_status)
                return dict(code=404, msg=file_status, data=data)

            if not os.path.isdir(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            with open(file_path, "wb") as f:
                f.write(file_content)
            return dict(code=200, msg=f"成功上传文件 {filename}", data=data)
        except Exception as e:
            msg = f"{filename} 文件上传失败，报错信息为: {e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return dict(code=500, msg=msg, data=data)

    params = [{"file": file, "knowledge_base_name": knowledge_base_name, "override": override} for file in files]
    for result in run_in_thread_pool(save_file, params=params):
        yield result


def upload_docs(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
        document_loader_name: str = Form(..., description="文件加载类型", examples=["default"]),
        override: bool = Form(False, description="覆盖已有文件"),
        to_vector_store: bool = Form(True, description="上传文件后是否进行向量化"),
        chunk_size: int = Form(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Form(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Form(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        docs: Json = Form({}, description="自定义的docs，需要转为json字符串",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Form(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """
    API接口：上传文件，并/或向量化
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    file_names = list(docs.keys())

    # 先将上传的文件保存到磁盘
    for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
        filename = result["data"]["file_name"]
        if result["code"] != 200:
            failed_files[filename] = result["msg"]

        if filename not in file_names:
            file_names.append(filename)

    # 对保存的文件进行向量化
    if to_vector_store:
        result = update_docs(
            knowledge_base_name=knowledge_base_name,
            document_loader_name=document_loader_name,
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=True,
        )
        failed_files.update(result.data["failed_files"])
        if not not_refresh_vs_cache:
            kb.save_vector_store("docs")
            kb.save_vector_store("question")
            kb.save_vector_store("answer")
            kb.save_vector_store("query")

    return BaseResponse(code=200, msg="文件上传与向量化完成", data={"failed_files": failed_files})


def delete_docs(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        file_names: List[str] = Body(..., examples=[["file_name.md", "test.txt"]]),
        document_loaders: List[str] = Body(...),
        delete_content: bool = Body(False),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    knowledge_base_name = urllib.parse.unquote(knowledge_base_name)
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}
    for file_name, document_loader_name in zip(file_names, document_loaders):
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"未找到文件 {file_name}"

        try:
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name,
                                    document_loader_name=document_loader_name)

            if document_loader_name == "CustomExcelLoader":
                kb.delete_faq(kb_file, delete_content, not_refresh_vs_cache=True)
            else:
                kb.delete_doc(kb_file, delete_content, not_refresh_vs_cache=True)
        except Exception as e:
            msg = f"{file_name} 文件删除失败，错误信息：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store("docs")
        kb.save_vector_store("question")
        kb.save_vector_store("answer")
        kb.save_vector_store("query")

    return BaseResponse(code=200, msg=f"文件删除完成", data={"failed_files": failed_files})


def update_docs(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        document_loader_name: str = Form(..., description="文件加载类型", examples=["default"]),
        file_names: List[str] = Body(..., description="文件名称，支持多文件", examples=[["file_name1", "text.txt"]]),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        override_custom_docs: bool = Body(False, description="是否覆盖之前自定义的docs"),
        docs: Json = Body({}, description="自定义的docs，需要转为json字符串",
                          examples=[{"test.txt": [Document(page_content="custom doc")]}]),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
) -> BaseResponse:
    """
    更新知识库文档
    """

    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    failed_files = {}

    if document_loader_name == "CustomExcelLoader":
        for file_name in file_names:
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name,
                                    document_loader_name="CustomExcelLoader")
            status = kb.update_faq(kb_file, not_refresh_vs_cache=False)
            if not status:
                failed_files[file_name] = f"加载FAQ文件 {kb_file.kb_name}/{kb_file.filename} 时出错"
    else:
        kb_files = []

        # 生成需要加载docs的文件列表
        for file_name in file_names:
            file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
            # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
            if file_detail.get("custom_docs") and not override_custom_docs:
                continue
            if file_name not in docs:
                try:
                    kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name,
                                                  document_loader_name=document_loader_name))
                except Exception as e:
                    msg = f"加载文档 {file_name} 时出错：{e}"
                    logger.error(f'{e.__class__.__name__}: {msg}',
                                 exc_info=e if log_verbose else None)
                    failed_files[file_name] = msg

        # 从文件生成docs，并进行向量化。
        # 这里利用了KnowledgeFile的缓存功能，在多线程中加载Document，然后传给KnowledgeFile
        for status, result in files2docs_in_thread(kb_files,
                                                   chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap,
                                                   zh_title_enhance=zh_title_enhance):
            if status:
                kb_name, file_name, new_docs = result
                kb_file = KnowledgeFile(filename=file_name,
                                        knowledge_base_name=knowledge_base_name,
                                        document_loader_name=document_loader_name)
                kb_file.splited_docs = new_docs
                kb.update_doc(kb_file, not_refresh_vs_cache=True)
            else:
                kb_name, file_name, error = result
                failed_files[file_name] = error

        # 将自定义的docs进行向量化
        for file_name, v in docs.items():
            try:
                v = [x if isinstance(x, Document) else Document(**x) for x in v]
                kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name,
                                        document_loader_name=document_loader_name)
                kb.update_doc(kb_file, docs=v, not_refresh_vs_cache=True)
            except Exception as e:
                msg = f"为 {file_name} 添加自定义docs时出错：{e}"
                logger.error(f'{e.__class__.__name__}: {msg}',
                             exc_info=e if log_verbose else None)
                failed_files[file_name] = msg

    if not not_refresh_vs_cache:
        kb.save_vector_store("docs")
        kb.save_vector_store("question")
        kb.save_vector_store("answer")
        kb.save_vector_store("query")

    return BaseResponse(code=200, msg=f"更新文档完成", data={"failed_files": failed_files})


def download_doc(
        knowledge_base_name: str = Query(..., description="知识库名称", examples=["samples"]),
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
        preview: bool = Query(False, description="是：浏览器内预览；否：下载"),
):
    """
    下载知识库文档
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    if preview:
        content_disposition_type = "inline"
    else:
        content_disposition_type = None

    try:
        filepath = get_file_path(knowledge_base_name, file_name)

        if os.path.exists(filepath):
            return FileResponse(
                path=filepath,
                filename=file_name,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"{file_name} 读取文件失败，错误信息是：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"{file_name} 读取文件失败")


def download_kb_files(
        knowledge_base_name: str = Query(..., description="知识库名称", examples=["samples"]),
):
    """
    下载知识库所有文档
    """
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")

    kb_content_path = get_doc_path(knowledge_base_name)

    temp_path = os.path.join(BASE_TEMP_DIR, knowledge_base_name)

    if os.path.exists(kb_content_path):
        archive_path = create_compressed_archive(kb_content_path, temp_path)
    else:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name} content文件夹")

    if not os.path.exists(archive_path):
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name} 下载文件 {archive_path}")

    try:
        return FileResponse(
            path=archive_path,
            filename=os.path.basename(archive_path),
            media_type="multipart/form-data",
        )
    except Exception as e:
        msg = f"{archive_path} 读取文件失败，错误信息是：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)


def gen_qa_for_kb(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        model_name: str = Body(...),
):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if not kb.exists():
        return BaseResponse(code=404, msg=f"未找到知识库 ‘{knowledge_base_name}’")
    else:
        kb_info = kb.kb_info

        FuturesAtomic.acquire()
        future = JobFutures.get(knowledge_base_name)

        if future is None or future.done():
            new_future = JobExecutor.submit(gen_qa_task, knowledge_base_name, kb_info, model_name, LITELLM_SERVER, 3)
            JobFutures[knowledge_base_name] = new_future
            FuturesAtomic.release()
            return BaseResponse(code=200, msg=f"使用{model_name}的文档问答生成任务提交成功")
        else:
            FuturesAtomic.release()
            return BaseResponse(code=404, msg=f"上次任务仍在运行中，请等待任务完成后再提交新任务")


def get_gen_qa_result(
        knowledge_base_name: str = Body(..., examples=["samples"]),
):
    FuturesAtomic.acquire()
    future = JobFutures.get(knowledge_base_name)

    if future is None:
        FuturesAtomic.release()
        return BaseResponse(code=404, msg=f"无效的任务ID")
    elif future.done():
        result = future.result()
        FuturesAtomic.release()
        return BaseResponse(code=200, msg=f"任务已结束", json={"task_result": result})
    else:
        FuturesAtomic.release()
        return BaseResponse(code=202, msg=f"任务正在运行中")

# def files2docs(files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
#                 knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
#                 override: bool = Form(False, description="覆盖已有文件"),
#                 save: bool = Form(True, description="是否将文件保存到知识库目录")):
#     def save_files(files, knowledge_base_name, override):
#         for result in _save_files_in_thread(files, knowledge_base_name=knowledge_base_name, override=override):
#             yield json.dumps(result, ensure_ascii=False)

#     def files_to_docs(files):
#         for result in files2docs_in_thread(files):
#             yield json.dumps(result, ensure_ascii=False)


# def update_docs_by_id(
#         knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
#         docs: Dict[str, Document] = Body(..., description="要更新的文档内容，形如：{id: Document, ...}")
# ) -> BaseResponse:
#     '''
#     按照文档 ID 更新文档内容
#     '''
#     kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
#     if kb is None:
#         return BaseResponse(code=500, msg=f"指定的知识库 {knowledge_base_name} 不存在")
#     if kb.update_doc_by_ids(docs=docs):
#         return BaseResponse(msg=f"文档更新成功")
#     else:
#         return BaseResponse(msg=f"文档更新失败")
# def update_info(
#         knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
#         kb_info: str = Body(..., description="知识库介绍", examples=["这是一个知识库"]),
# ):
#     if not validate_kb_name(knowledge_base_name):
#         return BaseResponse(code=403, msg="Don't attack me")
#
#     kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
#     if kb is None:
#         return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
#     kb.update_info(kb_info)
#
#     return BaseResponse(code=200, msg=f"知识库介绍修改完成", data={"kb_info": kb_info})
# def update_agent_guide(
#         knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
#         kb_agent_guide: str = Body(..., description="知识库Agent介绍", examples=["这是一个知识库"]),
# ):
#     if not validate_kb_name(knowledge_base_name):
#         return BaseResponse(code=403, msg="Don't attack me")
#
#     kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
#     if kb is None:
#         return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
#     kb.update_agent_guide(kb_agent_guide)
#
#     return BaseResponse(code=200, msg=f"知识库介绍修改完成", data={"kb_agent_guide": kb_agent_guide})


# def recreate_vector_store(
#         knowledge_base_name: str = Body(..., examples=["samples"]),
#         kb_info: str = Body(..., examples=["samples_introduction"]),
#         kb_agent_guide: str = Body(..., examples=["samples_introduction_for_agent"]),
#         allow_empty_kb: bool = Body(True),
#         vs_type: str = Body(DEFAULT_VS_TYPE),
#         embed_model: str = Body(EMBEDDING_MODEL),
#         chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
#         chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
#         zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
#         not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
#         search_enhance: bool = Body(SEARCH_ENHANCE),
# ):
#     """
#     recreate vector store from the content.
#     this is usefull when user can copy files to content folder directly instead of upload through network.
#     by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
#     set allow_empty_kb to True make it applied on empty knowledge base which it not in the info.db or having no documents.
#     """
#
#     def output():
#         kb = KBServiceFactory.get_service(knowledge_base_name, kb_info, kb_agent_guide, vs_type, embed_model,
#                                           search_enhance)
#         if not kb.exists() and not allow_empty_kb:
#             yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
#         else:
#             if kb.exists():
#                 kb.clear_vs()
#             kb.create_kb()
#             files = list_files_from_folder(knowledge_base_name)
#             kb_files = [(file, knowledge_base_name) for file in files]
#             i = 0
#             for status, result in files2docs_in_thread(kb_files,
#                                                        chunk_size=chunk_size,
#                                                        chunk_overlap=chunk_overlap,
#                                                        zh_title_enhance=zh_title_enhance):
#                 if status:
#                     kb_name, file_name, docs = result
#                     kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
#                     kb_file.splited_docs = docs
#                     yield json.dumps({
#                         "code": 200,
#                         "msg": f"({i + 1} / {len(files)}): {file_name}",
#                         "total": len(files),
#                         "finished": i + 1,
#                         "doc": file_name,
#                     }, ensure_ascii=False)
#                     kb.add_doc(kb_file, not_refresh_vs_cache=True)
#                 else:
#                     kb_name, file_name, error = result
#                     msg = f"添加文件‘{file_name}’到知识库‘{knowledge_base_name}’时出错：{error}。已跳过。"
#                     logger.error(msg)
#                     yield json.dumps({
#                         "code": 500,
#                         "msg": msg,
#                     })
#                 i += 1
#             if not not_refresh_vs_cache:
#                 kb.save_vector_store("docs")
#                 kb.save_vector_store("question")
#                 kb.save_vector_store("answer")
#                 kb.save_vector_store("query")
#
#     return EventSourceResponse(output())
