import os
import urllib
import json
import datetime
import shutil
import threading
from typing import List

from fastapi.responses import FileResponse
from fastapi import File, Form, Body, Query, UploadFile
from langchain.docstore.document import Document
from sse_starlette import EventSourceResponse
from pydantic import Json

from server.knowledge_base.kb_service.base import KBServiceFactory
from server.db.repository.knowledge_file_repository import get_file_detail
from server.utils import BaseResponse, ListResponse, run_in_thread_pool
from server.knowledge_base.utils import (validate_kb_name, list_files_from_folder, get_file_path, list_files_from_path,
                                         files2docs_in_thread, KnowledgeFile, DocumentWithScores, PythonScriptExecutor)
from configs import (DEFAULT_VS_TYPE, EMBEDDING_MODEL, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, BM_25_FACTOR,
                     CHUNK_SIZE, OVERLAP_SIZE, ZH_TITLE_ENHANCE, SEARCH_ENHANCE, logger, log_verbose,
                     QA_JOB_SCRIPT_PATH, BASE_TEMP_DIR)


def get_total_score_sorted(docs_data: List[DocumentWithScores], score_threshold) -> List[DocumentWithScores]:
    for ds in docs_data:
        sbert_doc = ds.scores.get("sbert_doc", 0.0)
        sbert_que = ds.scores.get("sbert_que", 0.0)
        sbert_ans = ds.scores.get("sbert_ans", 0.0)

        bm_doc = ds.scores.get("bm_doc", 0.0)
        bm_que = ds.scores.get("bm_que", 0.0)
        bm_ans = ds.scores.get("bm_ans", 0.0)

        doc = sbert_doc + bm_doc
        que = sbert_que + bm_que
        ans = sbert_ans + bm_ans
        qa = max(que, ans)

        ds.scores["total"] = doc + qa

    return sorted([ds for ds in docs_data if ds.scores["total"] >= score_threshold], key=lambda x: x.scores["total"],
                  reverse=True)


def search_docs(
        query: str = Body("", description="用户输入", examples=["你好"]),
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        top_k: int = Body(VECTOR_SEARCH_TOP_K, description="匹配向量数"),
        score_threshold: float = Body(SCORE_THRESHOLD,
                                      description="知识库匹配相关度阈值，取值范围在0-1之间，"
                                                  "SCORE越小，相关度越高，"
                                                  "取到1相当于不筛选，建议设置在0.5左右",
                                      ge=0, le=1),
) -> List[DocumentWithScores]:
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return []

    ks_docs_data, ks_qa_data = kb.search_allinone(query, top_k * 2, 0.0)

    if kb.search_enhance:
        bm25_docs_data, bm25_qa_data = kb.enhance_search_allinone(query, 2, BM_25_FACTOR)
        docs_data = kb.merge_docs(ks_docs_data, bm25_docs_data, is_max=True)
        qa_data = kb.merge_answers(ks_qa_data, bm25_qa_data, is_max=True)
    else:
        docs_data = ks_docs_data
        qa_data = ks_qa_data

    # print(f"final docs_data {docs_data}")
    # print(f"final qa_data {qa_data}")

    docs_data = docs_data + qa_data

    docs_data = get_total_score_sorted(docs_data, score_threshold)

    print(f"top_k {top_k} and {len(docs_data)} docs total searched ")
    print(docs_data)

    docs_data = docs_data[:top_k]

    return docs_data


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
        try:
            filename = file.filename
            file_path = get_file_path(knowledge_base_name=knowledge_base_name, doc_name=filename)
            data = {"knowledge_base_name": knowledge_base_name, "file_name": filename}

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


# TODO: 等langchain.document_loaders支持内存文件的时候再开通
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


def upload_docs(
        files: List[UploadFile] = File(..., description="上传文件，支持多文件"),
        knowledge_base_name: str = Form(..., description="知识库名称", examples=["samples"]),
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
            file_names=file_names,
            override_custom_docs=True,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            zh_title_enhance=zh_title_enhance,
            docs=docs,
            not_refresh_vs_cache=not_refresh_vs_cache,
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
    for file_name in file_names:
        if not kb.exist_doc(file_name):
            failed_files[file_name] = f"未找到文件 {file_name}"

        try:
            kb_file = KnowledgeFile(filename=file_name,
                                    knowledge_base_name=knowledge_base_name)
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


def update_info(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
        kb_info: str = Body(..., description="知识库介绍", examples=["这是一个知识库"]),
):
    if not validate_kb_name(knowledge_base_name):
        return BaseResponse(code=403, msg="Don't attack me")

    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if kb is None:
        return BaseResponse(code=404, msg=f"未找到知识库 {knowledge_base_name}")
    kb.update_info(kb_info)

    return BaseResponse(code=200, msg=f"知识库介绍修改完成", data={"kb_info": kb_info})


def update_docs(
        knowledge_base_name: str = Body(..., description="知识库名称", examples=["samples"]),
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
    kb_files = []

    # 生成需要加载docs的文件列表
    for file_name in file_names:
        file_detail = get_file_detail(kb_name=knowledge_base_name, filename=file_name)
        # 如果该文件之前使用了自定义docs，则根据参数决定略过或覆盖
        if file_detail.get("custom_docs") and not override_custom_docs:
            continue
        if file_name not in docs:
            if file_name.startswith("gen_") and file_name.endswith(".xlsx"):
                kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
                kb.update_faq(kb_file, True, not_refresh_vs_cache=False)
            elif file_name.startswith("faq_") and file_name.endswith(".xlsx"):
                kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
                kb.update_faq(kb_file, False, not_refresh_vs_cache=False)
            else:
                try:
                    kb_files.append(KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name))
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
                                    knowledge_base_name=knowledge_base_name)
            kb_file.splited_docs = new_docs
            kb.update_doc(kb_file, not_refresh_vs_cache=not_refresh_vs_cache)
        else:
            kb_name, file_name, error = result
            failed_files[file_name] = error

    # 将自定义的docs进行向量化
    for file_name, v in docs.items():
        try:
            v = [x if isinstance(x, Document) else Document(**x) for x in v]
            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)
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
        kb_file = KnowledgeFile(filename=file_name,
                                knowledge_base_name=knowledge_base_name)

        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"{file_name} 读取文件失败，错误信息是：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"{file_name} 读取文件失败")


def download_faq(
        knowledge_base_name: str = Query(..., description="知识库名称", examples=["samples"]),
        file_name: str = Query(..., description="文件名称", examples=["test.txt"]),
        preview: bool = Query(False, description="是：浏览器内预览；否：下载"),
        faq_prefix="faq_"
):
    """
    下载知识库文档对应FAQ
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

    file_name = f"{faq_prefix}{file_name}"

    try:
        kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=knowledge_base_name)

        if os.path.exists(kb_file.filepath):
            return FileResponse(
                path=kb_file.filepath,
                filename=kb_file.filename,
                media_type="multipart/form-data",
                content_disposition_type=content_disposition_type,
            )
    except Exception as e:
        msg = f"{file_name} 读取文件失败，错误信息是：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        return BaseResponse(code=500, msg=msg)

    return BaseResponse(code=500, msg=f"{file_name} 读取文件失败")


def recreate_vector_store(
        knowledge_base_name: str = Body(..., examples=["samples"]),
        kb_info: str = Body(..., examples=["samples_introduction"]),
        allow_empty_kb: bool = Body(True),
        vs_type: str = Body(DEFAULT_VS_TYPE),
        embed_model: str = Body(EMBEDDING_MODEL),
        chunk_size: int = Body(CHUNK_SIZE, description="知识库中单段文本最大长度"),
        chunk_overlap: int = Body(OVERLAP_SIZE, description="知识库中相邻文本重合长度"),
        zh_title_enhance: bool = Body(ZH_TITLE_ENHANCE, description="是否开启中文标题加强"),
        not_refresh_vs_cache: bool = Body(False, description="暂不保存向量库（用于FAISS）"),
        search_enhance: bool = Body(SEARCH_ENHANCE),
):
    """
    recreate vector store from the content.
    this is usefull when user can copy files to content folder directly instead of upload through network.
    by default, get_service_by_name only return knowledge base in the info.db and having document files in it.
    set allow_empty_kb to True make it applied on empty knowledge base which it not in the info.db or having no documents.
    """

    def output():
        kb = KBServiceFactory.get_service(knowledge_base_name, kb_info, vs_type, embed_model, search_enhance)
        if not kb.exists() and not allow_empty_kb:
            yield {"code": 404, "msg": f"未找到知识库 ‘{knowledge_base_name}’"}
        else:
            if kb.exists():
                kb.clear_vs()
            kb.create_kb()
            files = list_files_from_folder(knowledge_base_name)
            kb_files = [(file, knowledge_base_name) for file in files]
            i = 0
            for status, result in files2docs_in_thread(kb_files,
                                                       chunk_size=chunk_size,
                                                       chunk_overlap=chunk_overlap,
                                                       zh_title_enhance=zh_title_enhance):
                if status:
                    kb_name, file_name, docs = result
                    kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=kb_name)
                    kb_file.splited_docs = docs
                    yield json.dumps({
                        "code": 200,
                        "msg": f"({i + 1} / {len(files)}): {file_name}",
                        "total": len(files),
                        "finished": i + 1,
                        "doc": file_name,
                    }, ensure_ascii=False)
                    kb.add_doc(kb_file, not_refresh_vs_cache=True)
                else:
                    kb_name, file_name, error = result
                    msg = f"添加文件‘{file_name}’到知识库‘{knowledge_base_name}’时出错：{error}。已跳过。"
                    logger.error(msg)
                    yield json.dumps({
                        "code": 500,
                        "msg": msg,
                    })
                i += 1
            if not not_refresh_vs_cache:
                kb.save_vector_store("docs")
                kb.save_vector_store("question")
                kb.save_vector_store("answer")
                kb.save_vector_store("query")

    return EventSourceResponse(output())


def gen_qa_for_kb_job(knowledge_base_name, kb_info):
    filepaths = list_files_from_folder(knowledge_base_name)

    now = datetime.datetime.now()
    timestamp = f"{knowledge_base_name}_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}"

    output_path = os.path.join(BASE_TEMP_DIR, timestamp)

    failed_files = []

    total_count = 0
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        if filename.endswith(".html"):
            total_count += 1
            title = filename[:-5]
            executor = PythonScriptExecutor()
            script_command = f"{QA_JOB_SCRIPT_PATH} -f {filepath} -t {title} -o {output_path}"
            results = executor.execute_script(script_command)
            return_code = results['return_code']
            if return_code != 0:
                failed_files.append(filepath)

    failed_count = len(failed_files)
    if failed_count >= total_count:
        # msg = f"{failed_count}个问答文件生成任务全部出错"
        # return BaseResponse(code=500, msg=msg)
        return

    qa_filepath_list = list_files_from_path(output_path)

    new_kb_name = f"faq_{knowledge_base_name}"
    new_kb_info = f"{kb_info} 生成问答"

    kb = KBServiceFactory.get_service_by_name(new_kb_name)
    if kb is not None:
        status = kb.clear_vs()
        if not status:
            # msg = f"创建知识库出错，知识库已存在并且清楚出错"
            # return BaseResponse(code=500, msg=msg)
            return

        status = kb.drop_kb()
        if not status:
            # msg = f"创建知识库出错，知识库已存在并且删除出错"
            # return BaseResponse(code=500, msg=msg)
            return

    kb = KBServiceFactory.get_service(new_kb_name, new_kb_info, "faiss", EMBEDDING_MODEL, SEARCH_ENHANCE)
    status = kb.create_kb()
    if not status:
        # msg = f"创建知识库出错"
        # return BaseResponse(code=500, msg=msg)
        return

    count = 0
    for qa_filepath in qa_filepath_list:
        if qa_filepath.endswith(".xlsx"):
            qa_filepath = os.path.join(output_path, qa_filepath)
            file_name = os.path.basename(qa_filepath)
            new_file_path = get_file_path(knowledge_base_name=new_kb_name, doc_name=file_name)

            if not os.path.isdir(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))

            shutil.move(qa_filepath, new_file_path)

            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=new_kb_name)
            status = kb.update_faq(kb_file, True, not_refresh_vs_cache=False)

            if status:
                count += 1

    # return BaseResponse(code=200, msg=f"已新增知识库 {new_kb_name}, 其中包含 {count}篇文档生成的问答")


def gen_qa_for_kb(
        knowledge_base_name: str = Body(..., examples=["samples"]),
):
    kb = KBServiceFactory.get_service_by_name(knowledge_base_name)
    if not kb.exists():
        return BaseResponse(code=404, msg=f"未找到知识库 ‘{knowledge_base_name}’")
    else:
        kb_info = kb.kb_info

        threading.Thread(target=gen_qa_for_kb_job, args=(knowledge_base_name, kb_info)).start()

        return BaseResponse(code=200, msg=f"文档问答生成任务提交成功")
