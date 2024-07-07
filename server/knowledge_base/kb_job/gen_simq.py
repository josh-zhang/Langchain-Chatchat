import os
import datetime
import shutil

from tqdm import tqdm

from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.utils import (list_files_from_folder, get_file_path, list_files_from_path,
                                         KnowledgeFile, get_doc_path)
from configs import (EMBEDDING_MODEL, USE_BM25, GEN_SIMQ_JOB_SCRIPT_PATH, BASE_TEMP_DIR, logger)
from server.knowledge_base.kb_job.job_utils import PythonScriptExecutor


def gen_simq_task(kb_owner, knowledge_base_name, kb_info, model_name, url, concurrency):
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d_%H%M%S")
    task_id = f"{knowledge_base_name}_{now_str}"

    logger.info(f"task started {task_id}")

    filepaths = list_files_from_folder(knowledge_base_name)
    doc_path = get_doc_path(knowledge_base_name)
    filepaths = [os.path.join(doc_path, filepath) for filepath in filepaths]

    logger.info("gen qa")

    output_path = os.path.join(BASE_TEMP_DIR, task_id)
    os.makedirs(output_path)

    failed_files = []

    total_count = 0
    for filepath in tqdm(filepaths):
        filename = os.path.basename(filepath)
        if filename.endswith(".xlsx"):
            total_count += 1
            executor = PythonScriptExecutor()
            script_command = f'{GEN_SIMQ_JOB_SCRIPT_PATH} -f "{filepath}" -o "{output_path}"'
            results = executor.execute_script(script_command)
            return_code = results['return_code']
            if return_code != 0:
                failed_files.append(filepath)

    failed_count = len(failed_files)
    if failed_count >= total_count:
        msg = f"{failed_count}个相似问生成任务全部出错"
        logger.error(msg)
        # return BaseResponse(code=500, msg=msg)
        return

    simq_filepath_list = list_files_from_path(output_path)

    if not simq_filepath_list:
        msg = f"没有相似问文件生成"
        logger.error(msg)
        # return BaseResponse(code=500, msg=msg)
        return

    logger.info("create_kb")

    new_kb_name = f"{knowledge_base_name}_simq"
    new_kb_info = f"{kb_info}-相似问"
    new_kb_agent_guide = f"关于{kb_info}的介绍"

    kb = KBServiceFactory.get_service_by_name(new_kb_name)
    if kb is not None:
        status = kb.clear_vs()
        if not status:
            msg = f"创建知识库出错，知识库已存在并且清楚出错"
            logger.error(msg)
            # return BaseResponse(code=500, msg=msg)
            return

        status = kb.drop_kb()
        if not status:
            msg = f"创建知识库出错，知识库已存在并且删除出错"
            logger.error(msg)
            # return BaseResponse(code=500, msg=msg)
            return

    kb = KBServiceFactory.get_service(kb_owner, kb_owner, new_kb_name, new_kb_info, new_kb_agent_guide,
                                      EMBEDDING_MODEL, USE_BM25, "milvus")
    status = kb.create_kb()
    if not status:
        msg = f"创建知识库出错"
        logger.error(msg)
        # return BaseResponse(code=500, msg=msg)
        return

    logger.info("update_faq")

    count = 0
    for simq_filepath in tqdm(simq_filepath_list):
        if simq_filepath.endswith(".xlsx"):
            simq_filepath = os.path.join(output_path, simq_filepath)
            file_name = os.path.basename(simq_filepath)
            new_file_path = get_file_path(knowledge_base_name=new_kb_name, doc_name=file_name)

            if not os.path.isdir(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))

            shutil.move(simq_filepath, new_file_path)

            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=new_kb_name,
                                    document_loader_name="UnstructuredExcelLoader")
            status = kb.update_doc(kb_file, not_refresh_vs_cache=False)

            if status:
                count += 1

    msg = f"已新增知识库 {new_kb_name}"
    logger.info(msg)

    logger.info(f"task ended {task_id}")
    # return BaseResponse(code=200, msg=f"已新增知识库 {new_kb_name}, 其中包含 {count}篇文档生成的问答")
