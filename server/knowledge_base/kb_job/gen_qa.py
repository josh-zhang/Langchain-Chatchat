import os
import time
import datetime
import shutil
import threading
import subprocess

from tqdm import tqdm

from server.knowledge_base.kb_service.base import KBServiceFactory
from server.knowledge_base.utils import (list_files_from_folder, get_file_path, list_files_from_path,
                                         KnowledgeFile, get_doc_path)
from configs import (EMBEDDING_MODEL, SEARCH_ENHANCE, QA_JOB_SCRIPT_PATH, BASE_TEMP_DIR, logger)

import concurrent.futures


class PythonScriptExecutor:
    def __init__(self):
        # Constructor for any initialization if needed
        pass

    def execute_script(self, script_path):
        """
        Executes a Python script and captures its output, error, and execution time.
        """
        start_time = time.time()

        # Execute the script using subprocess
        with subprocess.Popen(f"python3.10 {script_path}", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              shell=True) as process:
            for line in process.stdout:
                logger.info(line.strip())

        end_time = time.time()
        duration = end_time - start_time

        # Logging the outcome
        if process.returncode == 0:
            logger.info(f"Script {script_path} executed successfully in {duration:.2f} seconds.")
        else:
            logger.error(f"Script {script_path} failed")

        # Returning the results in a structured format
        return {
            'return_code': process.returncode,
            'execution_time': duration
        }


def gen_qa_task(knowledge_base_name, kb_info, model_name, url, concurrency):
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
        if filename.endswith(".html"):
            total_count += 1
            title = filename[:-5]
            executor = PythonScriptExecutor()
            script_command = f'{QA_JOB_SCRIPT_PATH} -f "{filepath}" -t "{title}" -o "{output_path}"'
            results = executor.execute_script(script_command)
            return_code = results['return_code']
            if return_code != 0:
                failed_files.append(filepath)

    failed_count = len(failed_files)
    if failed_count >= total_count:
        msg = f"{failed_count}个问答文件生成任务全部出错"
        logger.error(msg)
        # return BaseResponse(code=500, msg=msg)
        return

    qa_filepath_list = list_files_from_path(output_path)

    if not qa_filepath_list:
        msg = f"没有任何问答文件生成"
        logger.error(msg)
        # return BaseResponse(code=500, msg=msg)
        return

    logger.info("create_kb")

    new_kb_name = f"{knowledge_base_name}_faq"
    new_kb_info = f"{kb_info}-生成问答"
    new_kb_agent_guide = f"关于{kb_info}的FAQ"

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

    kb = KBServiceFactory.get_service(new_kb_name, new_kb_info, new_kb_agent_guide, "faiss", EMBEDDING_MODEL,
                                      SEARCH_ENHANCE)
    status = kb.create_kb()
    if not status:
        msg = f"创建知识库出错"
        logger.error(msg)
        # return BaseResponse(code=500, msg=msg)
        return

    logger.info("update_faq")

    count = 0
    for qa_filepath in tqdm(qa_filepath_list):
        if qa_filepath.endswith(".xlsx"):
            qa_filepath = os.path.join(output_path, qa_filepath)
            file_name = os.path.basename(qa_filepath)
            new_file_path = get_file_path(knowledge_base_name=new_kb_name, doc_name=file_name)

            if not os.path.isdir(os.path.dirname(new_file_path)):
                os.makedirs(os.path.dirname(new_file_path))

            shutil.move(qa_filepath, new_file_path)

            kb_file = KnowledgeFile(filename=file_name, knowledge_base_name=new_kb_name,
                                    document_loader_name="CustomExcelLoader")
            status = kb.update_faq(kb_file, not_refresh_vs_cache=False)

            if status:
                count += 1

    msg = f"已新增知识库 {new_kb_name}"
    logger.info(msg)

    logger.info(f"task ended {task_id}")
    # return BaseResponse(code=200, msg=f"已新增知识库 {new_kb_name}, 其中包含 {count}篇文档生成的问答")


JobExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
FuturesAtomic = threading.Lock()
JobFutures = {}
