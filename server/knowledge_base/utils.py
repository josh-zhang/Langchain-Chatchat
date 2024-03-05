import os
import json
import importlib
import shutil
from typing import List, Union, Dict, Tuple, Generator
from pathlib import Path

import chardet
import langchain_community.document_loaders
from unstructured.documents.elements import Element, Text, ElementMetadata, Title
from unstructured.partition.common import get_last_modified_date
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter, MarkdownHeaderTextSplitter
from transformers import AutoTokenizer, GPT2TokenizerFast

from configs import (
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    logger,
    log_verbose,
    text_splitter_dict,
    LLM_MODELS,
    TEXT_SPLITTER_NAME,
)
from text_splitter import zh_title_enhance as func_zh_title_enhance
from server.utils import run_in_thread_pool
from server.knowledge_base.faq_utils import load_gen_file


class DocumentWithScores(Document):
    scores: Dict = {}


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str, vector_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store", vector_name)


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def list_kbs_from_folder():
    return [f for f in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, f))]


def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    result = []

    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False

    def process_entry(entry):
        if is_skiped_path(entry.path):
            return

        if entry.is_symlink():
            target_path = os.path.realpath(entry.path)
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    process_entry(target_entry)
        elif entry.is_file():
            file_path = (Path(os.path.relpath(entry.path, doc_path)).as_posix())  # 路径统一为 posix 格式
            result.append(file_path)
        elif entry.is_dir():
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    process_entry(sub_entry)

    with os.scandir(doc_path) as it:
        for entry in it:
            process_entry(entry)

    return result


def list_files_from_path(folder_path):
    result = []

    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False

    def process_entry(entry):
        if is_skiped_path(entry.path):
            return

        if entry.is_symlink():
            target_path = os.path.realpath(entry.path)
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    process_entry(target_entry)
        elif entry.is_file():
            file_path = (Path(os.path.relpath(entry.path, folder_path)).as_posix())  # 路径统一为 posix 格式
            result.append(file_path)
        elif entry.is_dir():
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    process_entry(sub_entry)

    with os.scandir(folder_path) as it:
        for entry in it:
            process_entry(entry)

    return result


LOADER_DICT = {"CustomHTMLLoader": ['.html'],
               "UnstructuredHTMLLoader": ['.htm'],
               "MHTMLLoader": ['.mhtml'],
               "UnstructuredMarkdownLoader": ['.md'],
               "JSONLoader": [".json"],
               "JSONLinesLoader": [".jsonl"],
               "CSVLoader": [".csv"],
               # "FilteredCSVLoader": [".csv"], 如果使用自定义分割csv
               # "RapidOCRPDFLoader": [".pdf"],
               # "RapidOCRDocLoader": ['.docx', '.doc'],
               # "RapidOCRPPTLoader": ['.ppt', '.pptx', ],
               # "RapidOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
               "UnstructuredFileLoader": ['.txt'],
               "UnstructuredEmailLoader": ['.eml', '.msg'],
               # "UnstructuredEPubLoader": ['.epub'],
               "UnstructuredExcelLoader": ['.xlsx', '.xls', '.xlsd'],
               # "NotebookLoader": ['.ipynb'],
               # "UnstructuredODTLoader": ['.odt'],
               # "PythonLoader": ['.py'],
               # "UnstructuredRSTLoader": ['.rst'],
               # "UnstructuredRTFLoader": ['.rtf'],
               # "SRTLoader": ['.srt'],
               # "TomlLoader": ['.toml'],
               "UnstructuredTSVLoader": ['.tsv'],
               "UnstructuredWordDocumentLoader": ['.docx', '.doc'],
               "UnstructuredXMLLoader": ['.xml'],
               "UnstructuredPowerPointLoader": ['.ppt', '.pptx'],
               # "TextLoader": ['.txt'],
               # "EverNoteLoader": ['.enex'],
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


class CustomHTMLLoader(langchain_community.document_loaders.unstructured.UnstructuredFileLoader):
    delimiter = " 之 "

    def load_content(self, file_path) -> List[Element]:
        elements: List[Element] = []

        last_modification_date = get_last_modified_date(file_path)
        file_directory, file_name = os.path.split(file_path)
        ext = os.path.splitext(file_name)[-1].lower()
        file_name = file_name[:-len(ext)]

        chapters_list = load_gen_file(file_path)

        for ii, chapter_tuple in enumerate(chapters_list):
            chapter_number = ii + 1

            chapter_title = file_name + self.delimiter + chapter_tuple[0]
            chapter_title_ele = Title(text=chapter_title, metadata=ElementMetadata(filename=file_path,
                                                                                   filetype="html",
                                                                                   page_number=chapter_number))
            chapter_title_ele.metadata.last_modified = last_modification_date
            chapter_title_ele.metadata.category_depth = 0

            elements.append(chapter_title_ele)

            paragraphs = chapter_tuple[1]
            for iii, paragraph_tuple in enumerate(paragraphs):
                # paragraph_title = chapter_tuple[0] + self.delimiter + paragraph_tuple[0]
                paragraph_title = file_name + self.delimiter + paragraph_tuple[0]
                paragraph_text = paragraph_tuple[1]
                sub_paragraphs = paragraph_tuple[2]

                paragraph_title_ele = Title(text=paragraph_title, metadata=ElementMetadata(filename=file_path,
                                                                                           filetype="html",
                                                                                           page_number=chapter_number))
                paragraph_title_ele.metadata.parent_id = chapter_title_ele.id
                paragraph_title_ele.metadata.last_modified = last_modification_date
                paragraph_title_ele.metadata.category_depth = 1

                elements.append(paragraph_title_ele)

                if sub_paragraphs:
                    for iii, sub_paragraph_tuple in enumerate(sub_paragraphs):
                        if sub_paragraph_tuple[0]:
                            sub_paragraph_title = file_name + self.delimiter + paragraph_tuple[0] + self.delimiter + \
                                                  sub_paragraph_tuple[0]
                        else:
                            sub_paragraph_title = file_name + self.delimiter + paragraph_tuple[0] + f" 段落{iii + 1}"

                        sub_paragraph_text = sub_paragraph_tuple[1]

                        sub_paragraph_title_ele = Title(text=sub_paragraph_title,
                                                        metadata=ElementMetadata(filename=file_path,
                                                                                 filetype="html",
                                                                                 page_number=chapter_number))
                        sub_paragraph_title_ele.metadata.parent_id = paragraph_title_ele.id
                        sub_paragraph_title_ele.metadata.last_modified = last_modification_date
                        sub_paragraph_title_ele.metadata.category_depth = 2

                        sub_paragraph_text_ele = Text(text=sub_paragraph_text,
                                                      metadata=ElementMetadata(filename=file_path,
                                                                               filetype="html",
                                                                               page_number=chapter_number))
                        sub_paragraph_text_ele.metadata.parent_id = paragraph_title_ele.id
                        sub_paragraph_text_ele.metadata.last_modified = last_modification_date

                        elements.append(sub_paragraph_title_ele)
                        elements.append(sub_paragraph_text_ele)
                else:
                    paragraph_text_ele = Text(text=paragraph_text, metadata=ElementMetadata())
                    paragraph_title_ele.metadata.parent_id = chapter_title_ele.id
                    paragraph_title_ele.metadata.last_modified = last_modification_date

                    elements.append(paragraph_text_ele)

        return elements

    def _get_elements(self) -> List:
        """Convert given content to documents."""
        return self.load_content(self.file_path)


langchain_community.document_loaders.CustomHTMLLoader = CustomHTMLLoader


class CustomExcelLoader(langchain_community.document_loaders.unstructured.UnstructuredFileLoader):

    def _get_elements(self) -> List:
        """Convert given content to documents."""
        return []


langchain_community.document_loaders.CustomExcelLoader = CustomExcelLoader


# patch json.dumps to disable ensure_ascii
def _new_json_dumps(obj, **kwargs):
    kwargs["ensure_ascii"] = False
    return _origin_json_dumps(obj, **kwargs)


if json.dumps is not _new_json_dumps:
    _origin_json_dumps = json.dumps
    json.dumps = _new_json_dumps


class JSONLinesLoader(langchain_community.document_loaders.JSONLoader):
    '''
    行式 Json 加载器，要求文件扩展名为 .jsonl
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True


langchain_community.document_loaders.JSONLinesLoader = JSONLinesLoader


def get_LoaderClass(file_extension, kb_name):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


# 把一些向量化共用逻辑从KnowledgeFile抽取出来，等langchain支持内存文件的时候，可以将非磁盘文件向量化
def get_loader(loader_name: str, file_path: str, loader_kwargs: Dict = None):
    '''
    根据loader_name和文件路径或内容返回文档加载器。
    '''
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in ["RapidOCRPDFLoader", "RapidOCRLoader", "FilteredCSVLoader",
                           "RapidOCRDocLoader", "RapidOCRPPTLoader"]:
            document_loaders_module = importlib.import_module('document_loaders')
        else:
            document_loaders_module = importlib.import_module('langchain_community.document_loaders')
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        document_loaders_module = importlib.import_module('langchain_community.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, 'rb') as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]

    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader


def make_text_splitter(
        splitter_name: str = TEXT_SPLITTER_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        llm_model: str = LLM_MODELS[0],
):
    """
    根据参数获取特定的分词器
    """
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if splitter_name == "MarkdownHeaderTextSplitter":  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = text_splitter_dict[splitter_name]['headers_to_split_on']
            text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        else:

            try:  ## 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module('text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:  ## 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)

            if text_splitter_dict[splitter_name]["source"] == "tiktoken":  ## 从tiktoken加载
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
            elif text_splitter_dict[splitter_name]["source"] == "huggingface":  ## 从huggingface加载
                if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "":
                    assert False, f"{splitter_name} tokenizer_name_or_path is empty"

                if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "gpt2":
                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:  ## 字符长度加载
                    tokenizer = AutoTokenizer.from_pretrained(
                        text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True)
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
    except Exception as e:
        print(e)
        # text_splitter_module = importlib.import_module('langchain.text_splitter')
        # TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        # text_splitter = TextSplitter(chunk_size=250, chunk_overlap=50)
        assert False, f"{splitter_name} load failed"
    return text_splitter


class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            knowledge_base_name: str,
            loader_kwargs: Dict = {},
    ):
        '''
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        '''
        self.kb_name = knowledge_base_name
        self.filename = str(Path(filename).as_posix())
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.filename}")
        self.loader_kwargs = loader_kwargs
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.splited_docs = None
        self.document_loader_name = get_LoaderClass(self.ext, knowledge_base_name)
        self.text_splitter_name = TEXT_SPLITTER_NAME

    def file2docs(self, refresh: bool = False):
        if self.docs is None or refresh:
            logger.info(f"{self.document_loader_name} used for {self.filepath}")
            loader = get_loader(loader_name=self.document_loader_name,
                                file_path=self.filepath,
                                loader_kwargs=self.loader_kwargs)
            self.docs = loader.load()
        return self.docs

    def docs2texts(
            self,
            docs: List[Document] = None,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    ):
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(splitter_name=self.text_splitter_name, chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        print(f"文档 {self.filename} 切分示例：{docs[0]}")
        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        return self.splited_docs

    def file2text(
            self,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    ):
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(docs=docs,
                                                zh_title_enhance=zh_title_enhance,
                                                refresh=refresh,
                                                chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                text_splitter=text_splitter)
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)


def files2docs_in_thread(
        files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
) -> Generator:
    '''
    利用多线程批量将磁盘文件转化成langchain Document.
    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    '''

    def file2docs(*, file: KnowledgeFile, **kwargs) -> Tuple[bool, Tuple[str, str, List[Document]]]:
        try:
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))
        except Exception as e:
            msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return False, (file.kb_name, file.filename, msg)

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    for result in run_in_thread_pool(func=file2docs, params=kwargs_list):
        yield result


def create_compressed_archive(folder_path, output_path, archive_format='zip'):
    """
    Create a compressed archive of the specified folder and save it to a specific path.

    :param folder_path: Path to the folder to be archived.
    :param output_path: Full path (including filename) where the archive will be saved.
    :param archive_format: Format of the archive ('zip', 'tar', etc.)
    :return: Path to the created archive.
    """
    # Ensure the folder exists
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    # Create the archive
    try:
        # Extract directory and archive name from the output path
        output_dir, archive_name = os.path.split(output_path)
        archive_name = os.path.splitext(archive_name)[0]  # Remove extension if any

        # Change the current working directory to the output directory if it exists
        original_cwd = os.getcwd()
        if output_dir and os.path.exists(output_dir):
            os.chdir(output_dir)

        # Create the archive
        archive_path = shutil.make_archive(archive_name, archive_format, folder_path)

        # Change back to the original directory
        os.chdir(original_cwd)

        print(f"Archive created at: {archive_path}")
        return archive_path
    except Exception as e:
        raise Exception(f"An error occurred while creating the archive: {e}")


if __name__ == "__main__":
    from pprint import pprint

    #
    # kb_file = KnowledgeFile(
    #     filename="/home/congyin/Code/Project_Langchain_0814/Langchain-Chatchat/knowledge_base/csv1/content/gm.csv",
    #     knowledge_base_name="samples")
    # # kb_file.text_splitter_name = "RecursiveCharacterTextSplitter"
    # docs = kb_file.file2docs()
    # # pprint(docs[-1])

    result = list_files_from_path("/Users/josh/projects/Langchain-Chatchat/server/db")

    for r in result:
        print(r)
