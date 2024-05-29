import re
from pathlib import Path

import operator
from abc import ABC, abstractmethod

import os
import asyncio
import hashlib
import numpy as np

from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document

from server.db.repository.knowledge_base_repository import (
    add_kb_to_db, delete_kb_from_db, list_kbs_from_db, kb_exists,
    load_kb_from_db, get_kb_detail,
)
from server.db.repository.knowledge_file_repository import (
    add_file_to_db, delete_file_from_db, delete_files_from_db, file_exists_in_db, list_files_info_from_db,
    count_files_from_db, list_files_from_db, get_file_detail, list_docs_from_db, list_question_from_db,
    list_answer_from_db, add_answer_to_db, add_question_to_db, get_answer_id_by_question_raw_id_from_db,
    get_answer_doc_id_by_answer_id_from_db, delete_docs_from_db, delete_answer_from_db, delete_question_from_db
)

from configs import (kbs_config, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD, EMBEDDING_MODEL, SEARCH_ENHANCE)
from server.knowledge_base.utils import (
    get_kb_path, get_doc_path, KnowledgeFile,
    list_kbs_from_folder, list_files_from_folder, DocumentWithScores
)
from server.knowledge_base.faq_utils import load_faq

from typing import List, Union, Dict, Optional

from server.embeddings_api import embed_texts, embed_documents, aembed_texts, aembed_documents_api
from server.knowledge_base.model.kb_document_model import DocumentWithVSId
from server.knowledge_base.kb_cache.bm25_cache import kb_bm25_pool, ThreadSafeBM25, get_score


def normalize(embeddings: List[List[float]]) -> np.ndarray:
    '''
    sklearn.preprocessing.normalize 的替代（使用 L2），避免安装 scipy, scikit-learn
    '''
    norm = np.linalg.norm(embeddings, axis=1)
    norm = np.reshape(norm, (norm.shape[0], 1))
    norm = np.tile(norm, (1, len(embeddings[0])))
    return np.divide(embeddings, norm)


class SupportedVSType:
    FAISS = 'faiss'
    MILVUS = 'milvus'
    DEFAULT = 'default'
    ZILLIZ = 'zilliz'
    PG = 'pg'
    ES = 'es'


def merge_scores(x, y, is_max=False):
    if is_max:
        return {k: max(x.get(k, 0.0), y.get(k, 0.0)) for k in set(x) | set(y)}
    else:
        return {k: x.get(k, 0.0) + y.get(k, 0.0) for k in set(x) | set(y)}


class KBService(ABC):

    def __init__(self,
                 knowledge_base_name: str,
                 kb_info: str,
                 kb_agent_guide: str,
                 search_enhance: bool = SEARCH_ENHANCE,
                 embed_model: str = EMBEDDING_MODEL,
                 ):
        self.kb_name = knowledge_base_name
        self.kb_info = kb_info
        self.kb_agent_guide = kb_agent_guide
        self.embed_model = embed_model
        self.search_enhance = search_enhance
        self.kb_path = get_kb_path(self.kb_name)
        self.doc_path = get_doc_path(self.kb_name)
        self.do_init()
        self.kb_summary = kb_info

    def __repr__(self) -> str:
        return f"{self.kb_name} @ {self.embed_model}"

    def save_vector_store(self, vector_name):
        '''
        保存向量库:FAISS保存到磁盘，milvus保存到数据库。PGVector暂未支持
        '''
        pass

    def create_kb(self):
        """
        创建知识库
        """
        print("create_kb")

        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)

        self.do_create_kb("docs")
        self.do_create_kb("question")
        self.do_create_kb("answer")
        self.do_create_kb("query")

        status = add_kb_to_db(self.kb_name, self.kb_info, self.kb_agent_guide, self.kb_summary, self.vs_type(),
                              self.embed_model, self.search_enhance)

        return status

    def clear_vs(self):
        """
        删除向量库中所有内容
        """
        print("clear_vs")

        self.do_clear_vs("docs")
        self.do_clear_vs("question")
        self.do_clear_vs("answer")
        self.do_clear_vs("query")

        status = delete_files_from_db(self.kb_name)
        return status

    def drop_kb(self):
        """
        删除知识库
        """
        print("drop_kb")

        self.do_drop_kb()
        status = delete_kb_from_db(self.kb_name)
        return status

    def _docs_to_embeddings(self, docs: List[Document], use_async_thres=50) -> Dict:
        '''
        将 List[Document] 转化为 VectorStore.add_embeddings 可以接受的参数
        '''
        if len(docs) > use_async_thres * 2:
            return asyncio.run(aembed_documents_api(docs, self.embed_model, use_async_thres))
        else:
            return embed_documents(docs=docs, embed_model=self.embed_model, to_query=False)

    def add_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        向知识库添加文件
        如果指定了docs，则不再将文本向量化，并将数据库对应条目标为custom_docs=True
        """
        print("add_doc")

        if docs:
            custom_docs = True
            for doc in docs:
                doc.metadata.setdefault("source", kb_file.filename)
        else:
            docs = kb_file.file2text()
            custom_docs = False

        if docs:
            # 将 metadata["source"] 改为相对路径
            for idx, doc in enumerate(docs):
                doc.metadata["idx"] = str(idx)
                source = doc.metadata.get("source", "")
                try:
                    if os.path.isabs(source):
                        rel_path = Path(source).relative_to(self.doc_path)
                        doc.metadata["source"] = str(rel_path.as_posix().strip("/"))
                except Exception as e:
                    print(f"cannot convert absolute path ({source}) to relative path. error is : {e}")

            # self.delete_doc(kb_file)

            doc_infos = self.do_add_doc("docs", docs, **kwargs)
            status = add_file_to_db(kb_file,
                                    custom_docs=custom_docs,
                                    docs_count=len(docs),
                                    doc_infos=doc_infos)
        else:
            status = False
        return status

    def delete_doc(self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs):
        """
        从知识库删除文件
        """
        print("delete_doc")

        self.do_delete_doc("docs", kb_file, **kwargs)

        delete_docs_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        delete_answer_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        delete_question_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        delete_file_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)

        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)

    def add_faq(self, kb_file: KnowledgeFile, **kwargs):
        """
        向知识库添加文件
        如果指定了docs，则不再将文本向量化，并将数据库对应条目标为custom_docs=True
        """

        print("add_faq")

        filename = kb_file.filename

        label_list, std_label_list, answer_list, label_dict, as_count_list, _ = load_faq(kb_file.filepath)

        # self.delete_faq(kb_file, **kwargs)

        status = add_file_to_db(kb_file, docs_count=len(label_list))

        if not status:
            return status

        answer_dict = dict()
        for idx, question in enumerate(label_list):
            a_idx = as_count_list[idx]
            answer = answer_list[a_idx]

            docq = Document(page_content=question)
            docq.metadata["source"] = filename
            docq.metadata["raw_id"] = str(idx)
            docq.metadata["answer_id"] = str(a_idx)

            if a_idx in answer_dict:
                answer_dict[a_idx].append(docq)
            else:
                qna = "### 问题：" + std_label_list[a_idx] + "\n### 答案：\n" + re.sub("(\s*\n\s*)+", "\n",
                                                                                     answer).strip()
                doca = Document(page_content=qna)
                doca.metadata["source"] = filename
                doca.metadata["raw_id"] = str(a_idx)

                answer_dict[a_idx] = [doca, docq]

        answer_objs = list()
        question_objs = list()
        answer_id_for_question_list = list()

        for a_idx, ele_list in answer_dict.items():
            answer_id = str(a_idx)
            answer_obj = ele_list[0]
            questions = ele_list[1:]

            assert answer_id == answer_obj.metadata["raw_id"]

            answer_objs.append(answer_obj)
            question_objs += questions

            answer_id_for_question_list += [answer_id] * len(questions)

        answer_id_list = [a.metadata["raw_id"] for a in answer_objs]
        answer_file_name_list = [a.metadata["source"] for a in answer_objs]

        question_id_list = [q.metadata["raw_id"] for q in question_objs]
        question_file_name_list = [q.metadata["source"] for q in question_objs]

        a_doc_infos = self.do_add_doc('answer', answer_objs, **kwargs)

        this_status_a = add_answer_to_db(self.kb_name, answer_file_name_list, answer_id_list, doc_infos=a_doc_infos)

        q_doc_infos = self.do_add_doc('question', question_objs, **kwargs)

        this_status_q = add_question_to_db(self.kb_name, question_file_name_list, answer_id_for_question_list,
                                           question_id_list, doc_infos=q_doc_infos)
        status = this_status_a & this_status_q

        return status

    def delete_faq(self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs):
        """
        从知识库删除文件
        """
        print("delete_faq")

        self.do_delete_doc("answer", kb_file, **kwargs)
        self.do_delete_doc("question", kb_file, **kwargs)

        delete_docs_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        delete_answer_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        delete_question_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)
        delete_file_from_db(kb_name=kb_file.kb_name, file_name=kb_file.filename)

        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)

    def update_faq(self, kb_file: KnowledgeFile, **kwargs):
        """
        使用content中的文件更新向量库
        如果指定了docs，则使用自定义docs，并将数据库对应条目标为custom_docs=True
        """
        print("update_faq")

        if os.path.exists(kb_file.filepath):
            self.delete_faq(kb_file, **kwargs)
            return self.add_faq(kb_file, **kwargs)
        else:
            return False

    def update_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        使用content中的文件更新向量库
        如果指定了docs，则使用自定义docs，并将数据库对应条目标为custom_docs=True
        """
        print("update_doc")

        if os.path.exists(kb_file.filepath):
            self.delete_doc(kb_file, **kwargs)
            return self.add_doc(kb_file, docs=docs, **kwargs)
        else:
            return False

    def exist_doc(self, file_name: str):
        return file_exists_in_db(KnowledgeFile(knowledge_base_name=self.kb_name,
                                               filename=file_name, document_loader_name='unknown'))

    def list_files(self):
        return list_files_from_db(self.kb_name)

    def list_files_info(self):
        return list_files_info_from_db(self.kb_name)

    def count_files(self):
        return count_files_from_db(self.kb_name)

    def merge_answers(self, answer_data: List[DocumentWithScores], answer_data_2: List[DocumentWithScores],
                      is_max=False) -> List[DocumentWithScores]:
        new_answer_data_dict = {a.metadata["raw_id"]: a.scores for a in answer_data_2}

        merged_answer_data = list()
        for answer in answer_data:
            answer_raw_id = answer.metadata["raw_id"]

            if answer_raw_id in new_answer_data_dict:
                new_score = merge_scores(answer.scores, new_answer_data_dict[answer_raw_id], is_max=is_max)

                merged_answer_data.append((answer, new_score))
            else:
                merged_answer_data.append((answer, answer.scores))

        merged_answer_data_id = [a.metadata["raw_id"] for a, _ in merged_answer_data]

        for answer in answer_data_2:
            answer_raw_id = answer.metadata["raw_id"]

            if answer_raw_id not in merged_answer_data_id:
                merged_answer_data.append((answer, answer.scores))

        return [DocumentWithScores(**{"page_content": d.page_content, "metadata": d.metadata}, scores=s) for d, s in
                merged_answer_data]

    def merge_docs(self, docs_data: List[DocumentWithScores], docs_data_2: List[DocumentWithScores], is_max=False) -> \
            List[DocumentWithScores]:
        new_docs_data_dict = {d.page_content: d.scores for d in docs_data_2}
        docs_data_value = [d.page_content for d in docs_data]
        merged_docs_data = list()

        for d in docs_data:
            page_content = d.page_content

            if page_content in new_docs_data_dict:
                new_score = merge_scores(d.scores, new_docs_data_dict[page_content], is_max=is_max)

                merged_docs_data.append((d, new_score))
            else:
                merged_docs_data.append((d, d.scores))

        for d in docs_data_2:
            page_content = d.page_content

            if page_content not in docs_data_value:
                merged_docs_data.append((d, d.scores))

        return [DocumentWithScores(**{"page_content": d.page_content, "metadata": d.metadata}, scores=s) for d, s in
                merged_docs_data]

    def question_to_answer(self, question_data: List[DocumentWithScores]):
        doc_ids = list()
        scores_list = list()
        for qd in question_data:
            question_id = qd.metadata["raw_id"]
            scores = qd.scores

            # print(f"question_id {question_id} {type(question_id)}")
            answer_id = get_answer_id_by_question_raw_id_from_db(self.kb_name, question_id)
            # print(f"answer_id {answer_id} {type(answer_id)}")

            if not answer_id:
                continue

            doc_id = get_answer_doc_id_by_answer_id_from_db(self.kb_name, answer_id)
            # print(f"doc_id {doc_id} {type(doc_id)}")

            if not doc_id:
                continue

            if doc_id not in doc_ids:
                doc_ids.append(doc_id)
                scores_list.append(scores)

        answers = list(zip(self.get_doc_by_ids("answer", doc_ids), scores_list))

        return [DocumentWithScores(**{"page_content": d.page_content, "metadata": d.metadata}, scores=s) for d, s in
                answers]

    def search_allinone(self,
                        query: str,
                        top_k: int = VECTOR_SEARCH_TOP_K,
                        score_threshold: float = SCORE_THRESHOLD,
                        ) -> (List[DocumentWithScores], List[DocumentWithScores]):
        query_embedding, docs_data = self.search_docs("docs", query, top_k, score_threshold)

        _, answer_data = self.search_docs("answer", query, top_k, score_threshold, embeddings=query_embedding)

        # print(f"answer_data {answer_data}")

        _, question_data = self.search_docs("question", query, top_k, score_threshold, embeddings=query_embedding)

        # print(f"question_data {question_data}")

        if question_data:
            new_question_data = self.question_to_answer(question_data)
        else:
            new_question_data = list()
        # print(f"new_question_data {new_question_data}")
        merged_answer_data = self.merge_answers(answer_data, new_question_data, is_max=True)
        # print(f"merged_answer_data {merged_answer_data}")

        return docs_data, merged_answer_data

    def load_bm25_retriever(self, retriever_name, file_names, docs_text_list, metadata_list) -> ThreadSafeBM25:
        file_names_text = " ".join(file_names)

        file_md5_sum = hashlib.md5(file_names_text.encode('utf-8')).hexdigest()

        return kb_bm25_pool.load_retriever(self.kb_name, retriever_name, file_md5_sum, docs_text_list, metadata_list)

    def enhance_search_allinone(self, query: str, top_k: int, bm_factor: float) -> (
            List[DocumentWithScores], List[DocumentWithScores]):
        docs_data = list()
        answer_data = list()
        question_data = list()

        file_infos = self.list_files_info()

        file_names = list()
        faq_names = list()

        docs_text_list = list()
        docs_metadata_list = list()

        questions_text_list = list()
        questions_metadata_list = list()

        answers_text_list = list()
        answers_metadata_list = list()

        for file_name, document_loader_name in file_infos:
            if document_loader_name != "CustomExcelLoader":
                file_names.append(f"{file_name}_{document_loader_name}")
                documentWithVSIds = self.list_docs("docs", file_name=file_name)
                docs_text_list += [i.page_content for i in documentWithVSIds]
                docs_metadata_list += [i.metadata for i in documentWithVSIds]
            else:
                faq_names.append(f"{file_name}_{document_loader_name}")
                documentWithVSIds = self.list_docs("question", file_name=file_name)
                questions_text_list += [i.page_content for i in documentWithVSIds]
                questions_metadata_list += [i.metadata for i in documentWithVSIds]

                documentWithVSIds = self.list_docs("answer", file_name=file_name)
                answers_text_list += [i.page_content for i in documentWithVSIds]
                answers_metadata_list += [i.metadata for i in documentWithVSIds]

        if file_names:
            with self.load_bm25_retriever("docs", file_names, docs_text_list, docs_metadata_list).acquire() as vs:
                if len(vs.docs) > 0:
                    norm_scores = get_score(vs, query)
                    top_3_idx = np.argsort(norm_scores)[::-1][:top_k]
                    for idx, doc in enumerate(vs.docs):
                        if idx in top_3_idx:
                            docs_data.append((doc, norm_scores[idx] * bm_factor))

        if faq_names:
            with self.load_bm25_retriever("answer", faq_names, answers_text_list,
                                          answers_metadata_list).acquire() as vs:
                if len(vs.docs) > 0:
                    norm_scores = get_score(vs, query)

                    top_3_idx = np.argsort(norm_scores)[::-1][:top_k]

                    for idx, doc in enumerate(vs.docs):
                        if idx in top_3_idx:
                            answer_data.append((doc, norm_scores[idx] * bm_factor))

            with self.load_bm25_retriever("question", faq_names, questions_text_list,
                                          questions_metadata_list).acquire() as vs:
                if len(vs.docs) > 0:
                    norm_scores = get_score(vs, query)
                    top_3_idx = np.argsort(norm_scores)[::-1][:top_k]
                    for idx, doc in enumerate(vs.docs):
                        if idx in top_3_idx:
                            question_data.append((doc, norm_scores[idx] * bm_factor))

        # print(f"3 question_data {question_data}")

        docs_data = [DocumentWithScores(**d.dict(), scores={"bm_docs": s}) for d, s in docs_data]
        answer_data = [DocumentWithScores(**d.dict(), scores={"bm_answer": s}) for d, s in answer_data]
        question_data = [DocumentWithScores(**d.dict(), scores={"bm_question": s}) for d, s in question_data]

        if question_data:
            new_question_data = self.question_to_answer(question_data)
        else:
            new_question_data = list()
        # print(f"4 new_question_data {new_question_data}")
        merged_answer_data = self.merge_answers(answer_data, new_question_data, is_max=True)
        # print(f"5 merged_answer_data {merged_answer_data}")

        return docs_data, merged_answer_data

    def search_docs(self,
                    vector_name: str,
                    query: str,
                    top_k: int = VECTOR_SEARCH_TOP_K,
                    score_threshold: float = SCORE_THRESHOLD,
                    embeddings: List[float] = None,
                    ) -> (List[float], List[DocumentWithScores]):
        embedding, docs = self.do_search(vector_name, query, top_k, score_threshold, embeddings=embeddings)
        return embedding, docs

    def get_doc_by_ids(self, vector_name, ids: List[str]) -> List[Document]:
        return []

    def list_docs(self, vector_name, file_name: str = None, metadata: Dict = {}) -> List[DocumentWithVSId]:
        '''
        通过file_name或metadata检索Document
        '''
        if vector_name == "docs":
            doc_infos = list_docs_from_db(kb_name=self.kb_name, file_name=file_name, metadata=metadata)
        elif vector_name == "question":
            doc_infos = list_question_from_db(kb_name=self.kb_name, file_name=file_name, metadata=metadata)
        elif vector_name == "answer":
            doc_infos = list_answer_from_db(kb_name=self.kb_name, file_name=file_name, metadata=metadata)
        else:
            assert False

        doc_infos_ids = [x["id"] for x in doc_infos]
        doc_infos_s = self.get_doc_by_ids(vector_name, doc_infos_ids)
        docs = []
        for id, doc_info_s in zip(doc_infos_ids, doc_infos_s):
            if doc_info_s is not None:
                # 处理非空的情况
                doc_with_id = DocumentWithVSId(**doc_info_s.dict(), id=id)
                docs.append(doc_with_id)
            else:
                # 处理空的情况
                # 可以选择跳过当前循环迭代或执行其他操作
                pass
        return docs

    def get_relative_source_path(self, filepath: str):
        '''
        将文件路径转化为相对路径，保证查询时一致
        '''
        relative_path = filepath
        if os.path.isabs(relative_path):
            try:
                relative_path = Path(filepath).relative_to(self.doc_path)
            except Exception as e:
                print(f"cannot convert absolute path ({source}) to relative path. error is : {e}")

        relative_path = str(relative_path.as_posix().strip("/"))
        return relative_path

    @abstractmethod
    def do_create_kb(self, vs_path):
        """
        创建知识库子类实自己逻辑
        """
        pass

    @staticmethod
    def list_kbs_type():
        return list(kbs_config.keys())

    @classmethod
    def list_kbs(cls):
        return list_kbs_from_db()

    def exists(self, kb_name: str = None):
        kb_name = kb_name or self.kb_name
        return kb_exists(kb_name)

    @abstractmethod
    def vs_type(self) -> str:
        pass

    @abstractmethod
    def do_init(self):
        pass

    @abstractmethod
    def do_drop_kb(self):
        """
        删除知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_search(self,
                  vector_name: str,
                  query: str,
                  top_k: int,
                  score_threshold: float,
                  embeddings: List[float] = None,
                  ) -> (List[float], List[DocumentWithScores]):
        """
        搜索知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_add_doc(self,
                   vector_name: str,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:
        """
        向知识库添加文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_delete_doc(self,
                      vector_name: str,
                      kb_file: KnowledgeFile,
                      **kwargs
                      ):
        """
        从知识库删除文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_clear_vs(self, vector_name: str):
        """
        从知识库删除全部向量子类实自己逻辑
        """
        pass

    @abstractmethod
    def count_docs(self, vector_name: str, filename: str):
        pass


class KBServiceFactory:

    @staticmethod
    def get_service(kb_name: str,
                    kb_info: str,
                    kb_agent_guide: str,
                    vector_store_type: Union[str, SupportedVSType],
                    embed_model: str = EMBEDDING_MODEL,
                    search_enhance: bool = SEARCH_ENHANCE,
                    ) -> KBService:
        # if isinstance(vector_store_type, str):
        #     vector_store_type = getattr(SupportedVSType, vector_store_type.upper())
        # if SupportedVSType.MILVUS == vector_store_type:
        from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
        return MilvusKBService(kb_name, kb_info, kb_agent_guide, search_enhance, embed_model=embed_model)
        # else:
        #     from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
        #     return FaissKBService(kb_name, kb_info, kb_agent_guide, search_enhance, embed_model=embed_model)
        # elif SupportedVSType.PG == vector_store_type:
        #     from server.knowledge_base.kb_service.pg_kb_service import PGKBService
        #     return PGKBService(kb_name, embed_model=embed_model)
        # elif SupportedVSType.ZILLIZ == vector_store_type:
        #     from server.knowledge_base.kb_service.zilliz_kb_service import ZillizKBService
        #     return ZillizKBService(kb_name, embed_model=embed_model)
        # elif SupportedVSType.DEFAULT == vector_store_type:
        #     from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
        #     return FaissKBService(kb_name, search_enhance, embed_model=embed_model)
        # elif SupportedVSType.ES == vector_store_type:
        #     from server.knowledge_base.kb_service.es_kb_service import ESKBService
        #     return ESKBService(kb_name, embed_model=embed_model)
        # elif SupportedVSType.DEFAULT == vector_store_type:  # kb_exists of default kbservice is False, to make validation easier.
        #     from server.knowledge_base.kb_service.default_kb_service import DefaultKBService
        #     return DefaultKBService(kb_name)

    @staticmethod
    def get_service_by_name(kb_name: str) -> Optional[KBService]:
        kb_name, kb_info, kb_agent_guide, _, vs_type, embed_model, search_enhance = load_kb_from_db(kb_name)
        if kb_name is None:  # kb not in db, just return None
            return None
        return KBServiceFactory.get_service(kb_name, kb_info, kb_agent_guide, vs_type, embed_model, search_enhance)

    # @staticmethod
    # def get_default():
    #     return KBServiceFactory.get_service("default", SupportedVSType.DEFAULT)


def get_kb_details() -> List[Dict]:
    kbs_in_folder = list_kbs_from_folder()
    kbs_in_db = KBService.list_kbs()
    kbs_in_db = [i[0] for i in kbs_in_db]
    result = {}

    for kb_name in kbs_in_db:
        result[kb_name] = {
            "kb_name": kb_name,
            "vs_type": "",
            "kb_info": "",
            "kb_agent_guide": "",
            "kb_summary": "",
            "embed_model": "",
            "file_count": 0,
            "create_time": None,
            "in_folder": kb_name in kbs_in_folder,
            "in_db": True,
        }

    for kb_name in kbs_in_folder:
        if kb_name not in result:
            result[kb_name] = {
                "kb_name": kb_name,
                "vs_type": "",
                "kb_info": "",
                "kb_agent_guide": "",
                "kb_summary": "",
                "embed_model": "",
                "file_count": 0,
                "create_time": None,
                "in_folder": True,
                "in_db": kb_name in kbs_in_db,
            }

    for kb_name in kbs_in_db:
        kb_detail = get_kb_detail(kb_name)
        if kb_detail:
            kb_detail["in_db"] = True
            if kb_name in result:
                result[kb_name].update(kb_detail)
            else:
                kb_detail["in_folder"] = False
                result[kb_name] = kb_detail

    data = []
    for i, v in enumerate(result.values()):
        v['No'] = i + 1
        data.append(v)

    return data


def get_kb_file_details(kb_name: str) -> List[Dict]:
    kb = KBServiceFactory.get_service_by_name(kb_name)
    if kb is None:
        return []

    files_in_folder = list_files_from_folder(kb_name)
    files_in_db = kb.list_files()
    result = {}

    for doc in files_in_db:
        result[doc] = {
            "kb_name": kb_name,
            "file_name": doc,
            "file_ext": os.path.splitext(doc)[-1],
            "file_version": 0,
            "document_loader": "",
            "docs_count": 0,
            "text_splitter": "",
            "create_time": None,
            "in_folder": doc in files_in_folder,
            "in_db": True,
        }

    for doc in files_in_folder:
        if doc not in result:
            result[doc] = {
                "kb_name": kb_name,
                "file_name": doc,
                "file_ext": os.path.splitext(doc)[-1],
                "file_version": 0,
                "document_loader": "",
                "docs_count": 0,
                "text_splitter": "",
                "create_time": None,
                "in_folder": True,
                "in_db": doc in files_in_db,
            }

    lower_names = {x.lower(): x for x in result}
    for doc in files_in_db:
        doc_detail = get_file_detail(kb_name, doc)
        if doc_detail:
            doc_detail["in_db"] = True
            if doc.lower() in lower_names:
                result[lower_names[doc.lower()]].update(doc_detail)
            else:
                doc_detail["in_folder"] = False
                result[doc] = doc_detail

    data = []
    for i, v in enumerate(result.values()):
        v['No'] = i + 1
        data.append(v)

    return data


class EmbeddingsFunAdapter(Embeddings):
    def __init__(self, embed_model: str = EMBEDDING_MODEL):
        self.embed_model = embed_model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = embed_texts(texts=texts, embed_model=self.embed_model, to_query=False).data
        return normalize(embeddings).tolist()

    def embed_query(self, text: str) -> List[float]:
        embeddings = embed_texts(texts=[text], embed_model=self.embed_model, to_query=True).data
        query_embed = embeddings[0]
        query_embed_2d = np.reshape(query_embed, (1, -1))  # 将一维数组转换为二维数组
        normalized_query_embed = normalize(query_embed_2d)
        return normalized_query_embed[0].tolist()  # 将结果转换为一维数组并返回

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = (await aembed_texts(texts=texts, embed_model=self.embed_model, to_query=False)).data
        return normalize(embeddings).tolist()

    async def aembed_query(self, text: str) -> List[float]:
        embeddings = (await aembed_texts(texts=[text], embed_model=self.embed_model, to_query=True)).data
        query_embed = embeddings[0]
        query_embed_2d = np.reshape(query_embed, (1, -1))  # 将一维数组转换为二维数组
        normalized_query_embed = normalize(query_embed_2d)
        return normalized_query_embed[0].tolist()  # 将结果转换为一维数组并返回


def score_threshold_process(score_threshold, k, docs):
    if score_threshold is not None:
        cmp = (
            operator.le
        )
        docs = [
            (doc, similarity)
            for doc, similarity in docs
            if cmp(similarity, score_threshold)
        ]
    return docs[:k]

    # def update_info(self, kb_info: str):
    #     """
    #     更新知识库介绍
    #     """
    #     self.kb_info = kb_info
    #     status = add_kb_to_db(self.kb_name, self.kb_info, self.kb_agent_guide, self.kb_summary, self.vs_type(),
    #                           self.embed_model, self.search_enhance)
    #     return status

    # def update_agent_guide(self, kb_agent_guide: str):
    #     """
    #     更新知识库Agent介绍
    #     """
    #     self.kb_agent_guide = kb_agent_guide
    #     status = add_kb_to_db(self.kb_name, self.kb_info, self.kb_agent_guide, self.kb_summary, self.vs_type(),
    #                           self.embed_model, self.search_enhance)
    #     return status
