from pathlib import Path


import operator
from abc import ABC, abstractmethod
import logging
import itertools
import os
import re
import functools
from collections import Counter

import numpy as np
import pandas
from langchain.embeddings.base import Embeddings
from langchain.docstore.document import Document
from LAC import LAC

from server.db.repository.knowledge_base_repository import (
    add_kb_to_db, delete_kb_from_db, list_kbs_from_db, kb_exists,
    load_kb_from_db, get_kb_detail,
)
from server.db.repository.knowledge_file_repository import (
    add_file_to_db, delete_file_from_db, delete_files_from_db, file_exists_in_db,
    count_files_from_db, list_files_from_db, get_file_detail, list_docs_from_db,
    add_answer_to_db, add_question_to_db
)

from configs import (kbs_config, VECTOR_SEARCH_TOP_K, SCORE_THRESHOLD,
                     EMBEDDING_MODEL, KB_INFO, SEARCH_ENHANCE)
from server.knowledge_base.utils import (
    get_kb_path, get_doc_path, KnowledgeFile,
    list_kbs_from_folder, list_files_from_folder,
)

from typing import List, Union, Dict, Optional

from server.embeddings_api import embed_texts
from server.embeddings_api import embed_documents
from server.knowledge_base.model.kb_document_model import DocumentWithVSId


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

lac = LAC(mode='seg')
term_dict_file = "/opt/projects/zhimakaimen/data/kbqa/custom_20230720.txt"
if not os.path.exists(term_dict_file):
    term_dict_file = ""
term_dictionary = list()
if term_dict_file:
    with open(term_dict_file) as f:
        term_dictionary = [i.replace("\n", "")[:-4] for i in f.readlines()]
    logger.info(f"term_dictionary len {len(term_dictionary)}")
    lac.load_customization(term_dict_file, sep=None)

stopwords_file = "/opt/projects/zhimakaimen/data/kbqa/stopwords.txt"
if not os.path.exists(stopwords_file):
    stopwords_file = ""
stopwords = list()
if stopwords_file:
    with open(stopwords_file) as f:
        stopwords = [i.replace("\n", "") for i in f.readlines()]
    logger.info(f"stopwords len {len(stopwords)}")


class Query:

    def __init__(self, index, raw_q):
        self.index = index
        self.raw_q = raw_q
        self.q_entity = ""
        self.q_type = ""
        self.q_target = ""
        self.q_terms = ""
        self.attri = ""
        self.cls_list = list()
        self.tag_count = dict()

    def set_parse_attr(self, q_entity, q_type, q_target, q_terms):
        self.q_entity = q_entity
        self.q_type = q_type
        self.q_target = q_target
        self.q_terms = q_terms

    def count_tag(self, tag):
        if tag in self.tag_count:
            self.tag_count[tag] += 1
        else:
            self.tag_count[tag] = 1


class UserQuery(Query):

    def __init__(self, index, raw_q):
        super().__init__(index, raw_q)

        self.corrected_raw_q = raw_q

        self.std_q = clean_text(raw_q)
        self.std_q_no_stop = remove_stop(self.std_q)

        self.corrected_std_q = clean_text(self.corrected_raw_q)
        self.corrected_std_q_no_stop = remove_stop(self.corrected_std_q)

        self.label_count = dict()
        self.label_idx = -1

    def __str__(self):
        details = 'UserQuery\n'
        details += f'raw: {self.raw_q}\n'
        details += f'corrected: {self.corrected_raw_q}\n'
        details += f'corrected_std_q: {self.corrected_std_q}\n'
        details += f'corrected_std_q_no_stop: {self.corrected_std_q_no_stop}\n'
        # details += f'entity: {self.q_entity}\n'
        # details += f'type: {self.q_type}\n'
        # details += f'target: {self.q_target}\n'
        # details += f'terms: {self.q_terms}\n'
        return details

    # def correct(self):
    #     self.corrected_raw_q = do_correction.query_correction(self.corrected_raw_q)

    def count_label(self, label):
        if label in self.label_count:
            self.label_count[label] += 1
        else:
            self.label_count[label] = 1

    def gen_label_idx(self, label_list):
        label = max(self.label_count, key=self.label_count.get)
        self.label_idx = label_list.index(label)

    def update_raw_q(self, raw_q_updated):
        if raw_q_updated != self.raw_q:
            self.raw_q = raw_q_updated
            self.corrected_raw_q = raw_q_updated
            self.std_q = clean_text(raw_q_updated)
            self.std_q_no_stop = remove_stop(self.std_q)
            self.corrected_std_q = clean_text(self.corrected_raw_q)
            self.corrected_std_q_no_stop = remove_stop(self.corrected_std_q)


class StandardQuery(Query):

    def __init__(self, index, raw_q, raw_a):
        super().__init__(index, raw_q)

        self.raw_a = raw_a

        self.std_q = clean_text(raw_q)
        self.std_a = clean_answer(raw_a)
        self.std_a_valiad = is_valid_ans(self.std_a)

        self.std_q_no_stop = remove_stop(self.std_q)
        self.std_a_no_stop = remove_stop(self.std_a)

        self.end_dt = ""
        self.ref = ""

        self.extend_list = list()
        self.extend_reg_dict = dict()
        self.sample_list_of_list = list()

    def __str__(self):
        details = ''
        details += f'StandardQuery raw: {self.raw_q}\n'
        return details


@functools.lru_cache()
def seg_text(sentence):
    seg_result = lac.run([sentence])
    seg_result = seg_result[0]
    return [j.strip() for j in seg_result if j.strip()]


def seg_text_list(sentence_list, return_list_of_list=False):
    seg_result = lac.run(sentence_list)
    seg_result_text = list()
    for i in seg_result:
        if return_list_of_list:
            seg_result_text.append([j.strip() for j in i if j.strip()])
        else:
            text = " ".join([j.strip() for j in i if j.strip()])
            seg_result_text.append(text)
    return seg_result_text


def clean_text(lbl, remove_stop=False, return_list=False):
    lbl = re.sub("[^A-Za-z0-9\u4e00-\u9fa5]", "_", lbl)
    lbl = re.sub("_+", " ", lbl)
    lbl = re.sub(" {2,}", " ", lbl).strip()
    lbl = lbl.strip().upper()

    if remove_stop:
        result = list()
        for word in seg_text(lbl):
            if word not in stopwords:
                result.append(word)
        if return_list:
            lbl = result
        else:
            lbl = "".join(result)
    return lbl


def remove_stop(ans):
    ans = ans.upper()
    result = list()
    for word in seg_text(ans):
        if word not in stopwords:
            result.append(word)
    ans = "".join(result)
    return ans


def clean_answer(ans, remove_stop=False):
    ans = ans.replace("<HTML>", "")
    ans = ans.replace("</HTML>", "")
    ans = ans.replace("[Strong]", "")
    ans = ans.replace("[/Strong]", "")
    ans = ans.replace("<b>", "")
    ans = ans.replace("</b>", "")
    ans = ans.replace("<br>", "")
    ans = ans.replace("&lt;", "")
    ans = ans.replace("&gt;", "")
    ans = re.sub("\[MAT\].*\[\/MAT\]", " ", ans)
    ans = re.sub("\[biaoqian\].*\[\/biaoqian\]", " ", ans)
    ans = re.sub("\<a\shref.*\<\/a\>", " ", ans)
    ans = re.sub("\s{2,}", " ", ans).strip()

    if remove_stop:
        ans = ans.upper()
        result = list()
        for word in seg_text(ans):
            if word not in stopwords:
                result.append(word)
        ans = "".join(result)

    return ans


def is_valid_ans(std_ans_txt: str):
    for ch in std_ans_txt:
        if u'\u4e00' <= ch <= u'\u9fff':
            return True
    return False


def is_valid_query(raw_query_txt: str):
    return len(raw_query_txt) > 0


def is_valid_std_query(raw_query_txt: str, std_ans_txt: str):
    return is_valid_query(raw_query_txt) and is_valid_ans(std_ans_txt)


def is_valid_user_query(raw_query_txt: str):
    return is_valid_query(raw_query_txt)


def alter_tokens(tokens):
    new_tokens = list()
    for token in tokens:
        token = token.strip()
        if token:
            if "@" == token[0]:
                token = token[1:]
            if "?" == token[-1]:
                token = token[:-1]
            new_tokens.append(token)
    return new_tokens


def process_sys_reg(raw_text):
    token_list = re.findall(r"{.+?}|.+?", raw_text)
    token_list2 = list()
    token_list3 = list()

    for ele in token_list:
        if ele.startswith("{"):
            token_list3.append("".join(token_list2))
            token_list2 = list()
            token_list3.append(ele)
        else:
            token_list2.append(ele)
    if token_list2:
        token_list3.append("".join(token_list2))

    processed_list = list()

    for tokens in token_list3:
        if tokens.startswith("{"):
            tokens = tokens[1:-1]
            split_list = tokens.split("]|[")
            sub_processed_list = list()
            for splits in split_list:
                final_res_tokens = list()
                if not splits.startswith("["):
                    splits = "[" + splits
                if not splits.endswith("]"):
                    splits = splits + "]"
                ss_list = re.findall(r"\[.*?\]", splits)
                for ss in ss_list:
                    ss = ss[1:-1]
                    if "|" in ss:
                        final_res_tokens.append(alter_tokens(ss.split("|")))
                    else:
                        final_res_tokens.append(alter_tokens([ss]))
                for ele in itertools.product(*final_res_tokens):
                    sub_processed_list.append(ele)
            processed_list.append(sub_processed_list)
        else:
            split_list = tokens.split("]")

            final_res_tokens = list()
            for splits in split_list:
                if not splits.startswith("["):
                    splits = "[" + splits
                if not splits.endswith("]"):
                    splits = splits + "]"
                ss_list = re.findall(r"\[.*?\]", splits)
                for ss in ss_list:
                    ss = ss[1:-1]
                    if "|" in ss:
                        final_res_tokens.append(alter_tokens(ss.split("|")))
                    else:
                        final_res_tokens.append(alter_tokens([ss]))
            sub_processed_list = list()
            for ele in itertools.product(*final_res_tokens):
                sub_processed_list.append(ele)
            processed_list.append(sub_processed_list)

    final_list = list()
    for ele_list in itertools.product(*processed_list):
        if ele_list:
            sub_list = list()
            for ele in ele_list:
                if ele:
                    sub_list += [e for e in ele if e]
            if sub_list:
                final_list.append(sub_list)

    return final_list


def load_df_raw(faq_full_file):
    query_list = list()

    this_df = pandas.read_excel(faq_full_file)
    this_df.fillna("", inplace=True)
    this_df = this_df.astype(str)
    logger.info(f"df_raw {this_df.shape}")

    l_cls_2 = ""
    l_cls_3 = ""
    l_cls_4 = ""
    l_attri = ""
    l_raw_q = ""
    l_idx = -1
    sample_list = list()
    l_query = None
    this_query_list = list()

    for idx, row in this_df.iterrows():
        raw_q = row["标准问题"]
        raw_a = row["标准答案"]

        cls_2 = row["分类2"]
        cls_3 = row["分类3"]
        cls_4 = row["分类4"]

        raw_exq = row["扩展问"]
        raw_exq_type = row["模板类型"]
        attri = row["属性"]
        sample = row["测试样例"]
        channel = row["维度"]
        end_dt = row["有效时间止"]

        if cls_2:
            l_cls_2 = cls_2
            l_cls_3 = cls_3
            l_cls_4 = cls_4

        if raw_q and raw_a:
            l_idx = idx
            l_raw_q = raw_q
            l_attri = attri

        if "民生微信" in channel and not raw_exq_type and not sample:
            l_query = StandardQuery(l_idx, l_raw_q, raw_a)
            l_query.attri = l_attri
            l_query.ref = faq_full_file

            if end_dt:
                end_dt = str(int(float(end_dt)))
                l_query.end_dt = end_dt

            if l_cls_2:
                l_query.cls_list.append(l_cls_2)
            if l_cls_3:
                l_query.cls_list.append(l_cls_3)
            if l_cls_4:
                l_query.cls_list.append(l_cls_4)

            this_query_list.append(l_query)

        elif raw_exq_type == "0.0" and not sample:
            l_query.extend_list.append(raw_exq)

            sample_list = list()

        elif (raw_exq_type == "1.0" or raw_exq_type == "2.0") and not sample:
            l_query.extend_reg_dict[raw_exq] = process_sys_reg(raw_exq)

            sample_list = list()
            l_query.sample_list_of_list.append(sample_list)

        elif sample:
            sample_list.append(sample)
        else:
            pass

    for q in this_query_list:
        if is_valid_std_query(q.raw_q, q.std_a):
            query_list.append(q)

    return query_list


def load_df_processed(faq_file):
    query_list = list()
    this_df = pandas.read_excel(faq_file)
    this_df.set_index('序号')
    this_df.fillna("", inplace=True)
    this_df = this_df.astype(str)
    logger.info(f"this_df {this_df.shape}")
    for idx, row in this_df.iterrows():
        raw_q = row["标准问题"]
        raw_a = row["标准答案"]
        cls_2 = row["归类"]
        attri = row["意图"]
        ref = row["引用"]

        if is_valid_std_query(raw_q, raw_a):
            l_query = StandardQuery(idx, raw_q, raw_a)
            l_query.attri = attri
            l_query.ref = ref

            if cls_2:
                l_query.cls_list.append(cls_2)

            query_list.append(l_query)

    return query_list


def load_faq(faq_filepath, is_processed):
    if is_processed:
        raw_query_obj_list = load_df_processed(faq_filepath)
    else:
        raw_query_obj_list = load_df_raw(faq_filepath)

    conflict_list = list()

    as_count_list = list()
    label_dict = dict()
    label_list = list()
    std_label_list = list()
    answer_list = list()
    ref_list = list()

    for ix, std_query_obj in enumerate(raw_query_obj_list):
        if std_query_obj.raw_q not in std_label_list:
            std_label_list.append(std_query_obj.raw_q)
        else:
            assert False, std_query_obj.raw_q

        for raw_ext_q in std_query_obj.extend_list:
            conflict_list.append(raw_ext_q)
        for ext_q, seg_ext_q_list in std_query_obj.extend_reg_dict.items():
            for seg_ext_q in seg_ext_q_list:
                raw_seg_q = "".join(seg_ext_q)
                conflict_list.append(raw_seg_q)
        for samples in std_query_obj.sample_list_of_list:
            for sample in samples:
                conflict_list.append(sample)

    logger.info(f"std_label_list len {len(std_label_list)}")

    count_res = Counter(conflict_list)
    duplicates = list()
    for k, v in count_res.items():
        if v > 1:
            duplicates.append(k)
    conflict_list = duplicates + std_label_list

    for ix, std_query_obj in enumerate(raw_query_obj_list):
        label_dict[len(label_list)] = ix
        label_list.append(std_query_obj.raw_q)
        ref_list.append(std_query_obj.ref)
        answer_list.append(std_query_obj.std_a)
        as_count_list.append(ix)

        for raw_ext_q in std_query_obj.extend_list:
            if raw_ext_q not in conflict_list:
                label_dict[len(label_list)] = ix
                label_list.append(raw_ext_q)
                ref_list.append(std_query_obj.ref)
                as_count_list.append(ix)

        for ext_q, seg_ext_q_list in std_query_obj.extend_reg_dict.items():
            for seg_ext_q in seg_ext_q_list:
                raw_seg_q = "".join(seg_ext_q)
                if raw_seg_q not in conflict_list:
                    label_dict[len(label_list)] = ix
                    label_list.append(raw_seg_q)
                    ref_list.append(std_query_obj.ref)
                    as_count_list.append(ix)

        for samples in std_query_obj.sample_list_of_list:
            for sample in samples:
                if sample not in conflict_list:
                    label_dict[len(label_list)] = ix
                    label_list.append(sample)
                    ref_list.append(std_query_obj.ref)
                    as_count_list.append(ix)

    return label_list, std_label_list, answer_list, label_dict, as_count_list, ref_list


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


class KBService(ABC):

    def __init__(self,
                 knowledge_base_name: str,
                 embed_model: str = EMBEDDING_MODEL,
                 search_enhance: bool = SEARCH_ENHANCE,
                 ):
        self.kb_name = knowledge_base_name
        self.kb_info = KB_INFO.get(knowledge_base_name, f"关于{knowledge_base_name}的知识库")
        self.embed_model = embed_model
        self.search_enhance = search_enhance
        self.kb_path = get_kb_path(self.kb_name)
        self.doc_path = get_doc_path(self.kb_name)
        self.do_init()

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
        if not os.path.exists(self.doc_path):
            os.makedirs(self.doc_path)

        self.do_create_kb("docs")
        self.do_create_kb("question")
        self.do_create_kb("answer")
        self.do_create_kb("query")

        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)
        return status

    def clear_vs(self):
        """
        删除向量库中所有内容
        """
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
        self.do_drop_kb()
        status = delete_kb_from_db(self.kb_name)
        return status

    def _docs_to_embeddings(self, docs: List[Document]) -> Dict:
        '''
        将 List[Document] 转化为 VectorStore.add_embeddings 可以接受的参数
        '''
        return embed_documents(docs=docs, embed_model=self.embed_model, to_query=False)

    def add_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        向知识库添加文件
        如果指定了docs，则不再将文本向量化，并将数据库对应条目标为custom_docs=True
        """
        if docs:
            custom_docs = True
            for doc in docs:
                doc.metadata.setdefault("source", kb_file.filename)
        else:
            docs = kb_file.file2text()
            custom_docs = False

        if docs:
            # 将 metadata["source"] 改为相对路径
            for doc in docs:
                try:
                    source = doc.metadata.get("source", "")
                    rel_path = Path(source).relative_to(self.doc_path)
                    doc.metadata["source"] = str(rel_path.as_posix().strip("/"))
                except Exception as e:
                    print(f"cannot convert absolute path ({source}) to relative path. error is : {e}")
            self.delete_doc(kb_file)
            doc_infos = self.do_add_doc(docs, **kwargs)
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
        self.do_delete_doc(kb_file, **kwargs)
        status = delete_file_from_db(kb_file)
        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)
        return status

    def add_faq(self, kb_file: KnowledgeFile, is_generated, **kwargs):
        """
        向知识库添加文件
        如果指定了docs，则不再将文本向量化，并将数据库对应条目标为custom_docs=True
        """

        print("add_faq")
        print(f"is_generated {is_generated}")

        filepath = kb_file.filepath

        label_list, std_label_list, answer_list, label_dict, as_count_list, _ = load_faq(filepath,
                                                                                         is_processed=is_generated)

        self.delete_faq(kb_file, **kwargs)

        status = add_file_to_db(kb_file)

        answer_dict = dict()
        for idx, question in enumerate(label_list):
            a_idx = as_count_list[idx]
            answer = answer_list[a_idx]

            docq = Document(page_content=question)
            docq.metadata["source"] = filepath
            docq.metadata["raw_id"] = idx
            docq.metadata["answer_id"] = a_idx

            if a_idx in answer_dict:
                answer_dict[a_idx].append(docq)
            else:
                qna = "问题：" + std_label_list[a_idx] + "\n答案：" + answer
                doca = Document(page_content=qna)
                doca.metadata["source"] = filepath
                doca.metadata["raw_id"] = a_idx

                answer_dict[a_idx] = [doca, docq]

        for a_idx, ele_list in answer_dict.items():
            answer_obj = ele_list[0]
            answers = [answer_obj]
            questions = ele_list[1:]

            doc_infos = self.do_add_answer(answers, **kwargs)

            answer_id_list = [a.metadata["raw_id"] for a in answers]
            file_name_list = [a.metadata["source"] for a in answers]

            this_status = add_answer_to_db(self.kb_name, file_name_list, answer_id_list, doc_infos=doc_infos)

            status = this_status & status

            question_id_list = [q.metadata["raw_id"] for q in questions]

            doc_infos = self.do_add_question(questions, **kwargs)

            this_status = add_question_to_db(self.kb_name, answer_id_list[0], question_id_list, doc_infos=doc_infos)

            status = this_status & status

        return status

    def delete_faq(self, kb_file: KnowledgeFile, delete_content: bool = False, **kwargs):
        """
        从知识库删除文件
        """

        self.do_delete_answer(kb_file, **kwargs)
        self.do_delete_question(kb_file, **kwargs)

        status = delete_file_from_db(kb_file)
        if delete_content and os.path.exists(kb_file.filepath):
            os.remove(kb_file.filepath)
        return status

    def update_info(self, kb_info: str):
        """
        更新知识库介绍
        """
        self.kb_info = kb_info
        status = add_kb_to_db(self.kb_name, self.kb_info, self.vs_type(), self.embed_model)
        return status

    def update_faq(self, kb_file: KnowledgeFile, is_generated, **kwargs):
        """
        使用content中的文件更新向量库
        如果指定了docs，则使用自定义docs，并将数据库对应条目标为custom_docs=True
        """
        if os.path.exists(kb_file.filepath):
            self.delete_faq(kb_file, **kwargs)
            return self.add_faq(kb_file, is_generated, **kwargs)

    def update_doc(self, kb_file: KnowledgeFile, docs: List[Document] = [], **kwargs):
        """
        使用content中的文件更新向量库
        如果指定了docs，则使用自定义docs，并将数据库对应条目标为custom_docs=True
        """
        if os.path.exists(kb_file.filepath):
            self.delete_doc(kb_file, **kwargs)
            return self.add_doc(kb_file, docs=docs, **kwargs)

    def exist_doc(self, file_name: str):
        return file_exists_in_db(KnowledgeFile(knowledge_base_name=self.kb_name,
                                               filename=file_name))

    def list_files(self):
        return list_files_from_db(self.kb_name)

    def count_files(self):
        return count_files_from_db(self.kb_name)

    def search_docs(self,
                    query: str,
                    top_k: int = VECTOR_SEARCH_TOP_K,
                    score_threshold: float = SCORE_THRESHOLD,
                    embeddings: List[float] = None,
                    ):
        embedding, docs = self.do_search_docs(query, top_k, score_threshold, embeddings=embeddings)
        return embedding, docs

    def search_question(self,
                        query: str,
                        top_k: int = VECTOR_SEARCH_TOP_K,
                        score_threshold: float = SCORE_THRESHOLD,
                        embeddings: List[float] = None,
                        ):
        embedding, docs = self.do_search_question(query, top_k, score_threshold, embeddings=embeddings)
        return embedding, docs

    def search_answer(self,
                      query: str,
                      top_k: int = VECTOR_SEARCH_TOP_K,
                      score_threshold: float = SCORE_THRESHOLD,
                      embeddings: List[float] = None,
                      ):
        embedding, docs = self.do_search_answer(query, top_k, score_threshold, embeddings=embeddings)
        return embedding, docs

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        return []


    def get_answer_by_ids(self, ids: List[str]) -> List[Document]:
        return []

    def get_question_by_ids(self, ids: List[str]) -> List[Document]:
        return []

    def list_docs(self, file_name: str = None, metadata: Dict = {}) -> List[DocumentWithVSId]:
        '''
        通过file_name或metadata检索Document
        '''
        doc_infos = list_docs_from_db(kb_name=self.kb_name, file_name=file_name, metadata=metadata)
        docs = []
        for x in doc_infos:
            doc_info_s = self.get_doc_by_ids([x["id"]])
            if doc_info_s is not None and doc_info_s != []:
                # 处理非空的情况
                doc_with_id = DocumentWithVSId(**doc_info_s[0].dict(), id=x["id"])
                docs.append(doc_with_id)
            else:
                # 处理空的情况
                # 可以选择跳过当前循环迭代或执行其他操作
                pass
        return docs

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
    def do_search_docs(self,
                       query: str,
                       top_k: int,
                       score_threshold: float,
                       embeddings: List[float] = None,
                       ) -> (List[float], List[Document]):
        """
        搜索知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_search_question(self,
                           query: str,
                           top_k: int,
                           score_threshold: float,
                           embeddings: List[float] = None,
                           ) -> (List[float], List[Document]):
        """
        搜索知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_search_answer(self,
                         query: str,
                         top_k: int,
                         score_threshold: float,
                         embeddings: List[float] = None,
                         ) -> (List[float], List[Document]):
        """
        搜索知识库子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_add_doc(self,
                   docs: List[Document],
                   ) -> List[Dict]:
        """
        向知识库添加文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_delete_doc(self,
                      kb_file: KnowledgeFile):
        """
        从知识库删除文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_add_question(self,
                        docs: List[Document],
                        ) -> List[Dict]:
        """
        向知识库添加文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_delete_question(self,
                           kb_file: KnowledgeFile):
        """
        从知识库删除文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_add_answer(self,
                      docs: List[Document],
                      ) -> List[Dict]:
        """
        向知识库添加文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_delete_answer(self,
                         kb_file: KnowledgeFile):
        """
        从知识库删除文档子类实自己逻辑
        """
        pass

    @abstractmethod
    def do_clear_vs(self, vector_name):
        """
        从知识库删除全部向量子类实自己逻辑
        """
        pass


class KBServiceFactory:

    @staticmethod
    def get_service(kb_name: str,
                    vector_store_type: Union[str, SupportedVSType],
                    embed_model: str = EMBEDDING_MODEL,
                    search_enhance: bool = SEARCH_ENHANCE,
                    ) -> KBService:
        if isinstance(vector_store_type, str):
            vector_store_type = getattr(SupportedVSType, vector_store_type.upper())
        if SupportedVSType.FAISS == vector_store_type:
            from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
            return FaissKBService(kb_name, embed_model=embed_model, search_enhance=search_enhance)
        elif SupportedVSType.PG == vector_store_type:
            from server.knowledge_base.kb_service.pg_kb_service import PGKBService
            return PGKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.MILVUS == vector_store_type:
            from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
            return MilvusKBService(kb_name,embed_model=embed_model)
        elif SupportedVSType.ZILLIZ == vector_store_type:
            from server.knowledge_base.kb_service.zilliz_kb_service import ZillizKBService
            return ZillizKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.DEFAULT == vector_store_type:
            from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
            return FaissKBService(kb_name, embed_model=embed_model, search_enhance=search_enhance)
        elif SupportedVSType.ES == vector_store_type:
            from server.knowledge_base.kb_service.es_kb_service import ESKBService
            return ESKBService(kb_name, embed_model=embed_model)
        elif SupportedVSType.DEFAULT == vector_store_type:  # kb_exists of default kbservice is False, to make validation easier.
            from server.knowledge_base.kb_service.default_kb_service import DefaultKBService
            return DefaultKBService(kb_name)

    @staticmethod
    def get_service_by_name(kb_name: str) -> KBService:
        _, vs_type, embed_model = load_kb_from_db(kb_name)
        if _ is None:  # kb not in db, just return None
            return None
        return KBServiceFactory.get_service(kb_name, vs_type, embed_model)

    @staticmethod
    def get_default():
        return KBServiceFactory.get_service("default", SupportedVSType.DEFAULT)


def get_kb_details() -> List[Dict]:
    kbs_in_folder = list_kbs_from_folder()
    kbs_in_db = KBService.list_kbs()
    result = {}

    for kb in kbs_in_folder:
        result[kb] = {
            "kb_name": kb,
            "vs_type": "",
            "kb_info": "",
            "embed_model": "",
            "file_count": 0,
            "create_time": None,
            "in_folder": True,
            "in_db": False,
        }

    for kb in kbs_in_db:
        kb_detail = get_kb_detail(kb)
        if kb_detail:
            kb_detail["in_db"] = True
            if kb in result:
                result[kb].update(kb_detail)
            else:
                kb_detail["in_folder"] = False
                result[kb] = kb_detail

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

    for doc in files_in_folder:
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
            "in_db": False,
        }
    for doc in files_in_db:
        doc_detail = get_file_detail(kb_name, doc)
        if doc_detail:
            doc_detail["in_db"] = True
            if doc in result:
                result[doc].update(doc_detail)
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

    # TODO: 暂不支持异步
    # async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
    #     return normalize(await self.embeddings.aembed_documents(texts))

    # async def aembed_query(self, text: str) -> List[float]:
    #     return normalize(await self.embeddings.aembed_query(text))


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
