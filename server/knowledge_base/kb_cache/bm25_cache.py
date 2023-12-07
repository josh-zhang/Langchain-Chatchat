import os
import re
import functools
from LAC import LAC

from langchain.retrievers import BM25Retriever
from langchain.schema import Document
from server.knowledge_base.kb_cache.base import *
from configs import CACHED_VS_NUM

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


@functools.lru_cache()
def preprocess_func(sentence: str) -> List[str]:
    sentence = re.sub("[^A-Za-z0-9\u4e00-\u9fa5]", "_", sentence)
    sentence = re.sub("_+", " ", sentence)
    sentence = re.sub(" {2,}", " ", sentence).strip()
    sentence = sentence.strip().upper()

    seg_sentence = lac.run([sentence])[0]

    return [j.strip() for j in seg_sentence if j.strip()]


def norm(score_list: List[float]) -> List[float]:
    ma = max(score_list)
    mi = min(score_list)
    diff = ma - mi

    if diff == 0:
        score_list = [0.0 for _ in score_list]
    else:
        score_list = [(i - mi) / diff for i in score_list]

    return score_list


def get_score(retriever: BM25Retriever, query: str) -> (List[Document], List[float]):
    processed_query = preprocess_func(query)
    scores = retriever.vectorizer.get_scores(processed_query)
    scores = norm(scores)
    return scores


class ThreadSafeBM25(ThreadSafeObject):
    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}, docs_count: {self.docs_count()}>"

    def docs_count(self) -> int:
        return len(self._obj.docstore._dict)

    def save(self, path: str, create_path: bool = True):
        with self.acquire():
            if not os.path.isdir(path) and create_path:
                os.makedirs(path)
            ret = self._obj.save_local(path)
            logger.info(f"已将向量库 {self.key} 保存到磁盘")
        return ret

    def clear(self):
        ret = []
        with self.acquire():
            ids = list(self._obj.docstore._dict.keys())
            if ids:
                ret = self._obj.delete(ids)
                assert len(self._obj.docstore._dict) == 0
            logger.info(f"已将向量库 {self.key} 清空")
        return ret


class KBBM25Pool(CachePool):

    def load_retriever(
            self,
            kb_name: str,
            retriever_name: str,
            file_md5_sum,
            docs_text_list,
            metadata_list,
            # top_k: int,
    ) -> ThreadSafeBM25:
        self.atomic.acquire()

        print(f"load_retriever {kb_name} {retriever_name} {file_md5_sum}")

        cache = self.get((kb_name, retriever_name, file_md5_sum))
        if cache is None:
            item = ThreadSafeBM25((kb_name, retriever_name, file_md5_sum), pool=self)
            self.set((kb_name, retriever_name, file_md5_sum), item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading retriever in '{kb_name} {retriever_name}' to memory.")
                bm25_retriever = BM25Retriever.from_texts(docs_text_list, metadata_list,
                                                          preprocess_func=preprocess_func)
                # bm25_retriever.k = top_k
                item.obj = bm25_retriever
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get((kb_name, retriever_name, file_md5_sum))


kb_bm25_pool = KBBM25Pool(cache_num=CACHED_VS_NUM)
