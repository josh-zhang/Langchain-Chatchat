import os

from langchain.retrievers import BM25Retriever

from server.knowledge_base.kb_cache.base import *
from configs import CACHED_VS_NUM, CACHED_MEMO_VS_NUM


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
        vector_name: str,
        doc_list,
        top_k: int,
    ) -> ThreadSafeBM25:
        self.atomic.acquire()

        print(f"load_retriever {kb_name} {vector_name}")

        cache = self.get((kb_name, vector_name))
        if cache is None:
            item = ThreadSafeBM25((kb_name, vector_name), pool=self)
            self.set((kb_name, vector_name), item)
            with item.acquire(msg="初始化"):
                self.atomic.release()
                logger.info(f"loading retriever in '{kb_name} {vector_name}' to memory.")
                # create an empty vector store
                bm25_retriever = BM25Retriever.from_texts(doc_list)
                bm25_retriever.k = top_k
                item.obj = bm25_retriever
                item.finish_loading()
        else:
            self.atomic.release()
        return self.get((kb_name, vector_name))


kb_bm25_pool = KBBM25Pool(cache_num=CACHED_VS_NUM)