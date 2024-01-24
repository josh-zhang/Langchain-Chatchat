import os
import shutil
import logging

from configs import SCORE_THRESHOLD
from server.knowledge_base.kb_service.base import KBService, SupportedVSType, EmbeddingsFunAdapter
from server.knowledge_base.kb_cache.faiss_cache import kb_faiss_pool, ThreadSafeFaiss
from server.knowledge_base.utils import KnowledgeFile, get_kb_path, get_vs_path, DocumentWithScores
from server.utils import torch_gc
from langchain.docstore.document import Document
from typing import List, Dict


class FaissKBService(KBService):
    kb_path: str

    def vs_type(self) -> str:
        return SupportedVSType.FAISS

    def get_vs_path(self, vector_name):
        return get_vs_path(self.kb_name, vector_name)

    def get_kb_path(self):
        return get_kb_path(self.kb_name)

    def load_vector_store(self, vector_name) -> ThreadSafeFaiss:
        return kb_faiss_pool.load_vector_store(kb_name=self.kb_name,
                                               vector_name=vector_name,
                                               embed_model=self.embed_model)

    def save_vector_store(self, vector_name):
        vs_path = self.get_vs_path(vector_name)
        self.load_vector_store(vector_name).save(vs_path)

    def get_doc_by_ids(self, ids: List[str]) -> List[Document]:
        with self.load_vector_store("docs").acquire() as vs:
            return [vs.docstore._dict.get(id) for id in ids]

    def get_question_by_ids(self, ids: List[str]) -> List[Document]:
        with self.load_vector_store("question").acquire() as vs:
            return [vs.docstore._dict.get(id) for id in ids]

    def get_answer_by_ids(self, ids: List[str]) -> List[Document]:
        with self.load_vector_store("answer").acquire() as vs:
            return [vs.docstore._dict.get(id) for id in ids]

    def do_init(self):
        # self.vector_name = self.vector_name or self.embed_model
        self.kb_path = self.get_kb_path()
        # self.vs_path = self.get_vs_path()

    def do_create_kb(self, vector_name):
        vs_path = self.get_vs_path(vector_name)
        if not os.path.exists(vs_path):
            os.makedirs(vs_path)
        self.load_vector_store(vector_name)

    def do_drop_kb(self):
        self.clear_vs()
        try:
            shutil.rmtree(self.kb_path)
        except Exception:
            ...

    def do_search_docs(self,
                       query: str,
                       top_k: int,
                       score_threshold: float = SCORE_THRESHOLD,
                       embeddings: List[float] = None,
                       ) -> (List[float], List[DocumentWithScores]):
        if embeddings is None:
            embed_func = EmbeddingsFunAdapter(self.embed_model)
            embeddings = embed_func.embed_query(query)

        with self.load_vector_store("docs").acquire() as vs:
            if len(vs.docstore._dict) == 0:
                return embeddings, []
            score_threshold = 1 - score_threshold
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
            docs = [DocumentWithScores(**d.dict(), scores={"sbert_doc": 1 - s}) for d, s in docs]
            print(f"{len(docs)} docs found from faiss")
        return embeddings, docs

    def do_search_question(self,
                           query: str,
                           top_k: int,
                           score_threshold: float = SCORE_THRESHOLD,
                           embeddings: List[float] = None,
                           ) -> (List[float], List[DocumentWithScores]):
        if embeddings is None:
            embed_func = EmbeddingsFunAdapter(self.embed_model)
            embeddings = embed_func.embed_query(query)

        with self.load_vector_store("question").acquire() as vs:
            if len(vs.docstore._dict) == 0:
                return embeddings, []
            score_threshold = 1 - score_threshold
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
            docs = [DocumentWithScores(**d.dict(), scores={"sbert_que": 1 - s}) for d, s in docs]
            print(f"{len(docs)} question found from faiss")
        return embeddings, docs

    def do_search_answer(self,
                         query: str,
                         top_k: int,
                         score_threshold: float = SCORE_THRESHOLD,
                         embeddings: List[float] = None,
                         ) -> (List[float], List[DocumentWithScores]):
        if embeddings is None:
            embed_func = EmbeddingsFunAdapter(self.embed_model)
            embeddings = embed_func.embed_query(query)

        with self.load_vector_store("answer").acquire() as vs:
            if len(vs.docstore._dict) == 0:
                print(f"answer vector_store is empty")
                return embeddings, []
            score_threshold = 1 - score_threshold
            docs = vs.similarity_search_with_score_by_vector(embeddings, k=top_k, score_threshold=score_threshold)
            docs = [DocumentWithScores(**d.dict(), scores={"sbert_ans": 1 - s}) for d, s in docs]
            print(f"{len(docs)} answer found from faiss")
        return embeddings, docs

    def do_add_doc(self,
                   docs: List[Document],
                   **kwargs,
                   ) -> List[Dict]:

        data = self._docs_to_embeddings(docs)  # 将向量化单独出来可以减少向量库的锁定时间

        with self.load_vector_store("docs").acquire() as vs:
            ids = vs.add_embeddings(text_embeddings=zip(data["texts"], data["embeddings"]),
                                    metadatas=data["metadatas"],
                                    ids=kwargs.get("ids"))
            if not kwargs.get("not_refresh_vs_cache"):
                vs_path = self.get_vs_path("docs")
                vs.save_local(vs_path)
                print(f"{len(ids)} docs saved to faiss local")

        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        torch_gc()
        return doc_infos

    def do_delete_doc(self,
                      kb_file: KnowledgeFile,
                      **kwargs):
        vs_path = self.get_vs_path("docs")
        with self.load_vector_store("docs").acquire() as vs:
            ids = [k for k, v in vs.docstore._dict.items() if v.metadata.get("source") == kb_file.filename]
            if len(ids) > 0:
                vs.delete(ids)
            print(f"{len(ids)} docs deleted from faiss")
            if not kwargs.get("not_refresh_vs_cache"):
                vs.save_local(vs_path)
        return ids

    def do_add_question(self,
                        docs: List[Document],
                        **kwargs,
                        ) -> List[Dict]:
        data = self._docs_to_embeddings(docs)  # 将向量化单独出来可以减少向量库的锁定时间
        print(f"{len(docs)} question embedded")

        with self.load_vector_store("question").acquire() as vs:
            ids = vs.add_embeddings(text_embeddings=zip(data["texts"], data["embeddings"]),
                                    metadatas=data["metadatas"],
                                    ids=kwargs.get("ids"))

            print(f"add_question {len(data)}")

            if not kwargs.get("not_refresh_vs_cache"):
                vs_path = self.get_vs_path("question")
                vs.save_local(vs_path)

        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        torch_gc()
        return doc_infos

    def do_delete_question(self,
                           kb_file: KnowledgeFile,
                           **kwargs):
        vs_path = self.get_vs_path("question")
        with self.load_vector_store("question").acquire() as vs:
            ids = [k for k, v in vs.docstore._dict.items() if v.metadata.get("source") == kb_file.filename]
            if len(ids) > 0:
                vs.delete(ids)
            if not kwargs.get("not_refresh_vs_cache"):
                vs.save_local(vs_path)
        return ids

    def do_add_answer(self,
                      docs: List[Document],
                      **kwargs,
                      ) -> List[Dict]:
        data = self._docs_to_embeddings(docs)  # 将向量化单独出来可以减少向量库的锁定时间
        print(f"{len(docs)} answer embedded")

        with self.load_vector_store("answer").acquire() as vs:
            ids = vs.add_embeddings(text_embeddings=zip(data["texts"], data["embeddings"]),
                                    metadatas=data["metadatas"],
                                    ids=kwargs.get("ids"))

            if not kwargs.get("not_refresh_vs_cache"):
                vs_path = self.get_vs_path("answer")
                vs.save_local(vs_path)

        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        torch_gc()
        return doc_infos

    def do_delete_answer(self,
                         kb_file: KnowledgeFile,
                         **kwargs):
        vs_path = self.get_vs_path("answer")
        with self.load_vector_store("answer").acquire() as vs:
            ids = [k for k, v in vs.docstore._dict.items() if v.metadata.get("source") == kb_file.filename]
            if len(ids) > 0:
                vs.delete(ids)
            if not kwargs.get("not_refresh_vs_cache"):
                vs.save_local(vs_path)
        return ids

    def do_clear_vs(self, vector_name):
        vs_path = self.get_vs_path(vector_name)
        with kb_faiss_pool.atomic:
            kb_faiss_pool.pop((self.kb_name, vector_name))
        try:
            shutil.rmtree(vs_path)
        except Exception:
            ...
        os.makedirs(vs_path, exist_ok=True)

    def exist_doc(self, file_name: str):
        if super().exist_doc(file_name):
            return "in_db"

        content_path = os.path.join(self.kb_path, "content")
        if os.path.isfile(os.path.join(content_path, file_name)):
            return "in_folder"
        else:
            return False


if __name__ == '__main__':
    faissService = FaissKBService("test")
    faissService.add_doc(KnowledgeFile("README.md", "test"))
    faissService.delete_doc(KnowledgeFile("README.md", "test"))
    faissService.do_drop_kb()
    print(faissService.search_docs("如何启动api服务"))
