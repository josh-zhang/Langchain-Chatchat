import shutil
from typing import List, Dict

from langchain.schema import Document
from langchain.vectorstores.milvus import Milvus

from configs import kbs_config, MILVUS_NPROBE
from server.db.repository import list_file_num_docs_id_by_kb_name_and_file_name, \
    list_file_num_answer_id_by_kb_name_and_file_name, \
    list_file_num_question_id_by_kb_name_and_file_name

from server.knowledge_base.kb_service.base import KBService, SupportedVSType, EmbeddingsFunAdapter
from server.knowledge_base.utils import KnowledgeFile, DocumentWithScores


class MilvusKBService(KBService):
    milvus_d: Milvus
    milvus_q: Milvus
    milvus_a: Milvus

    def vs_type(self) -> str:
        return SupportedVSType.MILVUS

    def get_doc_by_ids(self, vector_name, ids: List[str], fields: List[str]) -> List[Document]:
        result = []
        if self.get_milvus(vector_name).col:
            # ids = [int(id) for id in ids]  # for milvus if needed #pr 2725
            data_list = self.get_milvus(vector_name).col.query(expr=f'pk in {[int(_id) for _id in ids]}',
                                                               output_fields=fields)
            for data in data_list:
                text = data.pop("text")
                result.append(Document(page_content=text, metadata=data))
        return result

    def get_docs_by_file_name(self, vector_name, file_name, fields: List[str]) -> List[Document]:
        result = []
        if self.get_milvus(vector_name).col:
            data_list = self.get_milvus(vector_name).col.query(expr=f'source == "{file_name}"',
                                                               output_fields=fields)
            for data in data_list:
                text = data.pop("text")
                result.append(Document(page_content=text, metadata=data))
        return result

    def do_init(self):
        self._load_milvus("docs")
        self._load_milvus("question")
        self._load_milvus("answer")

    def do_create_kb(self, vector_name):
        pass

    def do_drop_kb(self):
        if self.get_milvus("docs").col:
            self.get_milvus("docs").col.release()
            self.get_milvus("docs").col.drop()
        if self.get_milvus("question").col:
            self.get_milvus("question").col.release()
            self.get_milvus("question").col.drop()
        if self.get_milvus("answer").col:
            self.get_milvus("answer").col.release()
            self.get_milvus("answer").col.drop()
        try:
            shutil.rmtree(self.kb_path)
        except Exception:
            ...

    def do_search(self, vector_name: str, query: str, top_k: int, score_threshold: float,
                  embeddings: List[float] = None):
        self._load_milvus(vector_name)

        if embeddings is None:
            embed_func = EmbeddingsFunAdapter(self.embed_model)
            embeddings = embed_func.embed_query(query)

        docs = self.get_milvus(vector_name).similarity_search_with_score_by_vector(embeddings, top_k)
        # docs = score_threshold_process(score_threshold, top_k, docs)
        docs = [DocumentWithScores(**d.dict(), scores={f"sbert_{vector_name}": s}) for d, s in docs]

        return embeddings, docs

    def do_add_doc(self, vector_name: str, docs: List[Document], **kwargs) -> List[Dict]:
        for doc in docs:
            for k, v in doc.metadata.items():
                doc.metadata[k] = str(v)
            for field in self.get_milvus(vector_name).fields:
                doc.metadata.setdefault(field, "")
            doc.metadata.pop(self.get_milvus(vector_name)._text_field, None)
            doc.metadata.pop(self.get_milvus(vector_name)._vector_field, None)

        ids = self.get_milvus(vector_name).add_documents(docs)

        doc_infos = [{"id": id, "metadata": doc.metadata} for id, doc in zip(ids, docs)]
        return doc_infos

    def do_delete_doc(self, vector_name, kb_file: KnowledgeFile, **kwargs):
        id_list = list_file_num_docs_id_by_kb_name_and_file_name(kb_file.kb_name, kb_file.filename)
        if self.get_milvus(vector_name).col:
            self.get_milvus(vector_name).col.delete(expr=f'pk in {id_list}')

    def do_clear_vs(self, vector_name):
        if self.get_milvus(vector_name).col:
            self.get_milvus(vector_name).col.release()
            self.get_milvus(vector_name).col.drop()
            self._load_milvus(vector_name)

    @staticmethod
    def get_collection(milvus_name):
        from pymilvus import Collection
        return Collection(milvus_name)

    @staticmethod
    def search(milvus_name, content, limit=3):
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": MILVUS_NPROBE},
        }
        c = MilvusKBService.get_collection(milvus_name)
        return c.search(content, "embeddings", search_params, limit=limit, output_fields=["content"])

    def get_milvus(self, vector_name):
        if vector_name == "question":
            return self.milvus_q
        elif vector_name == "answer":
            return self.milvus_a
        else:
            return self.milvus_d

    def _load_milvus(self, vector_name):
        col_name = f"{vector_name}_{self.kb_name}"
        if vector_name == "question":
            self.milvus_q = Milvus(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                                   collection_name=col_name,
                                   connection_args=kbs_config.get("milvus"),
                                   index_params=kbs_config.get("milvus_kwargs")["index_params"],
                                   search_params=kbs_config.get("milvus_kwargs")["search_params"],
                                   auto_id=True
                                   )
        elif vector_name == "answer":
            self.milvus_a = Milvus(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                                   collection_name=col_name,
                                   connection_args=kbs_config.get("milvus"),
                                   index_params=kbs_config.get("milvus_kwargs")["index_params"],
                                   search_params=kbs_config.get("milvus_kwargs")["search_params"],
                                   auto_id=True
                                   )
        else:
            self.milvus_d = Milvus(embedding_function=EmbeddingsFunAdapter(self.embed_model),
                                   collection_name=col_name,
                                   connection_args=kbs_config.get("milvus"),
                                   index_params=kbs_config.get("milvus_kwargs")["index_params"],
                                   search_params=kbs_config.get("milvus_kwargs")["search_params"],
                                   auto_id=True
                                   )

    def del_doc_by_ids(self, vector_name, ids: List[str]):
        self.get_milvus(vector_name).col.delete(expr=f'pk in {[int(_id) for _id in ids]}')

    def count_docs(self, vector_name: str, filename: str):
        if vector_name == "question":
            count = len(list_file_num_question_id_by_kb_name_and_file_name(self.kb_name, filename))
        elif vector_name == "answer":
            count = len(list_file_num_answer_id_by_kb_name_and_file_name(self.kb_name, filename))
        else:
            count = len(list_file_num_docs_id_by_kb_name_and_file_name(self.kb_name, filename))

        return count


if __name__ == '__main__':
    # 测试建表使用
    from server.db.base import Base, engine

    Base.metadata.create_all(bind=engine)
    milvusService = MilvusKBService("test")
    # milvusService.add_doc(KnowledgeFile("README.md", "test"))

    print(milvusService.get_doc_by_ids(["444022434274215486"]))
    # milvusService.delete_doc(KnowledgeFile("README.md", "test"))
    # milvusService.do_drop_kb()
    # print(milvusService.search_docs("如何启动api服务"))
