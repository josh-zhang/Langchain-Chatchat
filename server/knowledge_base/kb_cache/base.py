import threading
from typing import List, Any, Union, Tuple, Optional
from collections import OrderedDict
from contextlib import contextmanager

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.faiss import FAISS

from configs import EMBEDDING_MODEL, CACHED_EMBED_NUM, logger, log_verbose, RERANKER_MODEL, MODEL_PATH, \
    CACHED_RERANK_NUM, RERANKER_MAX_LENGTH
from server.utils import embedding_device, get_model_path


class ThreadSafeObject:
    def __init__(self, key: Union[str, Tuple], obj: Any = None, pool: "CachePool" = None):
        self._obj = obj
        self._key = key
        self._pool = pool
        self._lock = threading.RLock()
        self._loaded = threading.Event()

    def __repr__(self) -> str:
        cls = type(self).__name__
        return f"<{cls}: key: {self.key}, obj: {self._obj}>"

    @property
    def key(self):
        return self._key

    @contextmanager
    def acquire(self, owner: str = "", msg: str = "") -> FAISS:
        owner = owner or f"thread {threading.get_native_id()}"
        try:
            self._lock.acquire()
            if self._pool is not None:
                self._pool._cache.move_to_end(self.key)
            if log_verbose:
                logger.info(f"{owner} 开始操作：{self.key}。{msg}")
            yield self._obj
        finally:
            if log_verbose:
                logger.info(f"{owner} 结束操作：{self.key}。{msg}")
            self._lock.release()

    def start_loading(self):
        self._loaded.clear()

    def finish_loading(self):
        self._loaded.set()

    def wait_for_loading(self):
        self._loaded.wait()

    @property
    def obj(self):
        return self._obj

    @obj.setter
    def obj(self, val: Any):
        self._obj = val


class CachePool:
    def __init__(self, cache_num: int = -1):
        self._cache_num = cache_num
        self._cache = OrderedDict()
        self.atomic = threading.RLock()

    def keys(self) -> List[str]:
        return list(self._cache.keys())

    def _check_count(self):
        if isinstance(self._cache_num, int) and self._cache_num > 0:
            while len(self._cache) > self._cache_num:
                self._cache.popitem(last=False)

    def get(self, key: str) -> Optional[ThreadSafeObject]:
        if cache := self._cache.get(key):
            cache.wait_for_loading()
            return cache
        else:
            return None

    def set(self, key: str, obj: ThreadSafeObject) -> ThreadSafeObject:
        self._cache[key] = obj
        self._check_count()
        return obj

    def pop(self, key: str = None) -> ThreadSafeObject:
        if key is None:
            return self._cache.popitem(last=False)
        else:
            return self._cache.pop(key, None)

    def acquire(self, key: Union[str, Tuple], owner: str = "", msg: str = ""):
        cache = self.get(key)
        if cache is None:
            raise RuntimeError(f"请求的资源 {key} 不存在")
        elif isinstance(cache, ThreadSafeObject):
            self._cache.move_to_end(key)
            return cache.acquire(owner=owner, msg=msg)
        else:
            return cache

    def load_kb_embeddings(
            self,
            kb_name: str,
            embed_device: str = embedding_device(),
            default_embed_model: str = EMBEDDING_MODEL,
    ) -> Embeddings:
        from server.db.repository.knowledge_base_repository import get_kb_detail
        from server.knowledge_base.kb_service.base import EmbeddingsFunAdapter

        kb_detail = get_kb_detail(kb_name)
        embed_model = kb_detail.get("embed_model", default_embed_model)

        if embed_model.endswith("-api"):
            return EmbeddingsFunAdapter(embed_model)
        else:
            return embeddings_pool.load_embeddings(model=embed_model, device=embed_device, normalize_embeddings=False)


class EmbeddingsPool(CachePool):

    def load_embeddings(self, model: str = None, device: str = None, normalize_embeddings=False) -> Embeddings:
        self.atomic.acquire()
        model = model or EMBEDDING_MODEL
        device = embedding_device()
        key = model + "_" + device + "_" + normalize_embeddings
        cache = self.get(key)
        if cache is None:
            item = ThreadSafeObject(key, pool=self)
            self.set(key, item)
            with item.acquire():
                self.atomic.release()
                if 'bge-' in model:
                    from langchain_community.embeddings import HuggingFaceBgeEmbeddings
                    if 'zh' in model:
                        # for chinese model
                        query_instruction = "为这个句子生成表示以用于检索相关文章："
                    elif 'en' in model:
                        # for english model
                        query_instruction = "Represent this sentence for searching relevant passages:"
                    else:
                        # maybe ReRanker or else, just use empty string instead
                        query_instruction = ""
                    embeddings = HuggingFaceBgeEmbeddings(model_name=get_model_path(model),
                                                          model_kwargs={'device': device},
                                                          encode_kwargs={'normalize_embeddings': normalize_embeddings},
                                                          query_instruction=query_instruction)
                    if model == "bge-large-zh-noinstruct":  # bge large -noinstruct embedding
                        embeddings.query_instruction = ""
                else:
                    from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
                    embeddings = HuggingFaceEmbeddings(model_name=get_model_path(model),
                                                       encode_kwargs={'normalize_embeddings': normalize_embeddings},
                                                       model_kwargs={'device': device})
                item.obj = embeddings
                item.finish_loading()
            return embeddings
        else:
            self.atomic.release()
            return cache.obj


class RerankerPool(CachePool):
    def load_reranker(self, model: str = None):
        model = model or RERANKER_MODEL
        reranker_model_path = MODEL_PATH["reranker"].get(model, "/opt/projects/hf_models/bge-reranker-v2-m3")

        key = model

        self.atomic.acquire()

        cache = self.get(key)

        if cache is None:
            item = ThreadSafeObject(key, pool=self)
            self.set(key, item)
            with item.acquire():
                self.atomic.release()

                tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
                model = AutoModelForSequenceClassification.from_pretrained(reranker_model_path)
                model.eval()

                item.obj = (tokenizer, model)
                item.finish_loading()
            return tokenizer, model
        else:
            self.atomic.release()
            return cache.obj

    def get_score(self, sentence_pairs, model: str = None):
        model = model or RERANKER_MODEL
        reranker_model_path = MODEL_PATH["reranker"].get(model, "/opt/projects/hf_models/bge-reranker-v2-m3")

        key = model

        self.atomic.acquire()

        cache = self.get(key)

        if cache is None:
            item = ThreadSafeObject(key, pool=self)
            self.set(key, item)
            with item.acquire():
                self.atomic.release()

                tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
                model = AutoModelForSequenceClassification.from_pretrained(reranker_model_path)
                model.eval()

                item.obj = (tokenizer, model)
                item.finish_loading()

                with torch.no_grad():
                    inputs = tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors='pt',
                                       max_length=RERANKER_MAX_LENGTH)
                    scores = model(**inputs, return_dict=True).logits.view(-1, ).float().tolist()
            return scores
        else:
            self.atomic.release()

            tokenizer, model = cache.obj
            with torch.no_grad():
                inputs = tokenizer(sentence_pairs, padding=True, truncation=True, return_tensors='pt',
                                   max_length=RERANKER_MAX_LENGTH)
                scores = model(**inputs, return_dict=True).logits.view(-1, ).float().tolist()
            return scores


embeddings_pool = EmbeddingsPool(cache_num=CACHED_EMBED_NUM)
reranker_pool = RerankerPool(cache_num=CACHED_RERANK_NUM)
