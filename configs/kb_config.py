import os

# 默认使用的知识库
DEFAULT_KNOWLEDGE_BASE = "samples"

# 默认向量库/全文检索引擎类型。可选：faiss, milvus(离线) & zilliz(在线), pgvector,全文检索引擎es
DEFAULT_VS_TYPE = "milvus"

# 缓存向量库数量（针对FAISS）
CACHED_VS_NUM = 20

# 缓存向量库数量（针对BM25）
CACHED_BM25_VS_NUM = 20

# 缓存向量库数量（针对EMBEDDING）
CACHED_EMBED_NUM = 1

# 缓存向量库数量（针对Reranker）
CACHED_RERANK_NUM = 1

# 缓存临时向量库数量（针对FAISS），用于文件对话
CACHED_MEMO_VS_NUM = 10

# 知识库中单段文本长度(不适用MarkdownHeaderTextSplitter)
CHUNK_SIZE = 800

# 知识库中相邻文本重合长度(不适用MarkdownHeaderTextSplitter)
OVERLAP_SIZE = 100

# 知识库匹配向量数量
VECTOR_SEARCH_TOP_K = 5

# 知识库匹配的距离阈值，取值范围在0-1之间，SCORE越小，距离越小从而相关度越高，
# 取到1相当于不筛选，实测bge-large的距离得分大部分在0.01-0.7之间，
# 相似文本的得分最高在0.55左右，因此建议针对bge设置得分为0.6
SCORE_THRESHOLD = 0.2

# 是否开启中文标题加强，以及标题增强的相关配置
# 通过增加标题判断，判断哪些文本为标题，并在metadata中进行标记；
# 然后将文本与往上一级的标题进行拼合，实现文本信息的增强。
ZH_TITLE_ENHANCE = False

# 每个知识库的初始化介绍，用于在初始化知识库时显示和Agent调用，没写则没有介绍，不会被Agent调用。
KB_INFO = {
    "知识库名称": "知识库介绍",
    "samples": "关于本项目issue的解答",
}

# PDF OCR 控制：只对宽高超过页面一定比例（图片宽/页面宽，图片高/页面高）的图片进行 OCR。
# 这样可以避免 PDF 中一些小图片的干扰，提高非扫描版 PDF 处理速度
PDF_OCR_THRESHOLD = (0.6, 0.6)

# 通常情况下不需要更改以下内容

# 知识库默认存储路径
KB_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")
if not os.path.exists(KB_ROOT_PATH):
    os.mkdir(KB_ROOT_PATH)
# 数据库默认存储路径。
# 如果使用sqlite，可以直接修改DB_ROOT_PATH；如果使用其它数据库，请直接修改SQLALCHEMY_DATABASE_URI。
DB_ROOT_PATH = os.path.join(KB_ROOT_PATH, "info.db")
SQLALCHEMY_DATABASE_URI = f"sqlite:///{DB_ROOT_PATH}"

# 可选向量库类型及对应配置
kbs_config = {
    "faiss": {
    },
    "milvus": {
        "host": "127.0.0.1",
        "port": "19530",
        "user": "",
        "password": "",
        "db_name": "",
        "secure": False,
    },
    # "pg": {
    #     "connection_uri": "postgresql://postgres:postgres@127.0.0.1:5432/langchain_chatchat",
    # },
    # "es": {
    #     "host": "127.0.0.1",
    #     "port": "9200",
    #     "index_name": "test_index",
    #     "user": "",
    #     "password": ""
    # },
    "milvus_kwargs": {
        "search_params": {"metric_type": "L2"},  # 在此处增加search_params
        "index_params": {"metric_type": "L2", "index_type": "HNSW", "efConstruction": 200, "M": 20}  # 在此处增加index_params
    }
}

# TextSplitter配置项，如果你不明白其中的含义，就不要修改。
text_splitter_dict = {
    "ChineseRecursiveTextSplitter": {
        "source": "huggingface",  # 选择tiktoken则使用openai的方法
        "tokenizer_name_or_path": "/opt/projects/hf_models/qwen1.5-14b-chat-gptq-int4",
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {
        "headers_to_split_on":
            [
                ("#", "head1"),
                ("##", "head2"),
                ("###", "head3"),
                ("####", "head4"),
            ]
    },
}

# TEXT_SPLITTER 名称
TEXT_SPLITTER_NAME = "ChineseRecursiveTextSplitter"

# Embedding模型定制词语的词表文件
EMBEDDING_KEYWORD_FILE = "embedding_keywords.txt"

SEARCH_ENHANCE = True

BM_25_FACTOR = 0.4

tokenizer_path_for_count_token = "/opt/projects/hf_models/qwen1.5-14b-chat-gptq-int4"

tokenizer_path_for_count_token_rerank = "/opt/projects/hf_models/bge-reranker-v2-m3"

MILVUS_NPROBE = 10
