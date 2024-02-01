import os

# 可以指定一个绝对路径，统一存放所有的Embedding和LLM模型。
# 每个模型可以是一个单独的目录，也可以是某个目录下的二级子目录。
# 如果模型目录名称和 MODEL_PATH 中的 key 或 value 相同，程序会自动检测加载，无需修改 MODEL_PATH 中的路径。
MODEL_ROOT_PATH = "/opt/projects/hf_models"

# 选用的 Embedding 名称
EMBEDDING_MODEL = "bge-m3"

# Embedding 模型运行设备。设为"auto"会自动检测，也可手动设定为"cuda","mps","cpu"其中之一。
EMBEDDING_DEVICE = "auto"

# 选用的reranker模型
RERANKER_MODEL = "bge-reranker-large"
# 是否启用reranker模型
USE_RERANKER = True
RERANKER_MAX_LENGTH = 1024

# 如果需要在 EMBEDDING_MODEL 中增加自定义的关键字时配置
EMBEDDING_KEYWORD_FILE = "keywords.txt"
EMBEDDING_MODEL_OUTPUT_PATH = "output"

# 要运行的 LLM 名称，可以包括本地模型和在线模型。列表中本地模型将在启动项目时全部加载。
# 列表中第一个模型将作为 API 和 WEBUI 的默认模型。
# 在这里，我们使用目前主流的两个离线模型，其中，chatglm3-6b 为默认加载模型。
# 如果你的显存不足，可使用 Qwen-1_8B-Chat, 该模型 FP16 仅需 3.8G显存。

# chatglm3-6b输出角色标签<|user|>及自问自答的问题详见项目wiki->常见问题->Q20.

LLM_MODELS = ["qwen-72b-chat-int4"]  # "Qwen-1_8B-Chat",

# AgentLM模型的名称 (可以不指定，指定之后就锁定进入Agent之后的Chain的模型，不指定就是LLM_MODELS[0])
Agent_MODEL = None

# LLM 运行设备。设为"auto"会自动检测，也可手动设定为"cuda","mps","cpu"其中之一。
LLM_DEVICE = "auto"

# 历史对话轮数
HISTORY_LEN = 3

# 大模型最长支持的长度，如果不填写，则使用模型默认的最大长度，如果填写，则为用户设定的最大长度
MAX_TOKENS = 5000

# LLM通用对话参数
TEMPERATURE = 0.0
TOP_P = 0.95  # ChatOpenAI暂不支持该参数


ONLINE_LLM_MODEL = {
    # 线上模型。请在server_config中为每个在线API设置不同的端口

    # "online-center-qwen14b-1": {
    #     "model_name": "Ywen-core",
    #     "api_base_url": "http://40.46.62.31:30151/v1",
    #     "api_key": "",
    #     "openai_proxy": "",
    # },

    # "online-center-qwen14b-2": {
    #     "model_name": "Ywen-core",
    #     "api_base_url": "http://40.46.62.31:30142/v1",
    #     "api_key": "",
    #     "openai_proxy": "",
    # },

    "online-qwen-14b": {
        "model_name": "Qwen-14B-Chat-Int4",
        "api_base_url": "http://198.203.120.8:40786/v3",
        "api_key": "",
        "openai_proxy": "",
    },

    "online-tongyi14b": {
        "model_name": "Tongyi-Finance-14B-Chat-Int4",
        "api_base_url": "http://198.203.120.8:40786/v3",
        "api_key": "",
        "openai_proxy": "",
    },

    "online-chatglm3-6b": {
        "model_name": "ChatGLM3-6B",
        "api_base_url": "http://198.203.120.8:40786/v3",
        "api_key": "",
        "openai_proxy": "",
    },

    "online-baichuan2-13b": {
        "model_name": "Baichuan2-13B-Chat-Int4",
        "api_base_url": "http://198.203.120.8:40786/v3",
        "api_key": "",
        "openai_proxy": "",
    }
}

# 在以下字典中修改属性值，以指定本地embedding模型存储位置。支持3种设置方法：
# 1、将对应的值修改为模型绝对路径
# 2、不修改此处的值（以 text2vec 为例）：
#       2.1 如果{MODEL_ROOT_PATH}下存在如下任一子目录：
#           - text2vec
#           - GanymedeNil/text2vec-large-chinese
#           - text2vec-large-chinese
#       2.2 如果以上本地路径不存在，则使用huggingface模型
MODEL_PATH = {
    "embed_model": {
        "bge-base-zh": "/opt/projects/hf_models/bge-base-zh",
        "bge-large-zh": "/opt/projects/hf_models/bge-large-zh",
        "bge-large-zh-v1.5": "/opt/projects/hf_models/bge-large-zh-v1.5",
        "bce-embedding-base_v1": "/opt/projects/hf_models/bce-embedding-base_v1",
        "m3e-base": "/opt/projects/hf_models/m3e-base",
        "stella-large-zh-v2": "/opt/projects/hf_models/stella-large-zh-v2",
        "jina-embeddings-v2-base-zh": "/opt/projects/hf_models/jina-embeddings-v2-base-zh",
        "bge-m3": "/opt/projects/hf_models/bge-m3",
        "dmeta-embedding": "/opt/projects/hf_models/Dmeta-embedding",
    },
    "reranker": {
        "bge-reranker-large": "/opt/projects/hf_models/bge-reranker-large",
        "bge-reranker-base": "/opt/projects/hf_models/bge-reranker-base",
        # TODO 增加在线reranker，如cohere
    }
}

# 通常情况下不需要更改以下内容

# nltk 模型存储路径
NLTK_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

# 你认为支持Agent能力的模型，可以在这里添加，添加后不会出现可视化界面的警告
# 经过我们测试，原生支持Agent的模型仅有以下几个
# SUPPORT_AGENT_MODEL = [
#     "azure-api",
#     "openai-api",
#     "qwen-api",
#     "Qwen",
#     "chatglm3",
#     "xinghuo-api",
# ]
