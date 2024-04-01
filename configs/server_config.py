import sys
from configs.model_config import LLM_DEVICE

# httpx 请求默认超时时间（秒）。如果加载模型或对话较慢，出现超时错误，可以适当加大该值。
HTTPX_DEFAULT_TIMEOUT = 300.0

# API 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False

# 各服务器默认绑定host。如改为"0.0.0.0"需要修改下方所有XX_SERVER的host
DEFAULT_BIND_HOST = "0.0.0.0" if sys.platform != "win32" else "127.0.0.1"

# webui.py server
WEBUI_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 8080,
}

LLM_SERVER = "198.203.120.5:40782"

# api.py server
API_SERVER = {
    "host": DEFAULT_BIND_HOST,
    "port": 6006,
}

API_SERVER_HOST_MAPPING = {
    "127.0.0.1": "198.203.120.8",
    "localhost": "198.203.120.8",
    "0.0.0.0": "198.203.120.8",
}

API_SERVER_PORT_MAPPING = {
    80: 40781,
    8000: 40782,
    8080: 40783,
    4141: 40784,
    6006: 40785,
    30000: 40786,
    30001: 40787,
    30002: 40788
}

# fastchat openai_api server
FSCHAT_OPENAI_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 30002,
}

# fastchat controller server
FSCHAT_CONTROLLER = {
    "host": DEFAULT_BIND_HOST,
    "port": 4141,
    "dispatch_method": "shortest_queue",
}

QA_JOB_SCRIPT_PATH = "/opt/projects/qabot-be/deploy/main.py"

# xinference server
XINFERENCE_API = {
    "host": DEFAULT_BIND_HOST,
    "port": 30002,
}

# LLM_SERVER = {
#     "host": DEFAULT_BIND_HOST,
#     "port": 30000,
# }

# fastchat model_worker server
# 这些模型必须是在model_config.MODEL_PATH或ONLINE_MODEL中正确配置的。
# 在启动startup.py时，可用通过`--model-name xxxx yyyy`指定模型，不指定则为LLM_MODELS
# FSCHAT_MODEL_WORKERS = {
#     # 所有模型共用的默认配置，可在模型专项配置中进行覆盖。
#     "default": {
#         "host": DEFAULT_BIND_HOST,
#         "port": 30001,
#         "device": LLM_DEVICE,
#         # False,'vllm',使用的推理加速框架,使用vllm如果出现HuggingFace通信问题，参见doc/FAQ
#         # vllm对一些模型支持还不成熟，暂时默认关闭
#         # fschat=0.2.33的代码有bug, 如需使用，源码修改fastchat.server.vllm_worker，
#         # 将103行中sampling_params = SamplingParams的参数stop=list(stop)修改为stop= [i for i in stop if i!=""]
#         "infer_turbo": False,
#
#         # model_worker多卡加载需要配置的参数
#         # "gpus": None, # 使用的GPU，以str的格式指定，如"0,1"，如失效请使用CUDA_VISIBLE_DEVICES="0,1"等形式指定
#         # "num_gpus": 1, # 使用GPU的数量
#         # "max_gpu_memory": "20GiB", # 每个GPU占用的最大显存
#
#         # 以下为model_worker非常用参数，可根据需要配置
#         # "load_8bit": False, # 开启8bit量化
#         # "cpu_offloading": None,
#         # "gptq_ckpt": None,
#         # "gptq_wbits": 16,
#         # "gptq_groupsize": -1,
#         # "gptq_act_order": False,
#         # "awq_ckpt": None,
#         # "awq_wbits": 16,
#         # "awq_groupsize": -1,
#         # "model_names": LLM_MODELS,
#         # "conv_template": None,
#         # "limit_worker_concurrency": 5,
#         # "stream_interval": 2,
#         # "no_register": False,
#         # "embed_in_truncate": False,
#
#         # 以下为vllm_worker配置参数,注意使用vllm必须有gpu，仅在Linux测试通过
#
#         # tokenizer = model_path # 如果tokenizer与model_path不一致在此处添加
#         # 'tokenizer_mode':'auto',
#         # 'trust_remote_code':True,
#         # 'download_dir':None,
#         # 'load_format':'auto',
#         # 'dtype':'auto',
#         # 'seed':0,
#         # 'worker_use_ray':False,
#         # 'pipeline_parallel_size':1,
#         # 'tensor_parallel_size':1,
#         # 'block_size':16,
#         # 'swap_space':4 , # GiB
#         # 'gpu_memory_utilization':0.90,
#         # 'max_num_batched_tokens':2560,
#         # 'max_num_seqs':256,
#         # 'disable_log_stats':False,
#         # 'conv_template':None,
#         # 'limit_worker_concurrency':5,
#         # 'no_register':False,
#         # 'num_gpus': 1
#         # 'engine_use_ray': False,
#         # 'disable_log_requests': False
#
#     },
#     # 可以如下示例方式更改默认配置
#     # "Qwen-1_8B-Chat": { # 使用default中的IP和端口
#     #    "device": "cpu",
#     # },
#     "chatglm3-6b": {  # 使用default中的IP和端口
#         "device": "cuda",
#     },
# }

# fastchat multi model worker server
# FSCHAT_MULTI_MODEL_WORKERS = {
#     # TODO:
# }