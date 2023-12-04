import os
from configs import (
    KB_ROOT_PATH,
    CHUNK_SIZE,
    OVERLAP_SIZE,
    ZH_TITLE_ENHANCE,
    logger,
    log_verbose,
    text_splitter_dict,
    LLM_MODELS,
    TEXT_SPLITTER_NAME,
)
import importlib
from text_splitter import zh_title_enhance as func_zh_title_enhance
import langchain.document_loaders
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter
from server.utils import run_in_thread_pool, get_model_worker_config
import json
from typing import List, Union, Dict, Tuple, Generator
import chardet
from unstructured.documents.elements import Element, Text, ElementMetadata, Title
from unstructured.partition.common import get_last_modified_date


def validate_kb_name(knowledge_base_id: str) -> bool:
    # 检查是否包含预期外的字符或路径攻击关键字
    if "../" in knowledge_base_id:
        return False
    return True


def get_kb_path(knowledge_base_name: str):
    return os.path.join(KB_ROOT_PATH, knowledge_base_name)


def get_doc_path(knowledge_base_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "content")


def get_vs_path(knowledge_base_name: str, vector_name: str):
    return os.path.join(get_kb_path(knowledge_base_name), "vector_store", vector_name)


def get_file_path(knowledge_base_name: str, doc_name: str):
    return os.path.join(get_doc_path(knowledge_base_name), doc_name)


def list_kbs_from_folder():
    return [f for f in os.listdir(KB_ROOT_PATH)
            if os.path.isdir(os.path.join(KB_ROOT_PATH, f))]


def list_files_from_folder(kb_name: str):
    doc_path = get_doc_path(kb_name)
    result = []

    def is_skiped_path(path: str):
        tail = os.path.basename(path).lower()
        for x in ["temp", "tmp", ".", "~$"]:
            if tail.startswith(x):
                return True
        return False

    def process_entry(entry):
        if is_skiped_path(entry.path):
            return

        if entry.is_symlink():
            target_path = os.path.realpath(entry.path)
            with os.scandir(target_path) as target_it:
                for target_entry in target_it:
                    process_entry(target_entry)
        elif entry.is_file():
            # result.append(entry.path)
            result.append(entry.name)
        elif entry.is_dir():
            with os.scandir(entry.path) as it:
                for sub_entry in it:
                    process_entry(sub_entry)

    with os.scandir(doc_path) as it:
        for entry in it:
            process_entry(entry)

    return result


LOADER_DICT = {"CustomHTMLLoader": ['.html'],
               # "UnstructuredHTMLLoader": ['.html'],
               "UnstructuredMarkdownLoader": ['.md'],
               "JSONLoader": [".json"],
               "JSONLinesLoader": [".jsonl"],
               "CSVLoader": [".csv"],
               # "FilteredCSVLoader": [".csv"], # 需要自己指定，目前还没有支持
               "RapidOCRPDFLoader": [".pdf"],
               "RapidOCRLoader": ['.png', '.jpg', '.jpeg', '.bmp'],
               "UnstructuredEmailLoader": ['.eml', '.msg'],
               "UnstructuredEPubLoader": ['.epub'],
               "UnstructuredExcelLoader": ['.xlsx', '.xlsd'],
               "NotebookLoader": ['.ipynb'],
               "UnstructuredODTLoader": ['.odt'],
               "PythonLoader": ['.py'],
               "UnstructuredRSTLoader": ['.rst'],
               "UnstructuredRTFLoader": ['.rtf'],
               "SRTLoader": ['.srt'],
               "TomlLoader": ['.toml'],
               "UnstructuredTSVLoader": ['.tsv'],
               "UnstructuredWordDocumentLoader": ['.docx', 'doc'],
               "UnstructuredXMLLoader": ['.xml'],
               "UnstructuredPowerPointLoader": ['.ppt', '.pptx'],
               "UnstructuredFileLoader": ['.txt'],
               }
SUPPORTED_EXTS = [ext for sublist in LOADER_DICT.values() for ext in sublist]


class CustomHTMLLoader(langchain.document_loaders.unstructured.UnstructuredFileLoader):
    delimiter = " 之 "

    def load_faq_html(self, file_path):
        return [
            ("一、卡片基本介绍", [
                ("产品名称", "民生长城欧拉银联绿色低碳联名信用卡", []),
                ("发行范围", "全国", []),
                ("卡片级别", "标准白金级别", []),
                ("卡片有效期", "10年", []),
                ("额度区间", "标准白金级别：10,000-100,000元", []),
                ("账户类型", "人民币美元账户", []),
                ("申请渠道", "网申渠道及线下手机APP渠道均可进件，同现有正常申请及审核流程。", []),
                ("卡片品牌及币种", "银联品牌，人民币单币种", []),

                ("卡片使用", """（一）虚拟银行：
        个人网上银行、手机银行（含信用卡版及总行版）、微信银行（含信用卡版及总行版）、短信银行均可以正常使用。
        （二）其他卡片功能及收费标准同民生标准卡（详见“010304-000171息费相关”）。""",
                 [
                     ("（一）虚拟银行",
                      """个人网上银行、手机银行（含信用卡版及总行版）、微信银行（含信用卡版及总行版）、短信银行均可以正常使用。"""),
                     ("", "（二）其他卡片功能及收费标准同民生标准卡（详见“010304-000171息费相关”）")
                 ]
                 ),
                ("年费标准及年费政策",
                 "主卡年费：600元/年；附属卡年费：300元/年。2022年12月31日（含）前，首年免年费，当年刷卡消费18笔或5万人民币（或等值外币），减免次年年费。",
                 []),
            ]),
            ("二、卡片权益", [("增值服务", "增值服务权益与同级别民生标准信用卡相同。", [])]),
            ("三、专属用卡权益", [
                ("新户消费达标即送花加鲜花好礼", """（一）活动时间：产品上线日起至2022年12月31日（含首尾两日）
        （二）活动内容：活动期间，因申请民生长城欧拉银联绿色低碳联名信用卡而首次核卡成为民生信用卡主卡持卡人的新客户，核卡30天内（含）激活卡片并任意消费一笔，即可获取花加“悦花包月服务1个月”礼品1份，每位持卡人仅可参与一次权益活动。权益礼品限3000份，数量有限，先到先得。
        （三）活动流程简介：
        满足条件--获得资质--用资质兑换礼品券码1张--使用券码
        （四）活动细则：
        1、本活动限因成功申请民生长城欧拉银联绿色低碳联名信用卡而首次成为民生信用卡主卡持卡人的客户，且消费达标日期在2022年12月31日（含）前，消费交易均以交易记账日为准，部分消费并非实时到账，遇此情况不予特殊处理，附属卡交易不计入主卡达标交易统计范围。持民生信用卡销卡后重新申请的持卡人，不能参加此活动。
        2、资质及礼品
        （1）资质获得时间：满足活动条件后均会于达标后（不含当日）3个自然日内获得1个“民生长城欧拉银联绿色低碳联名信用卡新客首刷鲜花好礼”资质。
        （2）资质兑换礼品券码时间：获得资质后1个月内兑换有效，过期视为自动放弃，例如2022年8月18日兑换，2022年9月17日失效。
        （3）资质查询及礼品券码兑换路径：
        全民生活APP-精选-“福利社”-“全民领福利--产品权益”-“民生长城欧拉银联绿色低碳联名信用卡新客首刷鲜花好礼”。
        民生信用卡微信公众号-查账-我的特权-“全民领福利--产品权益”-“民生长城欧拉银联绿色低碳联名信用卡新客首刷鲜花好礼”。
        （4）兑换后礼品券码查询渠道：
        全民生活APP-精选-“福利社”-“我的福利库”-权益码查看券码信息
        民生信用卡微信公众号-查账-我的特权-“我的福利库”-权益码查看券码信息
        （5）礼品券码使用渠道：前往“FLOWERPLUS花加”微信小程序-我的-我的服务-兑换花卡，输入兑换码及收花人等相关信息后点击“确认兑换”即可完成兑换。
        （6）礼品券码使用规则：
        1）兑换码有效期：兑换码自领取之日起1年内有效，例如2022年8月18日兑换，2023年8月17日失效；
        2）本券不可转让，不做退换，不兑现金，不设找零，抵用金额不可开发票。
        3）兑换码由FLOWERPLUS花加提供，如针对兑换码的使用有疑问，客户可致电FLOWERPLUS花加客服电话4008885928（工作时间每日9:00-18:00，含周末及节假日）。
        3、客户参加本权益活动时所持民生长城欧拉银联绿色低碳联名信用卡产品须为正常激活状态，否则客户无法参与本权益活动。客户达标消费统计限民生长城欧拉银联绿色低碳联名信用卡。若使用账户下其他卡片消费，则不参与达标门槛统计。
        4、若在获赠礼品前，持卡人卡片或账户处于逾期、冻结等非正常状态，或在活动结束前销卡（户）的，本行有权取消其领奖资质，持卡人亦不得以此要求本行对其进行任何形式的补偿。
        5、本活动及未尽事宜仍受《中国民生银行信用卡（个人卡）领用合约》、《中国民生银行民生信用卡章程》以及其他相关文件约束。在法律法规允许的范围内，本活动最终解释权归中国民生银行信用卡中心所有，如客户在参与权益活动及信用卡产品使用过程中有任何问题可联系我中心在线客服进行咨询。
        6、持卡人参与本活动即视为理解并同意本活动细则，在法律法规许可范围内，中国民生银行信用卡中心保留变更、调整、终止本活动之权利并有权调整或变更本活动规则，活动内容及细则以民生信用卡官网公布为准。""",
                 [
                     ("（一）活动时间", "产品上线日起至2022年12月31日（含首尾两日）"),
                     ("（二）活动内容",
                      "活动期间，因申请民生长城欧拉银联绿色低碳联名信用卡而首次核卡成为民生信用卡主卡持卡人的新客户，核卡30天内（含）激活卡片并任意消费一笔，即可获取花加“悦花包月服务1个月”礼品1份，每位持卡人仅可参与一次权益活动。权益礼品限3000份，数量有限，先到先得。"),
                     ("（三）活动流程简介", "满足条件--获得资质--用资质兑换礼品券码1张--使用券码"),
                     ("（四）活动细则", """1、本活动限因成功申请民生长城欧拉银联绿色低碳联名信用卡而首次成为民生信用卡主卡持卡人的客户，且消费达标日期在2022年12月31日（含）前，消费交易均以交易记账日为准，部分消费并非实时到账，遇此情况不予特殊处理，附属卡交易不计入主卡达标交易统计范围。持民生信用卡销卡后重新申请的持卡人，不能参加此活动。
        2、资质及礼品
        （1）资质获得时间：满足活动条件后均会于达标后（不含当日）3个自然日内获得1个“民生长城欧拉银联绿色低碳联名信用卡新客首刷鲜花好礼”资质。
        （2）资质兑换礼品券码时间：获得资质后1个月内兑换有效，过期视为自动放弃，例如2022年8月18日兑换，2022年9月17日失效。
        （3）资质查询及礼品券码兑换路径：
        全民生活APP-精选-“福利社”-“全民领福利--产品权益”-“民生长城欧拉银联绿色低碳联名信用卡新客首刷鲜花好礼”。
        民生信用卡微信公众号-查账-我的特权-“全民领福利--产品权益”-“民生长城欧拉银联绿色低碳联名信用卡新客首刷鲜花好礼”。
        （4）兑换后礼品券码查询渠道：
        全民生活APP-精选-“福利社”-“我的福利库”-权益码查看券码信息
        民生信用卡微信公众号-查账-我的特权-“我的福利库”-权益码查看券码信息
        （5）礼品券码使用渠道：前往“FLOWERPLUS花加”微信小程序-我的-我的服务-兑换花卡，输入兑换码及收花人等相关信息后点击“确认兑换”即可完成兑换。
        （6）礼品券码使用规则：
        1）兑换码有效期：兑换码自领取之日起1年内有效，例如2022年8月18日兑换，2023年8月17日失效；
        2）本券不可转让，不做退换，不兑现金，不设找零，抵用金额不可开发票。
        3）兑换码由FLOWERPLUS花加提供，如针对兑换码的使用有疑问，客户可致电FLOWERPLUS花加客服电话4008885928（工作时间每日9:00-18:00，含周末及节假日）。
        3、客户参加本权益活动时所持民生长城欧拉银联绿色低碳联名信用卡产品须为正常激活状态，否则客户无法参与本权益活动。客户达标消费统计限民生长城欧拉银联绿色低碳联名信用卡。若使用账户下其他卡片消费，则不参与达标门槛统计。
        4、若在获赠礼品前，持卡人卡片或账户处于逾期、冻结等非正常状态，或在活动结束前销卡（户）的，本行有权取消其领奖资质，持卡人亦不得以此要求本行对其进行任何形式的补偿。
        5、本活动及未尽事宜仍受《中国民生银行信用卡（个人卡）领用合约》、《中国民生银行民生信用卡章程》以及其他相关文件约束。在法律法规允许的范围内，本活动最终解释权归中国民生银行信用卡中心所有，如客户在参与权益活动及信用卡产品使用过程中有任何问题可联系我中心在线客服进行咨询。
        6、持卡人参与本活动即视为理解并同意本活动细则，在法律法规许可范围内，中国民生银行信用卡中心保留变更、调整、终止本活动之权利并有权调整或变更本活动规则，活动内容及细则以民生信用卡官网公布为准。"""),
                 ]
                 )
            ]
             )
        ]

    def load_content(self, file_path) -> List[Element]:
        elements: List[Element] = []

        last_modification_date = get_last_modified_date(file_path)
        file_directory, file_name = os.path.split(file_path)
        ext = os.path.splitext(file_name)[-1].lower()
        file_name = file_name[:-len(ext)]

        chapters_list = self.load_faq_html(file_path)

        for ii, chapter_tuple in enumerate(chapters_list):
            chapter_number = ii + 1

            chapter_title = file_name + self.delimiter + chapter_tuple[0]
            chapter_title_ele = Title(text=chapter_title, metadata=ElementMetadata(filename=file_path,
                                                                                   filetype="html",
                                                                                   page_number=chapter_number))
            chapter_title_ele.metadata.last_modified = last_modification_date
            chapter_title_ele.metadata.category_depth = 0

            elements.append(chapter_title_ele)

            paragraphs = chapter_tuple[1]
            for iii, paragraph_tuple in enumerate(paragraphs):
                paragraph_title = chapter_tuple[0] + self.delimiter + paragraph_tuple[0]
                paragraph_text = paragraph_tuple[1]
                sub_paragraphs = paragraph_tuple[2]

                paragraph_title_ele = Title(text=paragraph_title, metadata=ElementMetadata(filename=file_path,
                                                                                           filetype="html",
                                                                                           page_number=chapter_number))
                paragraph_title_ele.metadata.parent_id = chapter_title_ele.id
                paragraph_title_ele.metadata.last_modified = last_modification_date
                paragraph_title_ele.metadata.category_depth = 1

                elements.append(paragraph_title_ele)

                if sub_paragraphs:
                    for iii, sub_paragraph_tuple in enumerate(sub_paragraphs):
                        sub_paragraph_title = paragraph_tuple[0] + self.delimiter + sub_paragraph_tuple[0]
                        sub_paragraph_text = sub_paragraph_tuple[1]

                        sub_paragraph_title_ele = Title(text=sub_paragraph_title,
                                                        metadata=ElementMetadata(filename=file_path,
                                                                                 filetype="html",
                                                                                 page_number=chapter_number))
                        sub_paragraph_title_ele.metadata.parent_id = paragraph_title_ele.id
                        sub_paragraph_title_ele.metadata.last_modified = last_modification_date
                        sub_paragraph_title_ele.metadata.category_depth = 2

                        sub_paragraph_text_ele = Text(text=sub_paragraph_text,
                                                      metadata=ElementMetadata(filename=file_path,
                                                                               filetype="html",
                                                                               page_number=chapter_number))
                        sub_paragraph_text_ele.metadata.parent_id = paragraph_title_ele.id
                        sub_paragraph_text_ele.metadata.last_modified = last_modification_date

                        elements.append(sub_paragraph_title_ele)
                        elements.append(sub_paragraph_text_ele)
                else:
                    paragraph_text_ele = Text(text=paragraph_text, metadata=ElementMetadata())
                    paragraph_title_ele.metadata.parent_id = chapter_title_ele.id
                    paragraph_title_ele.metadata.last_modified = last_modification_date

                    elements.append(paragraph_text_ele)

        return elements

    def _get_elements(self) -> List:
        """Convert given content to documents."""
        return self.load_content(self.file_path)


langchain.document_loaders.CustomHTMLLoader = CustomHTMLLoader


# patch json.dumps to disable ensure_ascii
def _new_json_dumps(obj, **kwargs):
    kwargs["ensure_ascii"] = False
    return _origin_json_dumps(obj, **kwargs)


if json.dumps is not _new_json_dumps:
    _origin_json_dumps = json.dumps
    json.dumps = _new_json_dumps


class JSONLinesLoader(langchain.document_loaders.JSONLoader):
    '''
    行式 Json 加载器，要求文件扩展名为 .jsonl
    '''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._json_lines = True


langchain.document_loaders.JSONLinesLoader = JSONLinesLoader


def get_LoaderClass(file_extension):
    for LoaderClass, extensions in LOADER_DICT.items():
        if file_extension in extensions:
            return LoaderClass


# 把一些向量化共用逻辑从KnowledgeFile抽取出来，等langchain支持内存文件的时候，可以将非磁盘文件向量化
def get_loader(loader_name: str, file_path: str, loader_kwargs: Dict = None):
    '''
    根据loader_name和文件路径或内容返回文档加载器。
    '''
    loader_kwargs = loader_kwargs or {}
    try:
        if loader_name in ["RapidOCRPDFLoader", "RapidOCRLoader", "FilteredCSVLoader"]:
            document_loaders_module = importlib.import_module('document_loaders')
        else:
            document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, loader_name)
    except Exception as e:
        msg = f"为文件{file_path}查找加载器{loader_name}时出错：{e}"
        logger.error(f'{e.__class__.__name__}: {msg}',
                     exc_info=e if log_verbose else None)
        document_loaders_module = importlib.import_module('langchain.document_loaders')
        DocumentLoader = getattr(document_loaders_module, "UnstructuredFileLoader")

    if loader_name == "UnstructuredFileLoader":
        loader_kwargs.setdefault("autodetect_encoding", True)
    elif loader_name == "CSVLoader":
        if not loader_kwargs.get("encoding"):
            # 如果未指定 encoding，自动识别文件编码类型，避免langchain loader 加载文件报编码错误
            with open(file_path, 'rb') as struct_file:
                encode_detect = chardet.detect(struct_file.read())
            if encode_detect is None:
                encode_detect = {"encoding": "utf-8"}
            loader_kwargs["encoding"] = encode_detect["encoding"]
        ## TODO：支持更多的自定义CSV读取逻辑

    elif loader_name == "JSONLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)
    elif loader_name == "JSONLinesLoader":
        loader_kwargs.setdefault("jq_schema", ".")
        loader_kwargs.setdefault("text_content", False)

    loader = DocumentLoader(file_path, **loader_kwargs)
    return loader


def make_text_splitter(
        splitter_name: str = TEXT_SPLITTER_NAME,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        llm_model: str = LLM_MODELS[0],
):
    """
    根据参数获取特定的分词器
    """
    splitter_name = splitter_name or "SpacyTextSplitter"
    try:
        if splitter_name == "MarkdownHeaderTextSplitter":  # MarkdownHeaderTextSplitter特殊判定
            headers_to_split_on = text_splitter_dict[splitter_name]['headers_to_split_on']
            text_splitter = langchain.text_splitter.MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on)
        else:

            try:  ## 优先使用用户自定义的text_splitter
                text_splitter_module = importlib.import_module('text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)
            except:  ## 否则使用langchain的text_splitter
                text_splitter_module = importlib.import_module('langchain.text_splitter')
                TextSplitter = getattr(text_splitter_module, splitter_name)

            if text_splitter_dict[splitter_name]["source"] == "tiktoken":  ## 从tiktoken加载
                try:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter.from_tiktoken_encoder(
                        encoding_name=text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
            elif text_splitter_dict[splitter_name]["source"] == "huggingface":  ## 从huggingface加载
                if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "":
                    config = get_model_worker_config(llm_model)
                    text_splitter_dict[splitter_name]["tokenizer_name_or_path"] = \
                        config.get("model_path")

                if text_splitter_dict[splitter_name]["tokenizer_name_or_path"] == "gpt2":
                    from transformers import GPT2TokenizerFast
                    from langchain.text_splitter import CharacterTextSplitter
                    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
                else:  ## 字符长度加载
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        text_splitter_dict[splitter_name]["tokenizer_name_or_path"],
                        trust_remote_code=True)
                text_splitter = TextSplitter.from_huggingface_tokenizer(
                    tokenizer=tokenizer,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            else:
                try:
                    text_splitter = TextSplitter(
                        pipeline="zh_core_web_sm",
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
                except:
                    text_splitter = TextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap
                    )
    except Exception as e:
        print(e)
        text_splitter_module = importlib.import_module('langchain.text_splitter')
        TextSplitter = getattr(text_splitter_module, "RecursiveCharacterTextSplitter")
        text_splitter = TextSplitter(chunk_size=250, chunk_overlap=50)
    return text_splitter


class KnowledgeFile:
    def __init__(
            self,
            filename: str,
            knowledge_base_name: str,
            loader_kwargs: Dict = {},
    ):
        '''
        对应知识库目录中的文件，必须是磁盘上存在的才能进行向量化等操作。
        '''
        self.kb_name = knowledge_base_name
        self.filename = filename
        self.ext = os.path.splitext(filename)[-1].lower()
        if self.ext not in SUPPORTED_EXTS:
            raise ValueError(f"暂未支持的文件格式 {self.filename}")
        self.loader_kwargs = loader_kwargs
        self.filepath = get_file_path(knowledge_base_name, filename)
        self.docs = None
        self.splited_docs = None
        self.document_loader_name = get_LoaderClass(self.ext)
        self.text_splitter_name = TEXT_SPLITTER_NAME

    def file2docs(self, refresh: bool = False):
        if self.docs is None or refresh:
            logger.info(f"{self.document_loader_name} used for {self.filepath}")
            loader = get_loader(loader_name=self.document_loader_name,
                                file_path=self.filepath,
                                loader_kwargs=self.loader_kwargs)
            self.docs = loader.load()
        return self.docs

    def docs2texts(
            self,
            docs: List[Document] = None,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    ):
        docs = docs or self.file2docs(refresh=refresh)
        if not docs:
            return []
        if self.ext not in [".csv"]:
            if text_splitter is None:
                text_splitter = make_text_splitter(splitter_name=self.text_splitter_name, chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
            if self.text_splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        print(f"文档切分示例：{docs[0]}")
        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        self.splited_docs = docs
        return self.splited_docs

    def file2text(
            self,
            zh_title_enhance: bool = ZH_TITLE_ENHANCE,
            refresh: bool = False,
            chunk_size: int = CHUNK_SIZE,
            chunk_overlap: int = OVERLAP_SIZE,
            text_splitter: TextSplitter = None,
    ):
        if self.splited_docs is None or refresh:
            docs = self.file2docs()
            self.splited_docs = self.docs2texts(docs=docs,
                                                zh_title_enhance=zh_title_enhance,
                                                refresh=refresh,
                                                chunk_size=chunk_size,
                                                chunk_overlap=chunk_overlap,
                                                text_splitter=text_splitter)
        return self.splited_docs

    def file_exist(self):
        return os.path.isfile(self.filepath)

    def get_mtime(self):
        return os.path.getmtime(self.filepath)

    def get_size(self):
        return os.path.getsize(self.filepath)


def files2docs_in_thread(
        files: List[Union[KnowledgeFile, Tuple[str, str], Dict]],
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = OVERLAP_SIZE,
        zh_title_enhance: bool = ZH_TITLE_ENHANCE,
) -> Generator:
    '''
    利用多线程批量将磁盘文件转化成langchain Document.
    如果传入参数是Tuple，形式为(filename, kb_name)
    生成器返回值为 status, (kb_name, file_name, docs | error)
    '''

    def file2docs(*, file: KnowledgeFile, **kwargs) -> Tuple[bool, Tuple[str, str, List[Document]]]:
        try:
            return True, (file.kb_name, file.filename, file.file2text(**kwargs))
        except Exception as e:
            msg = f"从文件 {file.kb_name}/{file.filename} 加载文档时出错：{e}"
            logger.error(f'{e.__class__.__name__}: {msg}',
                         exc_info=e if log_verbose else None)
            return False, (file.kb_name, file.filename, msg)

    kwargs_list = []
    for i, file in enumerate(files):
        kwargs = {}
        try:
            if isinstance(file, tuple) and len(file) >= 2:
                filename = file[0]
                kb_name = file[1]
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            elif isinstance(file, dict):
                filename = file.pop("filename")
                kb_name = file.pop("kb_name")
                kwargs.update(file)
                file = KnowledgeFile(filename=filename, knowledge_base_name=kb_name)
            kwargs["file"] = file
            kwargs["chunk_size"] = chunk_size
            kwargs["chunk_overlap"] = chunk_overlap
            kwargs["zh_title_enhance"] = zh_title_enhance
            kwargs_list.append(kwargs)
        except Exception as e:
            yield False, (kb_name, filename, str(e))

    for result in run_in_thread_pool(func=file2docs, params=kwargs_list):
        yield result


if __name__ == "__main__":
    from pprint import pprint

    kb_file = KnowledgeFile(
        filename="/home/congyin/Code/Project_Langchain_0814/Langchain-Chatchat/knowledge_base/csv1/content/gm.csv",
        knowledge_base_name="samples")
    # kb_file.text_splitter_name = "RecursiveCharacterTextSplitter"
    docs = kb_file.file2docs()
    # pprint(docs[-1])
