import itertools
import os
import re
import functools
from collections import Counter

import pandas
# import hanlp

from configs import COMMON_PATH, logger

stopwords_file = f"{COMMON_PATH}/stopwords.txt"
if not os.path.exists(stopwords_file):
    stopwords_file = ""
stopwords = list()
if stopwords_file:
    with open(stopwords_file) as f:
        stopwords = [i.replace("\n", "") for i in f.readlines()]
    logger.info(f"stopwords len {len(stopwords)}")

from LAC import LAC

term_dict_file = f"{COMMON_PATH}/custom_20230720.txt"
if not os.path.exists(term_dict_file):
    term_dict_file = ""

lac = LAC(mode='seg')
if term_dict_file:
    lac.load_customization(term_dict_file, sep=None)


# term_dictionary = list()
# with open(term_dict_file) as f:
#     term_dictionary = [i.replace("\n", "")[:-4] for i in f.readlines()]
# logger.info(f"term_dictionary len {len(term_dictionary)}")

# tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)


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
    seg_result = lac.run([sentence])[0]
    # seg_result = tok_fine(sentence)
    seg_result = [j.strip() for j in seg_result]
    return [j for j in seg_result if j and j not in stopwords]


def clean_text(lbl, remove_stop=False, return_list=False):
    lbl = re.sub("[^A-Za-z0-9\u4e00-\u9fff]", "_", lbl)
    lbl = re.sub("_+", " ", lbl)
    lbl = re.sub(" {2,}", " ", lbl).strip()
    lbl = lbl.strip().upper()

    if remove_stop:
        result = seg_text(lbl)
        if return_list:
            lbl = result
        else:
            lbl = "".join(result)
    return lbl


def remove_stop(ans):
    ans = ans.upper()
    result = seg_text(ans)
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
    if len(raw_query_txt) > 3:
        for ch in raw_query_txt:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
    return False


def is_valid_std_query(raw_query_txt: str, std_ans_txt: str):
    return is_valid_query(raw_query_txt) and is_valid_ans(std_ans_txt)


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


def check_faq_is_generated(faq_file):
    this_df = pandas.read_excel(faq_file, dtype=str)
    col_names = this_df.columns.values.tolist()
    if "生成序号" in col_names and "生成问题" in col_names:
        return True, this_df
    else:
        return False, this_df


def check_faq_is_unitx(faq_filepath):
    excel_file = pandas.ExcelFile(faq_filepath)
    sheet_names = excel_file.sheet_names
    if "通用答案" in sheet_names and "相似问题" in sheet_names:
        return True
    else:
        return False


def load_df_generated(this_df):
    query_list = list()
    # this_df = pandas.read_excel(faq_file, dtype=str)
    this_df.set_index('生成序号')
    this_df.fillna("", inplace=True)
    logger.info(f"this_df {this_df.shape}")
    for idx, row in this_df.iterrows():
        raw_q = row["生成问题"]
        raw_a = row["生成答案"]
        if is_valid_std_query(raw_q, raw_a):
            l_query = StandardQuery(idx, raw_q, raw_a)
            query_list.append(l_query)

    return query_list


def load_df_raw(this_df, filename):
    query_list = list()

    # this_df = pandas.read_excel(faq_full_file, dtype=str)
    this_df.fillna("", inplace=True)
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
        # end_dt = row["有效时间止"]

        if cls_2:
            l_cls_2 = cls_2
            l_cls_3 = cls_3
            l_cls_4 = cls_4

        if raw_q and raw_a:
            l_idx = idx
            l_raw_q = raw_q
            l_attri = attri

        raw_exq_type = raw_exq_type[0] if raw_exq_type else ""

        if "民生微信" in channel and not raw_exq_type and not sample:
            l_query = StandardQuery(l_idx, l_raw_q, raw_a)
            l_query.attri = l_attri
            l_query.ref = filename

            # if end_dt:
            #     end_dt = str(int(float(end_dt)))
            #     l_query.end_dt = end_dt
            if l_cls_2:
                l_query.cls_list.append(l_cls_2)
            if l_cls_3:
                l_query.cls_list.append(l_cls_3)
            if l_cls_4:
                l_query.cls_list.append(l_cls_4)

            this_query_list.append(l_query)

        elif raw_exq_type == "0" and not sample:
            l_query.extend_list.append(raw_exq)

            sample_list = list()

        elif (raw_exq_type == "1" or raw_exq_type == "2") and not sample:
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


def load_df_raw_unitx(file_path):
    query_list = list()

    df_qa = pandas.read_excel(file_path, sheet_name="通用答案", dtype=str)
    df_q = pandas.read_excel(file_path, sheet_name="相似问题", dtype=str)

    df_qa.fillna("", inplace=True)
    logger.info(f"df_qa {df_qa.shape}")

    df_q.fillna("", inplace=True)
    logger.info(f"df_qa {df_q.shape}")

    lookup_dict = dict()
    for idx, row in df_q.iterrows():
        raw_q = row["标准问题(必填)"]
        sim_q = row["相似问题(必填)"]

        if raw_q and sim_q:
            if raw_q in lookup_dict:
                lookup_dict[raw_q].append(sim_q)
            else:
                lookup_dict[raw_q] = [sim_q]

    l_idx = 0

    for idx, row in df_qa.iterrows():
        raw_q = row["标准问题(必填)"]
        raw_a = row["通用答案1"]

        if raw_q and raw_a:
            l_query = StandardQuery(l_idx, raw_q, raw_a)
            l_query.ref = file_path

            if is_valid_std_query(l_query.raw_q, l_query.std_a):
                if raw_q in lookup_dict:
                    sample_list = lookup_dict[raw_q]
                    l_query.sample_list_of_list.append(sample_list)

                query_list.append(l_query)
                l_idx += 1

    return query_list


def load_faq(faq_filepath):
    is_unitx = check_faq_is_unitx(faq_filepath)

    if is_unitx:
        raw_query_obj_list = load_df_raw_unitx(faq_filepath)
    else:
        is_generated, this_df = check_faq_is_generated(faq_filepath)

        if is_generated:
            raw_query_obj_list = load_df_generated(this_df)
        else:
            raw_query_obj_list = load_df_raw(this_df, faq_filepath)

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


def load_gen_file(file_path):
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
