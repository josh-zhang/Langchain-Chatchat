import os
import re
import logging
import functools

from LAC import LAC

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# import do_correction

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
    seg_result = lac.run([sentence])
    seg_result = seg_result[0]
    return [j.strip() for j in seg_result if j.strip()]


def seg_text_list(sentence_list, return_list_of_list=False):
    seg_result = lac.run(sentence_list)
    seg_result_text = list()
    for i in seg_result:
        if return_list_of_list:
            seg_result_text.append([j.strip() for j in i if j.strip()])
        else:
            text = " ".join([j.strip() for j in i if j.strip()])
            seg_result_text.append(text)
    return seg_result_text


def clean_text(lbl, remove_stop=False, return_list=False):
    lbl = re.sub("[^A-Za-z0-9\u4e00-\u9fa5]", "_", lbl)
    lbl = re.sub("_+", " ", lbl)
    lbl = re.sub(" {2,}", " ", lbl).strip()
    lbl = lbl.strip().upper()

    if remove_stop:
        result = list()
        for word in seg_text(lbl):
            if word not in stopwords:
                result.append(word)
        if return_list:
            lbl = result
        else:
            lbl = "".join(result)
    return lbl


def remove_stop(ans):
    ans = ans.upper()
    result = list()
    for word in seg_text(ans):
        if word not in stopwords:
            result.append(word)
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
    return len(raw_query_txt) > 0


def is_valid_std_query(raw_query_txt: str, std_ans_txt: str):
    return is_valid_query(raw_query_txt) and is_valid_ans(std_ans_txt)


def is_valid_user_query(raw_query_txt: str):
    return is_valid_query(raw_query_txt)
