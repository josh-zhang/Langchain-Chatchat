import os
import sys

import numpy
import pandas
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from langchain_core.documents import Document

from server.knowledge_base.kb_cache.faiss_cache import kb_faiss_pool
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
from server.knowledge_base.kb_doc_api import get_total_score_sorted

kb_name = ""
kb_info = ""
search_enhance = False
embed_model = ""
BM_25_FACTOR = 0.4
# top_k = 10
score_threshold = 0.3

kb = FaissKBService(kb_name, kb_info, search_enhance, embed_model=embed_model)

if not os.path.exists(kb.doc_path):
    os.makedirs(kb.doc_path)

kb_faiss_pool.load_vector_store(kb_name=kb.kb_name,
                                vector_name="docs",
                                embed_model=kb.embed_model)


def text_to_doc(text, file_path):
    return Document(page_content=text, metadata={"source": file_path})


def render_cls_res(preds, labels, save_file):
    preds = numpy.array(preds)
    labels = numpy.array(labels)

    accuracy = accuracy_score(labels, preds)
    print(f"accuracy {accuracy}")

    cls_report = classification_report(labels, preds)
    print(cls_report)

    cls_report = classification_report(labels, preds, output_dict=True)

    df_cls = pandas.DataFrame(cls_report).T
    df_cls.to_csv(save_file)


def evaluate(load_path, save_path, top_n):
    all_files = os.listdir(load_path)
    all_files = [f for f in all_files if f.endswith(".xlsx")]

    total_gt_dict = dict()

    df_list = list()
    for excel_file in all_files:
        title = excel_file[:-5]

        df = pandas.read_excel(os.path.join(load_path, excel_file), dtype=str)
        df.fillna("", inplace=True)

        cleaned_df = df[df['问题序号'].str.strip() != '']

        unique_values = cleaned_df['标准问题'].unique()

        for q in unique_values:
            condition1 = cleaned_df['是否出现'] == "1"
            condition2 = cleaned_df['标准问题'] == q
            filtered_df = cleaned_df[condition1 & condition2]

            gt_dict = dict()

            for idx, row in filtered_df.iterrows():
                question = row['标准问题']
                answer = row['标准答案']
                gt = row['正文']

                if question in gt_dict:
                    gt_dict[question].append(gt)
                else:
                    gt_dict[question] = [gt]

            for k, v in gt_dict.items():
                assert k not in total_gt_dict
                total_gt_dict[k] = v

        cleaned_df["文章标题"] = title

        df_list.append(cleaned_df)

    combined_df = pandas.concat(df_list, ignore_index=True)

    unique_df = combined_df.drop_duplicates(subset=['文章标题', '正文'])
    label_list = list(zip(unique_df['文章标题'], unique_df['正文']))
    docs = [text_to_doc(text, file_path) for text, file_path in label_list]

    data = kb._docs_to_embeddings(docs)  # 将向量化单独出来可以减少向量库的锁定时间
    with kb.load_vector_store("docs").acquire() as vs:
        ids = vs.add_embeddings(text_embeddings=zip(data["texts"], data["embeddings"]), metadatas=data["metadatas"])

        vs_path = kb.get_vs_path("docs")
        vs.save_local(vs_path)

    base_list = [i[0] for i in label_list]

    pred_binarys = list()
    gt_binarys = list()

    for query, gt_list in total_gt_dict.items():
        ks_docs_data, ks_qa_data = kb.search_allinone(query, top_n * 3, 0.0)

        if kb.search_enhance:
            bm25_docs_data, bm25_qa_data = kb.enhance_search_allinone(query, 2, BM_25_FACTOR)
            docs_data = kb.merge_docs(ks_docs_data, bm25_docs_data, is_max=True)
            qa_data = kb.merge_answers(ks_qa_data, bm25_qa_data, is_max=True)
        else:
            docs_data = ks_docs_data
            qa_data = ks_qa_data

        docs_data = docs_data + qa_data

        docs_data = get_total_score_sorted(docs_data, score_threshold)

        print(f"top_n {top_n} and {len(docs_data)} docs total searched")
        print(docs_data)

        docs_data = docs_data[:top_n]
        preds = [d.page_content for d in docs_data]

        pred_binary = [1 if value in preds else 0 for value in base_list]
        gt_binary = [1 if value in gt_list else 0 for value in base_list]

        pred_binarys.append(pred_binary)
        gt_binarys.append(gt_binary)

    render_cls_res(pred_binarys, gt_binarys, save_path)


if __name__ == '__main__':
    load_path = sys.argv[1]
    save_path = sys.argv[2]
    top_n = int(sys.argv[3])

    evaluate(load_path, save_path, top_n)
