from typing import Dict, List, Set
import os
import re
import sys
import copy
import json
import argparse
from collections import defaultdict, OrderedDict

project_root_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
)
print(project_root_dir)

sys.path.insert(0, project_root_dir)

from ce_data.drop.utils import read_dataset, dataset_stats, get_contrative_answer_dict, write_dataset
from ce_data.drop.date_compare_data import get_datecompare_contrastive_answer_dict
from ce_data.drop.num_compare_data import get_numcompare_contrastive_answer_dict

from src.data.dataset_readers.drop.drop_utils import extract_answer_info_from_annotation, get_target_numbers


def jaccard(token_set1: Set[str], token_set2: Set[str]):
    intersection_size = len(token_set1.intersection(token_set2))
    union_size = len(token_set1.union(token_set2))
    return float(intersection_size)/float(union_size)


def tokenize(string):
    tokens = string.split(" ")
    tokens = [x for token in tokens for x in token.split("-")]
    return tokens


def filter_answer_candidate(candidate_answer: str, gold_ans_tokenset: List[Set[str]], max_len: int = 100):
    candidate_answer = candidate_answer.strip()
    if not candidate_answer:
        return None

    if len(candidate_answer) > max_len:
        return False

    candidate_token_set = set(tokenize(candidate_answer))
    if all(jaccard(candidate_token_set, x) < 0.5 for x in gold_ans_tokenset):
        return True


def get_topk_candidates(instance_dict, answer_annotation: Dict, max_contrastive_ans_len: int = 100):
    """Read instance prediction dictionary to get K-contrastive answers.

    Returns:
    --------
    query_id: str
    contrastive_answer_dict: Dict
        "contrastive_answer": {
            "number": "",
            "date": {
                "day": "",
                "month": "",
                "year": ""
            },
            "spans": ["ans1", "ans2", ... ]
        }
        All answers can be written in "spans" irrespective of type since the drop_reader is able to handle numbers from
        it. For different multi-span answers A1 and A2, just merge and add each span since the combinatorics will take
        care of that.
    """
    if answer_annotation is None:
        return None
    gold_answer_type, gold_answer_texts = extract_answer_info_from_annotation(answer_annotation)
    answer_text_tokensets: List[Set[str]] = [set(tokenize(answer)) for answer in gold_answer_texts]

    query_id = instance_dict["query_id"]
    # {head_name: {"logprobs": [], "values": []}}
    all_answers = instance_dict["all_answers"]

    # Get 3 "count" values, 10 "arithmetic" values, 10 "multi_span", 10 "passage_span", 5 "question_span"
    candidate_answers = []
    if "count" in all_answers:
        count_answers = all_answers["count"]["values"][0:4]
        candidate_answers.extend(count_answers)

    if "arithmetic" in all_answers:
        arithmetic_answers = all_answers["arithmetic"]["values"][0:10]
        candidate_answers.extend(arithmetic_answers)

    if "multi_span" in all_answers:
        multispans_answers = all_answers["multi_span"]["values"][0:20]
        multispans_answers = [x for spans in multispans_answers for x in spans]
        candidate_answers.extend(multispans_answers)

    if "passage_span" in all_answers:
        passage_span_answers = all_answers["passage_span"]["values"][0:20]
        candidate_answers.extend(passage_span_answers)

    if "question_span" in all_answers:
        question_span_answers = all_answers["question_span"]["values"][0:10]
        candidate_answers.extend(question_span_answers)

    filtered_contrastive_answers = []
    for candidate_ans in candidate_answers:
        if filter_answer_candidate(candidate_ans, answer_text_tokensets, max_len=max_contrastive_ans_len):
            filtered_contrastive_answers.append(candidate_ans)

    filtered_contrastive_answers = list(OrderedDict.fromkeys(filtered_contrastive_answers))

    if not filtered_contrastive_answers:
        return None

    contrastive_answer_dict = {
        "number": "",
        "date": {
            "day": "",
            "month": "",
            "year": ""
        },
        "spans": filtered_contrastive_answers
    }

    return contrastive_answer_dict


def get_data_with_contrastive_answer(topk_preds_jsonl: str, drop_json: str, output_json: str):
    drop_data = read_dataset(drop_json)

    print("\nPreparing id2goldanswer ... ")
    id2goldanswer = {}
    for pid, pinfo in drop_data.items():
        for qa in pinfo["qa_pairs"]:
            query_id = qa["query_id"]
            if "answer" in qa:
                answer_dict = qa["answer"]
                id2goldanswer[query_id] = answer_dict

    print("\nPreparing contrastive answers ... ")
    total_contrastive_ans = 0
    id2contrastive_answer = {}
    with open(topk_preds_jsonl) as f:
        for line in f:
            instance_pred_dict = json.loads(line)
            query_id = instance_pred_dict["query_id"]
            answer_dict = id2goldanswer.get(query_id, None)
            contrastive_answer_dict = get_topk_candidates(instance_pred_dict, answer_dict)
            if contrastive_answer_dict is not None:
                id2contrastive_answer[query_id] = contrastive_answer_dict
                total_contrastive_ans += len(contrastive_answer_dict["spans"])

    avg_num_contrastive_answers = total_contrastive_ans/float(len(id2contrastive_answer))
    print("\nInstances w/ contrastive answer: {}   Avg number of answers: {}".format(
        len(id2contrastive_answer), avg_num_contrastive_answers
    ))

    print("\nAdding contrastive answers to drop data ... ")
    total_qa_w_contrastans = 0
    for pid, passage_info in drop_data.items():
        passage = passage_info["passage"]
        for qa in passage_info["qa_pairs"]:
            query_id = qa["query_id"]
            if query_id in id2contrastive_answer:
                contrastive_answer_dict = id2contrastive_answer[query_id]
                qa["contrastive_answer"] = contrastive_answer_dict
            if "contrastive_answer" in qa:
                total_qa_w_contrastans += 1

    print("Total questions with contrastive answers: {}".format(total_qa_w_contrastans))
    write_dataset(dataset=drop_data, output_json=output_json)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "topk_preds_jsonl", type=str, help="JSONL with topk-candidates")
    parser.add_argument(
        "drop_json", type=str, help="Input drop_json"
    )
    parser.add_argument(
        "output_json", type=str, help="Output drop_json"
    )
    args = parser.parse_args()

    get_data_with_contrastive_answer(topk_preds_jsonl=args.topk_preds_jsonl,
                                     drop_json=args.drop_json,
                                     output_json=args.output_json)

    # python ce_data/drop/topk_candidates.py \
    #   ~/nfs2_nitishg/checkpoints/tase/drop/predictions/drop_train_topk_50.jsonl \
    #   drop_data/drop_dataset_train.json \
    #   drop_data/drop_dataset_train_topkv1.json


