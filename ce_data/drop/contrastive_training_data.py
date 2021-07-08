from typing import Dict, List
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



def count_bias_answer(**kwargs):
    """ If answer is number and in range [0, 10], add contrastive answer as '2'. """
    qa = kwargs["qa"]
    if 'answer' in qa and qa['answer']:
        answer = qa['answer']
    else:
        return None
    answer_type, answer_texts = extract_answer_info_from_annotation(answer)
    if answer_type is "number":
        numbers = get_target_numbers(answer_texts)
        if 2 in numbers:
            return None
        if any([x in range(0,10) for x in numbers]):
            contrastive_answer_dict = get_contrative_answer_dict("2")
            return contrastive_answer_dict
    return None


N_yard_pattern = re.compile("[0-9]+( |\-)yard")


def get_N_yard_contrastive_answers(**kwargs):
    qa = kwargs["qa"]
    passage = kwargs["passage"]

    if 'answer' in qa and qa['answer']:
        answer = qa['answer']
    else:
        return None
    answer_type, answer_texts = extract_answer_info_from_annotation(answer)
    if any(N_yard_pattern.match(x) is not None for x in answer_texts):
        # Answer contains "N-yard" pattern, find "X-yard" patterns in passage and add as contrastive answers
        regex_matches = list(N_yard_pattern.finditer(passage))
        contrastive_ans_spans = []
        for match in regex_matches:
            contrastive_ans_spans.append(match.string[match.start():match.end()])
        contrastive_ans_spans = list(OrderedDict.fromkeys(contrastive_ans_spans))
        # Now remove answer texts from this list
        contrastive_ans_spans = [x for x in contrastive_ans_spans if x not in answer_texts]
        if contrastive_ans_spans:
            return get_contrative_answer_dict(contrastive_ans_spans)


def get_data_with_contrastive_answer(drop_json: str, output_json: str, prune_data: bool):
    drop_data = read_dataset(drop_json)

    contrastive_types = {
        "count_bias": count_bias_answer,
        "N_yard": get_N_yard_contrastive_answers,
        "datecompare": get_datecompare_contrastive_answer_dict,
        "numcompare": get_numcompare_contrastive_answer_dict,
    }

    contrast_count = defaultdict(int)
    total_qa_w_contrastans = 0

    paras_to_remove = []
    for pid, passage_info in drop_data.items():
        passage = passage_info["passage"]
        new_qas = []
        for qa in passage_info["qa_pairs"]:
            for contrast_type, contrast_function in contrastive_types.items():
                # Get contrastive_answer_dict for this contrastive-type
                contrast_dict = contrast_function(qa=qa, passage=passage)
                if contrast_dict is not None:
                    contrast_count[contrast_type] += 1
                    # If already have contrastive answers for this question, add to exising
                    if "contrastive_answer" in qa:
                        # Unique spans
                        text_spans = list(OrderedDict.fromkeys(qa["contrastive_answer"]["spans"] + contrast_dict["spans"]))
                        qa["contrastive_answer"]["spans"] = text_spans
                    else:
                        qa["contrastive_answer"] = contrast_dict
            if "contrastive_answer" in qa:
                total_qa_w_contrastans += 1
                new_qas.append(qa)
        if not new_qas:
            paras_to_remove.append(pid)
        if prune_data:
            passage_info["qa_pairs"] = new_qas
    if prune_data:
        for pid in paras_to_remove:
            drop_data.pop(pid)

    print("Contrastive answer type distribution")
    print(contrast_count)
    print("Total questions with contrastive answers: {}".format(total_qa_w_contrastans))
    write_dataset(dataset=drop_data, output_json=output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=str, help="Input data file")
    parser.add_argument(
        "output_json", type=str, help="Output data file"
    )
    parser.add_argument(
        "--prune-data",
        dest="prune_data",
        help="Should we only keep examples for which at least one correct logical-form is found?",
        action="store_true",
    )
    args = parser.parse_args()

    get_data_with_contrastive_answer(drop_json=args.input_json,
                                     output_json=args.output_json,
                                     prune_data=args.prune_data)


