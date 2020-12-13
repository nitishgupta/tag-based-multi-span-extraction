from typing import List, Dict, Tuple

import os
import sys
import json
import copy
import string
from enum import Enum
from collections import defaultdict
import random
import argparse

from allennlp.data.tokenizers import SpacyTokenizer

random.seed(100)
spacy_tokenizer = SpacyTokenizer()
IGNORED_TOKENS = {"a", "an", "the"}
STRIPPED_CHARACTERS = string.punctuation + "".join(["‘", "’", "´", "`", "_"])

project_root_dir = os.path.dirname(
    os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
)
print(project_root_dir)

sys.path.insert(0, project_root_dir)

from ce_data.drop.utils import read_dataset, dataset_stats, get_contrative_answer_dict, write_dataset
from src.data.dataset_readers.drop.drop_utils import extract_answer_info_from_annotation, get_target_numbers


""" This script is used to augment date-comparison-data by flipping events in the questions """

FIRST = "first"
SECOND = "second"

FIRST_operator_tokens = ["first", "earlier"]
SECOND_operator_tokens = ["later", "last", "second"]

NUMBER_COMPARISON = ["were there more", "were there fewer", "which age group", "which group"]


def is_num_comparison(question: str):
    question_lower = question.lower()
    if any(span in question_lower for span in NUMBER_COMPARISON):
        return True
    else:
        return False

def tokenize(text: str) -> List[str]:
    tokens = spacy_tokenizer.tokenize(text)
    return [t.text for t in tokens]


def quesEvents(question_tokenized: str) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """ For the "which group" questions output the two-relevant question attentions """
    or_split = question_tokenized.split(" or ")
    if len(or_split) != 2:
        return None
    tokens = question_tokenized.split(" ")
    or_idx = tokens.index("or")
    # Last token is ? which we don't want to attend to
    event2 = tokens[or_idx + 1 : len(tokens) - 1]
    event2_span = (or_idx + 1, len(tokens) - 1)
    # Gets first index of the item
    try:
        comma_idx = tokens.index(",")
    except:
        comma_idx = 100000
    try:
        colon_idx = tokens.index(":")
    except:
        colon_idx = 100000

    try:
        hyphen_idx = tokens.index("-")
    except:
        hyphen_idx = 100000

    split_idx = min(comma_idx, colon_idx, hyphen_idx)

    if split_idx == 100000 or (or_idx - split_idx <= 1):
        # print(f"{qstr} first_split:{split_idx} or:{or_idx}")
        if "more" in tokens:
            split_idx = tokens.index("more")
        elif "fewer" in tokens:
            split_idx = tokens.index("fewer")
        elif "last" in tokens:
            split_idx = tokens.index("last")
        elif "later" in tokens:
            split_idx = tokens.index("later")
        elif "larger" in tokens:
            split_idx = tokens.index("larger")
        else:
            split_idx = -1

    if split_idx == -1:
        print(f"Cannot split -- {question_tokenized} {split_idx} {or_idx}")
        return None
    event1_span = (split_idx + 1, or_idx)
    return event1_span, event2_span


def get_numcompare_contrastive_answer_dict(**kwargs):
    qa = kwargs["qa"]
    question = qa["question"]
    if not is_num_comparison(question):
        return None

    if 'answer' in qa and qa['answer']:
        answer = qa['answer']
    else:
        return None

    if not answer["spans"]:
        # No span answers
        return None

    question_tokens = tokenize(question)
    question_tokenized_text = " ".join(question_tokens)

    event_spans = quesEvents(question_tokenized_text)
    if event_spans is None:
        return None
    event1_span, event2_span = event_spans
    event1_tokens = question_tokens[event1_span[0]:event1_span[1]]
    event2_tokens = question_tokens[event2_span[0]:event2_span[1]]

    answer_text = answer["spans"][0]
    answer_tokens = tokenize(answer_text)

    event1, event2 = set(event1_tokens), set(event2_tokens)
    ans_event = 1 if len(event1.intersection(answer_tokens)) > len(event2.intersection(answer_tokens)) else 2

    if ans_event == 1:
        contrastive_answer_span = event2_span
    else:
        contrastive_answer_span = event1_span

    contrastive_answer_text = " ".join(question_tokens[contrastive_answer_span[0]:contrastive_answer_span[1]])
    contrastive_answer_text = contrastive_answer_text.strip()
    if contrastive_answer_text:
        return get_contrative_answer_dict(contrastive_answer_text)
    else:
        return None


def augmentNumComparisonData(drop_json, output_json):
    """
    """
    drop_data = read_dataset(drop_json)

    total_qa_w_contrastans = 0

    paras_to_remove = []
    for pid, passage_info in drop_data.items():
        passage = passage_info["passage"]
        new_qas = []
        for qa in passage_info["qa_pairs"]:
            question = qa["question"]
            if not is_num_comparison(question):
                continue
            if 'answer' in qa and qa['answer']:
                answer = qa['answer']
            else:
                continue

            if not answer["spans"]:
                # No span answers
                continue

            contrastive_answer_dict = get_numcompare_contrastive_answer_dict(qa=qa)
            if contrastive_answer_dict is None:
                continue
            qa["contrastive_answer"] = contrastive_answer_dict

            total_qa_w_contrastans += 1
            new_qas.append(qa)

        passage_info["qa_pairs"] = new_qas
        if not new_qas:
            paras_to_remove.append(pid)

    for pid in paras_to_remove:
        drop_data.pop(pid)

    print("Total questions with contrastive answers: {}".format(total_qa_w_contrastans))
    write_dataset(dataset=drop_data, output_json=output_json)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=str, help="Input data file")
    parser.add_argument(
        "output_json", type=str, help="Output data file"
    )
    args = parser.parse_args()

    augmentNumComparisonData(args.input_json, args.output_json)
