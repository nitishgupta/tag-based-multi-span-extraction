import json
import os
from typing import List, Dict, Union

def read_dataset(input_json):
    print("Reading DROP: {}".format(input_json))
    with open(input_json) as data_file:
        dataset = json.load(data_file)
    dataset_stats(dataset)
    return dataset

def write_dataset(dataset, output_json):
    print("Writing data to: {}".format(output_json))
    dataset_stats(dataset)
    output_dir = os.path.split(output_json)[0]
    os.makedirs(output_dir, exist_ok=True)

    with open(output_json, 'w') as outf:
        json.dump(dataset, outf, indent=4)
    print("Written!")

def dataset_stats(dataset):
    nump = len(dataset)
    numq = sum([len(pinfo["qa_pairs"]) for _, pinfo in dataset.items()])
    print("Num passages: {}  Num questions: {}".format(nump, numq))


def get_contrative_answer_dict(text_spans: Union[str, List[str]]):
    if isinstance(text_spans, str):
        text_spans = [text_spans]

    return {
            "number": "",
            "date": {
                "day": "",
                "month": "",
                "year": ""
            },
            "spans": text_spans
    }
