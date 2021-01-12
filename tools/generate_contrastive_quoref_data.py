import json
from collections import defaultdict
import argparse
import random

random.seed(53901)


def get_heuristic_contrastive_answers(paragraph_info):
    contrastive_qas = []
    if len(paragraph_info["qas"]) > 1:
        answer_grouped_questions = defaultdict(list)
        for qa_info in paragraph_info["qas"]:
            if len(qa_info["answers"]) == 1:
                answer = qa_info["answers"][0]
                answer_text = answer["text"]
                answer_key = (answer_text,)
            else:
                answer_key = tuple(answer["text"] for answer in qa_info["answers"])

            answer_grouped_questions[answer_key].append({"question": qa_info["question"],
                                                         "id": qa_info["id"]})

        if len(answer_grouped_questions) == 1:
            # This means there is only one answer. Let's skip.
            return contrastive_qas

        all_answers = list(answer_grouped_questions.keys())
        all_answer_entities = set()
        for answer in all_answers:
            for entity in answer:
                all_answer_entities.add(entity)

        for answer, questions in answer_grouped_questions.items():
            contrastive_answers = []
            if len(answer) == 1:
                for entity in all_answer_entities:
                    if entity != answer[0] and entity not in answer[0] and answer[0] not in entity:
                        contrastive_answers.append(entity)
            else:
                # Make sets of two entities, irrespective of the number of answer spans.
                entity_list = list(all_answer_entities)
                for i in range(len(entity_list) - 1):
                    if entity_list[i] in answer:
                        continue
                    for j in range(i+1, len(entity_list)):
                        if entity_list[j] in answer:
                            continue
                        contrastive_answers.append([entity_list[i], entity_list[j]])

            # Choosing a random element among the options because we can have only one contrastive answer as of
            # now. This should change.
            # TODO (pradeep): Use NER information to choose.
            if len(contrastive_answers) > 1:
                chosen_contrastive_answer = random.choice(contrastive_answers)
            elif contrastive_answers:
                chosen_contrastive_answer = contrastive_answers[0]
            else:
                chosen_contrastive_answer = None

            if isinstance(chosen_contrastive_answer, str):
                chosen_contrastive_answer = [chosen_contrastive_answer]
            # TODO: Make contrastive questions.
            for question_info in questions:
                new_question_info = {"question": question_info["question"],
                                     "query_id": question_info["id"]}
                new_question_info["answer"] = {"number": "",
                                               "date": {"day": "", "month": "", "year": ""},
                                               "spans": list(answer)}
                if chosen_contrastive_answer:
                    new_question_info["contrastive_answer"] = {"number": "",
                                                               "date": {"day": "", "month": "", "year": ""},
                                                               "spans": chosen_contrastive_answer}
                contrastive_qas.append(new_question_info)
    return contrastive_qas


def read_topk_predictions(topk_data, max_k=10):
    topk_contrastive_answers = {}
    for datum in topk_data:
        answer = datum["answer"]["value"]
        topk_answers = []
        logprobs = datum["all_answers"]["multi_span"]["logprobs"]
        answers = datum["all_answers"]["multi_span"]["values"]
        for logprob, candidate in list(zip(logprobs, answers))[:max_k]:
            if candidate != answer:
                topk_contrastive_answers[datum["passage_id"]] = {datum["query_id"]: candidate}

    print(f"Read a total of {len(topk_data)} data points")
    print(f"Found {len(topk_contrastive_answers)} within a top_k value of {max_k}")
    return topk_contrastive_answers


def get_topk_contrastive_answers(contrastive_answers, paragraph_info):
    contrastive_qas = []
    if paragraph_info["context_id"] in contrastive_answers:
        for qa_info in paragraph_info["qas"]:
            if qa_info["id"] in contrastive_answers[paragraph_info["context_id"]]:
                contrastive_spans = contrastive_answers[paragraph_info["context_id"]][qa_info["id"]]
                new_question_info = {"question": qa_info["question"],
                                     "query_id": qa_info["id"]}
                new_question_info["answer"] = {"number": "",
                                               "date": {"day": "", "month": "", "year": ""},
                                               "spans": [answer["text"] for answer in qa_info["answers"]],
                                               "contrastive_answer": {"number": "",
                                                                       "date": {"day": "", "month": "", "year": ""},
                                                                       "spans": contrastive_spans}}
                contrastive_qas.append(new_question_info)
    return contrastive_qas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--topk", type=str, nargs="+")

    args = parser.parse_args()

    data = json.load(open(args.input))

    topk_data = []
    if args.topk:
        for filename in args.topk:
            for line in open(filename):
                topk_data.append(json.loads(line))

        topk_contrastive_answers = read_topk_predictions(topk_data)

    output_data = {}

    for datum in data["data"]:
        for paragraph_info in datum["paragraphs"]:
            if args.topk:
                contrastive_qas = get_topk_contrastive_answers(topk_contrastive_answers, paragraph_info)
            else:
                contrastive_qas = get_heuristic_contrastive_answers(paragraph_info)
            if contrastive_qas:
                output_data[paragraph_info["context_id"]] = {"passage": paragraph_info["context"],
                                                             "qa_pairs": contrastive_qas}

    with open(args.output, "w") as outfile:
        json.dump(output_data, outfile, indent=2)


if __name__ == "__main__":
    main()
