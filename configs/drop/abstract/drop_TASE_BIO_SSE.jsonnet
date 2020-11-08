local config = import "drop_model.jsonnet";
config {
    "multi_span_training_style": "contrastive",
    "pspan_span_training_style": "contrastive",
    "qspan_span_training_style": "contrastive",
    "arithmetic_training_style": "soft_em",

    "dataset_reader"+: {
        "answer_generator_names_per_type": {
            "multiple_span": ["tagged_answer"],
            "single_span": ["tagged_answer", "passage_span_answer", "question_span_answer"],
            "number": ["arithmetic_answer", "count_answer", "passage_span_answer", "question_span_answer", "tagged_answer"],
            "date": ["arithmetic_answer", "passage_span_answer", "question_span_answer", "tagged_answer"]
        },
        "pickle"+: {
            "file_name": "all_heads_" + $.pretrained_model,
        },
        "max_instances": 100
    },
    "model"+: {
        "heads": {
            "passage_span": $.heads.passage_span,
            "question_span": $.heads.question_span,
            "multi_span": $.heads.multi_span,
            "arithmetic": $.heads.arithmetic,
            "count": $.heads.count,
        }
    }
}
