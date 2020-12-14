local config = import '../abstract/drop_TASE_BIO_SSE.jsonnet';

config {
    "pretrained_model": "bert-base-uncased",
    "bert_dim": 768,

    "multi_span_training_style": "topk_contrastive",
    "pspan_span_training_style": "topk_contrastive",
    "qspan_span_training_style": "topk_contrastive",
    "count_training_style": "topk_contrastive",
    "arithmetic_training_style": "soft_em",

    "iterator"+: {
        "batch_size": 2
    },
    "trainer"+: {
        "cuda_device": -1,
        "optimizer"+: {
            "lr": 1e-05
        },
        "num_steps_to_accumulate": 1,
    },

    "train_data_path": "drop_data/drop_sample_train.json",
    "validation_data_path": "drop_data/drop_dataset_dev.json",
    "dataset_reader"+: {
        "max_instances": 100
    },
}

// rm -rf /tmp/drop/; allennlp train configs/drop/bert/drop_sample.jsonnet -s /tmp/drop --include-package src
