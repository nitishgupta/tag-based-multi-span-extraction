local config = import '../abstract/drop_TASE_BIO_SSE.jsonnet';

config {
    "pretrained_model": "bert-large-uncased-whole-word-masking",
    "bert_dim": 1024,

    "multi_span_training_style": "contrastive",
    "pspan_span_training_style": "contrastive",
    "qspan_span_training_style": "contrastive",
    "count_training_style": "contrastive",
    "arithmetic_training_style": "soft_em",

    "iterator"+: {
        "batch_size": 2
    },
    "trainer"+: {
        "optimizer"+: {
            "lr": 1e-05
        },
        "num_steps_to_accumulate": 6
    },

    "dataset_reader"+: {
        "max_instances": -1
    },
    "train_data_path": "drop_data/drop_dataset_train.json",
    "validation_data_path": "drop_data/drop_dataset_dev.json",
}
