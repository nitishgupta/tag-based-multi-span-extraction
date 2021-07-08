local config = import '../abstract/drop_TASE_BIO_SSE.jsonnet';

config {
    "pretrained_model": "bert-large-uncased-whole-word-masking",
    "bert_dim": 1024,

    "multi_span_training_style": "soft_em",
    "pspan_span_training_style": "soft_em",
    "qspan_span_training_style": "soft_em",
    "count_training_style": "soft_em",
    "arithmetic_training_style": "soft_em",

    "model"+: {
        "training_style": "contrastive"
    },

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
    "train_data_path": "drop_data/drop_dataset_train_topkv1.json",
    "validation_data_path": "drop_data/drop_dataset_dev.json",

    "model"+: {
        "initializer":
        [
            [".*",
                 {
                     "type": "pretrained",
                     "weights_file_path": "/home/nitishg/nfs2_nitishg/checkpoints/tase/drop/best.th"
                 },
            ],
        ]
    }
}
