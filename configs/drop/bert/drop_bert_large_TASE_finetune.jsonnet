local config = import '../abstract/drop_TASE_BIO_SSE.jsonnet';

config {
    "pretrained_model": "bert-large-uncased-whole-word-masking",
    "bert_dim": 1024,
    "iterator"+: {
        "batch_size": 2
    },
    "trainer"+: {
        "optimizer"+: {
            "lr": 1e-05
        },
        "num_steps_to_accumulate": 6
    },
    "train_data_path": "drop_data/datecompare_train.json",
    "validation_data_path": "drop_data/datecompare_dev.json",

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
