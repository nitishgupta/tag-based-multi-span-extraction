#!/usr/bin/env

CONFIG_DIR=configs/drop/bert

SDIR=/home/nitishg/nfs2_nitishg/checkpoints/tase/drop
mkdir ${SDIR}/predictions

EVAL_FILE=drop_data/drop_dataset_train.json
# drop_dataset_train.json

METRICS_JSON=${SDIR}/predictions/metrics.json
PRED_FILE=${SDIR}/predictions/drop_train_topk_50.jsonl
# drop_train_topk_50.json

#srun --gpus 1 -w allennlp-server4 \
#  allennlp evaluate \
#  ${SDIR}/model.tar.gz \
#  ${EVAL_FILE} \
#  --cuda-device 0 \
#  --output-file ${METRICS_JSON} \
#  --include-package src


srun --gpus 1 -w allennlp-server4 \
  allennlp predict \
  ${SDIR}/model.tar.gz \
  ${EVAL_FILE} \
  --silent \
  --predictor drop_topk \
  --cuda-device 0 \
  --output-file ${PRED_FILE} \
  --use-dataset-reader \
  --overrides "{"model": {"output_all_answers": true, "topk_decoding": true}}" \
  --include-package src

# echo -e "Preds written to ${METRICS_JSON}"
echo -e "Preds written to ${PRED_FILE}"