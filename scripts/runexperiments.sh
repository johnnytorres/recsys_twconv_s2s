
# exit on error
set -e

BASE_DIR=$HOME/dataset
DATA_DIR=$BASE_DIR/twconv/trec
DATA_STAGGING=${DATA_DIR}/datastagging

RESULTS_DIR=$DATA_DIR/resultstagging/random
python twconvrecusers.task \
  --dataset-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  random

RESULTS_DIR=$DATA_DIR/resultstagging/tfidf
python twconvrecusers.task \
  --dataset-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  tfidf

RESULTS_DIR=$DATA_DIR/resultstagging/rnn
python twconvrecusers.task \
  --dataset-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  rnn \
  --train \
  --test \
  --embedding-trainable \
  --num-distractor=5 \
  --max-content-len=10 \
  --max-utterance-len=10 \
  --train-steps=50 \
  --train-batch-size=6 \
  --num-epochs=1 \
  --eval-every-secs=1 \
  --eval-batch-size=20 \
  --eval-steps=1

