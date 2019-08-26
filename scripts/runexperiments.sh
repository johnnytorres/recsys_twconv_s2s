
# exit on error
set -e

BASE_DIR=$HOME/dataset
DATA_DIR=$BASE_DIR/twconv/trec
DATA_STAGGING=${DATA_DIR}/staggingdata

RESULTS_DIR=$DATA_DIR/staggingresults/random
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  random

RESULTS_DIR=$DATA_DIR/resultstagging/tfidf
python twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  tfidf

RESULTS_DIR=$DATA_DIR/resultstagging/rnn
python twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
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

