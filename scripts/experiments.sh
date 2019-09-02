
# exit on error
set -e

BASE_DIR=$HOME/data
DATA_DIR=$BASE_DIR/twconv/trec
DATA_STAGGING=${DATA_DIR}/staggingdata

RESULTS_DIR=$DATA_DIR/staggingresults/random
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  random

RESULTS_DIR=$DATA_DIR/staggingresults/tfidf
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  tfidf

RESULTS_DIR=$DATA_DIR/staggingresults/rnn
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  rnn \
  --train-files=train.tfrecords \
  --eval-files=valid.tfrecords \
  --test-files=test.tfrecords \
  --vocab-path=vocabulary.txt \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=14575 \
  --train-batch-size=64 \
  --num-epochs=5 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=300 \
  --rnn-dim=300 \
  --train \
  --test

RESULTS_DIR=$DATA_DIR/staggingresults/lstm
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  lstm \
  --train-files=train.tfrecords \
  --eval-files=valid.tfrecords \
  --test-files=test.tfrecords \
  --vocab-path=vocabulary.txt \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=14575 \
  --train-batch-size=64 \
  --num-epochs=5 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=300 \
  --rnn-dim=300 \
  --train \
  --test \


RESULTS_DIR=$DATA_DIR/staggingresults/bilstm
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  bilstm \
  --train-files=train.tfrecords \
  --eval-files=valid.tfrecords \
  --test-files=test.tfrecords \
  --vocab-path=vocabulary.txt \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=14575 \
  --train-batch-size=64 \
  --num-epochs=5 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=300 \
  --rnn-dim=300 \
  --train \
  --test

RESULTS_DIR=$DATA_DIR/staggingresults/mf
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  mf \
  --train-files=train.tfrecords \
  --eval-files=valid.tfrecords \
  --test-files=test.tfrecords \
  --vocab-path=vocabulary.txt \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=14575 \
  --train-batch-size=64 \
  --num-epochs=5 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=300 \
  --rnn-dim=300 \
  --train \
  --test

