
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

# NEURAL MODELS

# debug
#--data-dir=~/data/twconv/trec/sampledata
#--job-dir=~/data/twconv/trec/sampleresults/rnn
#rnn
#--train
#--test
#--force-tb-logs
#--num-distractor=5
#--max-content-len=10
#--max-utterance-len=10
#--train-steps=50
#--train-batch-size=6
#--num-epochs=1
#--eval-every-secs=1
#--eval-batch-size=20
#--eval-steps=1

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
  #--train-steps=550 \ will be calculated automatically
  #--eval-every-secs=5 \ default 1 will eval each checkpoint
  #--eval-steps=313 None will use all the datasets


RESULTS_DIR=$DATA_DIR/staggingresults/lstm
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  lstm \
  --train \
  --test \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=14575 \
  --train-batch-size=64 \
  --num-epochs=5 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=300 \
  --rnn-dim=300

RESULTS_DIR=$DATA_DIR/staggingresults/bilstm
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  bilstm \
  --train \
  --test \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=14575 \
  --train-batch-size=64 \
  --num-epochs=5 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=300 \
  --rnn-dim=300

