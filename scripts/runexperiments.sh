
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

RESULTS_DIR=$DATA_DIR/resultstagging/rnn
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  rnn \
  --train \
  --test \
  --num-distractors=9 \
  --max-content-len=160 \
  --max-utterance-len=160 \
  --train-size=35255 \
  --train-batch-size=64 \
  --num-epochs=1 \
  --eval-batch-size=128
  #--train-steps=550 \ will be calculated automatically
  #--eval-every-secs=5 \ default 1 will eval each checkpoint
  #--eval-steps=313 None will use all the dataset

