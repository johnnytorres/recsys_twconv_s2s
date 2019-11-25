
# exit on error
set -e

[[ "$1" != "" ]] && DATASET=$1 || DATASET=twconv_2011_trec
MODEL=rnn
RUN_MODE=gcloud

export TRAIN_SIZE=20385
export TRAIN_BATCH_SIZE=64
export EVAL_BATCH_SIZE=128
export NUM_EPOCHS=50
export MAX_INPUT_LEN=120
export MAX_SOURCE_LEN=120
export MAX_TARGET_LEN=120
export NUM_DISTRACTORS=9
export EMBEDDING_SIZE=300
export RNN_DIM=300
export LEARNING_RATE=0.0001

./scripts/experiments.sh ${DATASET} ${MODEL} ${RUN_MODE}
