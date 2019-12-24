
# exit on error
set -e

DATASET=2011_trec
MODEL=mf
[[ "$1" != "" ]] && RUN_MODE=$1 || RUN_MODE=local

export TRAIN_SIZE=20385
export TRAIN_BATCH_SIZE=64
export EVAL_BATCH_SIZE=128
export NUM_EPOCHS=50
export MAX_INPUT_LEN=120
export MAX_SOURCE_LEN=120
export MAX_TARGET_LEN=120
export NUM_DISTRACTORS=9
export EMBEDDING_SIZE=16
export LEARNING_RATE=0.0001

./scripts/run.sh ${DATASET} ${MODEL} ${RUN_MODE}
