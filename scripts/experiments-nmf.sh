
# exit on error
set -e

DATASET=2011_trec
MODEL=nmf
RUN_MODE=gcloud

export TRAIN_SIZE=20385
export TRAIN_BATCH_SIZE=64
export EVAL_BATCH_SIZE=128
export NUM_EPOCHS=50
export MAX_INPUT_LEN=120
export MAX_SOURCE_LEN=20
export MAX_TARGET_LEN=20
export NUM_DISTRACTORS=9
export EMBEDDING_SIZE=8
export LEARNING_RATE=0.0001

./scripts/experiments.sh ${DATASET} ${MODEL} ${RUN_MODE}
