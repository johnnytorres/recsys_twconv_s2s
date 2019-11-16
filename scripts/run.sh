
# exit on error
set -e

DATASET=2011_trec
MODEL=nmf
RUNNER=gcloud

export NUM_DISTRACTORS=9
export MAX_SOURCE_LEN=120
export MAX_TARGET_LEN=120
export TRAIN_SIZE=20385
export TRAIN_BATCH_SIZE=64
export EVAL_BATCH_SIZE=128
export NUM_EPOCHS=15
export LEARNING_RATE=0.0001
export EMBEDDING_SIZE=300
export RNN_DIM=300

#./scripts/experiments.sh ${DATASET} ${MODEL} ${RUNNER}
#./scripts/experiments.sh ${DATASET} ${MODEL} ${RUNNER}
#./scripts/experiments.sh ${DATASET} ${MODEL} ${RUNNER}
#./scripts/experiments.sh ${DATASET} ${MODEL} ${RUNNER}
#./scripts/experiments.sh ${DATASET} ${MODEL} ${RUNNER}
#
#export NUM_EPOCHS=50
#./scripts/experiments.sh ${DATASET} ${MODEL} ${RUNNER}


export NUM_EPOCHS=50
export LEARNING_RATE=0.0001
export EMBEDDING_SIZE=8
export MAX_SOURCE_LEN=12
export MAX_TARGET_LEN=12
./scripts/experiments.sh ${DATASET} ${MODEL} ${RUNNER}
