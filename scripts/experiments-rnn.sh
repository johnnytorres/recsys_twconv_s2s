
# exit on error
set -e

[[ "$1" != "" ]] && DATASET=$1 || ( echo "error: specify dataset (ej twconv_2011_trec or twconv_2016_usersec)" && exit 1 )
[[ "$2" != "" ]] && SUBSET=$2 || ( echo "error: specify subset (sample or alldata)" && exit 1 )
[[ "$3" != "" ]] && TRAIN_SIZE=$3 || ( echo "error: specify training size" && exit 1 )


MODEL=rnn
RUN_MODE=gcloud

export TAG=""

if [ ${SUBSET} == "sample" ]; then
	export TRAIN_SIZE
	export TRAIN_BATCH_SIZE=6
	export EVAL_BATCH_SIZE=20
	export NUM_EPOCHS=2
	export MAX_INPUT_LEN=10
	export MAX_SOURCE_LEN=10
	export MAX_TARGET_LEN=10
	export NUM_DISTRACTORS=5
	export EMBEDDING_SIZE=50
	export RNN_DIM=50
	export LEARNING_RATE=0.0001
	export SUBSET='sampledata'
else
	export TRAIN_SIZE
	export TRAIN_BATCH_SIZE=64
	export EVAL_BATCH_SIZE=128
	export NUM_EPOCHS=15
	export MAX_INPUT_LEN=120
	export MAX_SOURCE_LEN=120
	export MAX_TARGET_LEN=120
	export NUM_DISTRACTORS=9
	export EMBEDDING_SIZE=300
	export RNN_DIM=300
	export LEARNING_RATE=0.0001
	export SUBSET
fi

./scripts/experiments.sh ${DATASET} ${MODEL} ${RUN_MODE}
