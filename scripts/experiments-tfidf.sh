
# exit on error
set -e

[[ "$1" != "" ]] && DATASET=$1 || DATASET=twconv_2011_trec
#[[ "$2" != "" ]] && RUN_MODE=$2 || RUN_MODE=local
MODEL=tfidf
export NUM_DISTRACTORS=5
export MAX_INPUT_LEN=10
export MAX_SOURCE_LEN=10
export MAX_TARGET_LEN=10


python -m twconvrecsys.task \
	--data-dir=${DATASET} \
	--job-dir=~/data/results/twconvrecsys/tfidf \
	--estimator=${MODEL}
