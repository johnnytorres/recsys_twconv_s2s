
# exit on error
set -e

[[ "$1" != "" ]] && DATASET=$1 || DATASET=~/data/twconv_2011_trec/alldata
MODEL=tfidf
export NUM_DISTRACTORS=9
export MAX_INPUT_LEN=120
export MAX_SOURCE_LEN=120
export MAX_TARGET_LEN=120


python -m twconvrecsys.task \
	--data-dir=${DATASET} \
	--job-dir=~/data/results/twconvrecsys/tfidf \
	--estimator=${MODEL} \
	--max-input-len=${MAX_INPUT_LEN} \
	--max-source-len=${MAX_SOURCE_LEN} \
	--max-target-len=${MAX_TARGET_LEN} \
	--num-distractors=${NUM_DISTRACTORS}
