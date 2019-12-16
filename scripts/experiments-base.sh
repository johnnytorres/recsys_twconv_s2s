
# exit on error
set -e

[[ "$1" != "" ]] && DATASET=$1 || ( echo "error: specify dataset (ej twconv_2011_trec or twconv_2016_usersec)" && exit 1 )
[[ "$2" != "" ]] && SUBSET=$2 || ( echo "error: specify subset (sample or alldata)" && exit 1 )
[[ "$3" != "" ]] && MODEL=$3 || ( echo "error: specify model (tfidf or random)" && exit 1 )


#MODEL=tfidf
#export TAG=""
#export NUM_DISTRACTORS=9
#export MAX_INPUT_LEN=120
#export MAX_SOURCE_LEN=120
#export MAX_TARGET_LEN=120


python -m twconvrecsys.task \
	--data-dir=${DATASET} \
	--data-subdir=${SUBSET} \
	--job-dir=~/data/results/twconvrecsys/${DATASET}/${MODEL} \
	--estimator=${MODEL} \
#	--max-input-len=${MAX_INPUT_LEN} \
#	--max-source-len=${MAX_SOURCE_LEN} \
#	--max-target-len=${MAX_TARGET_LEN} \
#	--num-distractors=${NUM_DISTRACTORS}
