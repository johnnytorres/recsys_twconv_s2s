
# exit on error
set -e

DATASET_NAME=$1
MODEL_NAME=$2 # change to your model name
RUN_MODE=$3

[ "${DATASET_NAME}" == "" ] && echo "must specify dataset" && exit 1;
[ "${SUBSET}" == "" ] && echo "must specify subset" && exit 1;
[ "${MODEL_NAME}" == "" ] && echo "must specify model" && exit 1;
[ "${RUN_MODE}" == "" ] && echo "must specify run mode" && exit 1;
[ "${TAG}" == "" ] && TAG=""

BUCKET="jtresearchbucket"
PACKAGE_NAME="twconvrecsys"
JOB_DIR="gs://${BUCKET}/${PACKAGE_NAME}/${DATASET_NAME}/${MODEL_NAME}${TAG}"
LOCAL_JOB_DIR=${HOME}/data/results/${PACKAGE_NAME}/${DATASET_NAME}/${MODEL_NAME}
REGION="us-central1"
PACKAGE_PATH=${PACKAGE_NAME} # this can be a gcs location to a zipped and uploaded package
CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${PACKAGE_NAME}_${DATASET_NAME}_${SUBSET}_${MODEL_NAME}${TAG}_${CURRENT_DATE}

echo "Job: ${JOB_NAME}"

if [ "${RUN_MODE}" == "gcloud" ]
	then
	echo 'running gcloud'
	gcloud auth activate-service-account --key-file=gcloud/credentials.json
	gcloud ai-platform jobs submit training ${JOB_NAME} \
					--stream-logs \
					--region=${REGION} \
					--runtime-version="1.14" \
					--module-name=${PACKAGE_PATH}.task \
					--package-path=${PACKAGE_PATH}  \
					--config=config.yaml \
					--job-dir=${JOB_DIR} \
					-- \
					--data-dir=${DATASET_NAME} \
					--data-subdir=${SUBSET} \
					--estimator=${MODEL_NAME} \
					--train-files=train.tfrecords \
					--eval-files=valid.tfrecords \
					--test-files=test.tfrecords \
					--vocab-path=vocabulary.txt \
					--num-distractors=${NUM_DISTRACTORS} \
					--max-input-len=${MAX_INPUT_LEN} \
					--max-source-len=${MAX_SOURCE_LEN} \
					--max-target-len=${MAX_TARGET_LEN} \
					--train-size=${TRAIN_SIZE} \
					--train-batch-size=${TRAIN_BATCH_SIZE} \
					--num-epochs=${NUM_EPOCHS} \
					--eval-batch-size=${EVAL_BATCH_SIZE} \
					--learning-rate=${LEARNING_RATE} \
					--embedding-size=${EMBEDDING_SIZE} \
					--rnn-dim=${RNN_DIM} \
					--train \
					--test
else
	if [ "${RUN_MODE}" == "glocal" ]
		then
		echo 'running gcloud locally'
		gcloud ai-platform local train \
				--module-name=${PACKAGE_PATH}.task \
				--package-path=${PACKAGE_PATH}  \
				--job-dir=${LOCAL_JOB_DIR} \
				-- \
				--data-dir=${DATASET_NAME} \
				--data-subdir=${SUBSET} \
				--estimator=${MODEL_NAME} \
				--train-files=train.tfrecords \
				--eval-files=valid.tfrecords \
				--test-files=test.tfrecords \
				--vocab-path=vocabulary.txt \
				--num-distractors=${NUM_DISTRACTORS} \
				--max-input-len=${MAX_INPUT_LEN} \
				--max-source-len=${MAX_SOURCE_LEN} \
				--max-target-len=${MAX_TARGET_LEN} \
				--train-size=${TRAIN_SIZE} \
				--train-batch-size=${TRAIN_BATCH_SIZE} \
				--num-epochs=${NUM_EPOCHS} \
				--eval-batch-size=${EVAL_BATCH_SIZE} \
				--learning-rate=${LEARNING_RATE} \
				--embedding-size=${EMBEDDING_SIZE} \
				--rnn-dim=${RNN_DIM} \
				--train \
				--test
	else
		echo 'running locally'
		python -m twconvrecsys.task \
				--data-dir=${DATASET_NAME} \
				--data-subdir=${SUBSET} \
				--job-dir=${LOCAL_JOB_DIR} \
				--estimator=${MODEL_NAME} \
				--train-files=train.tfrecords \
				--eval-files=valid.tfrecords \
				--test-files=test.tfrecords \
				--vocab-path=vocabulary.txt \
				--num-distractors=${NUM_DISTRACTORS} \
				--max-input-len=${MAX_INPUT_LEN} \
				--max-source-len=${MAX_SOURCE_LEN} \
				--max-target-len=${MAX_TARGET_LEN} \
				--train-size=${TRAIN_SIZE} \
				--train-batch-size=${TRAIN_BATCH_SIZE} \
				--num-epochs=${NUM_EPOCHS} \
				--eval-batch-size=${EVAL_BATCH_SIZE} \
				--learning-rate=${LEARNING_RATE} \
				--embedding-size=${EMBEDDING_SIZE} \
				--rnn-dim=${RNN_DIM} \
				--train \
				--test
	fi
fi