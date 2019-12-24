#!/bin/bash
set -e # exit on error

[ "${DATASET}" == "" ] && echo "must specify dataset" && exit 1;
[ "${SUBSET}" == "" ] && echo "must specify subset" && exit 1;
[ "${MODEL}" == "" ] && echo "must specify model" && exit 1;
[ "${RUN}" == "" ] && RUN="local"
[ "${STORE}" == "" ] && STORE="local"
[ "${TAG}" == "" ] && TAG="" || TAG="_"${TAG}
[ "${EMBEDDING_PATH}" == "" ] && EMBEDDING_PATH="embeddings.vec"
[ "${EMBEDDING_ENABLED}" == "" ] && EMBEDDING_ENABLED=0
[ "${EMBEDDING_TRAINABLE}" == "" ] && EMBEDDING_TRAINABLE=0
[ "${REUSE_JOBDIR}" == "" ] && REUSE_JOBDIR=0

BUCKET="mlresearchbucket"
PACKAGE="twconvrecsys"

if [ ${STORE} == "gcloud" ]; then
	JOB_DIR="gs://${BUCKET}/${PACKAGE}/${DATASET}_${SUBSET}/${MODEL}${TAG}"
	echo "storing results on google cloud: $JOB_DIR"
else
	JOB_DIR=${HOME}/data/results/${PACKAGE}/${DATASET}_${SUBSET}/${MODEL}${TAG}
	echo "storing results locally: $JOB_DIR"
fi



if [ "${RUN}" == "gcloud" ]
		then
		echo 'training model on google cloud'
		CURRENT_DATE=`date +%Y%m%d_%H%M%S`
		JOB_NAME=train_${PACKAGE}_${DATASET}_${SUBSET}_${MODEL}_${TAG}_${CURRENT_DATE}
		echo "Job: ${JOB_NAME}"
		gcloud auth activate-service-account --key-file=gcloud/credentials.json
		#--stream-logs \
		gcloud ai-platform jobs submit training ${JOB_NAME} \
				--region="us-central1" \
				--runtime-version="1.14" \
				--module-name=${PACKAGE}.task \
				--package-path=${PACKAGE}  \
				--config=config.yaml \
				--scale-tier=BASIC_GPU \
				--job-dir=${JOB_DIR} \
				-- \
				--data-dir=${DATASET} \
				--data-subdir=${SUBSET} \
				--estimator=${MODEL} \
				--train-files=train.tfrecords \
				--eval-files=valid.tfrecords \
				--test-files=test.tfrecords \
				--vocab-path=vocabulary.txt \
				--embedding-path=${EMBEDDING_PATH} \
				--embedding-size=${EMBEDDING_SIZE} \
				--embedding-trainable=${EMBEDDING_TRAINABLE} \
				--embedding-enabled=${EMBEDDING_ENABLED} \
				--num-distractors=${NUM_DISTRACTORS} \
				--max-input-len=${MAX_INPUT_LEN} \
				--max-source-len=${MAX_SOURCE_LEN} \
				--max-target-len=${MAX_TARGET_LEN} \
				--train-batch-size=${TRAIN_BATCH_SIZE} \
				--num-epochs=${NUM_EPOCHS} \
				--eval-batch-size=${EVAL_BATCH_SIZE} \
				--learning-rate=${LEARNING_RATE} \
				--rnn-dim=${RNN_DIM} \
				--reuse-job-dir=${REUSE_JOBDIR} \
				--train \
				--test
else
		echo 'training model locally'
		python -m twconvrecsys.task \
				--data-dir=${DATASET} \
				--data-subdir=${SUBSET} \
				--job-dir=${JOB_DIR} \
				--estimator=${MODEL} \
				--train-files=train.tfrecords \
				--eval-files=valid.tfrecords \
				--test-files=test.tfrecords \
				--vocab-path=vocabulary.txt \
				--embedding-path=${EMBEDDING_PATH} \
				--embedding-size=${EMBEDDING_SIZE} \
				--embedding-trainable=${EMBEDDING_TRAINABLE} \
				--embedding-enabled=${EMBEDDING_ENABLED} \
				--num-distractors=${NUM_DISTRACTORS} \
				--max-input-len=${MAX_INPUT_LEN} \
				--max-source-len=${MAX_SOURCE_LEN} \
				--max-target-len=${MAX_TARGET_LEN} \
				--train-batch-size=${TRAIN_BATCH_SIZE} \
				--num-epochs=${NUM_EPOCHS} \
				--eval-batch-size=${EVAL_BATCH_SIZE} \
				--learning-rate=${LEARNING_RATE} \
				--rnn-dim=${RNN_DIM} \
				--reuse-job-dir=${REUSE_JOBDIR} \
				--train \
				--test
fi