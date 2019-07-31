#!/bin/bash

# HYPERPARAMS
# - tweets tokenizer
# - no embeddings

echo "Submitting a Cloud ML Engine job..."

REGION="us-central1"
TIER="BASIC_GPU" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1

MODEL_NAME="twconvrsu_v1d2" # change to your model name

PROJECT_ID=$(gcloud config list project --format "value(core.project)")
#BUCKET=${PROJECT_ID}-chatbot
BUCKET=${PROJECT_ID}-chatbot
DATA_DIR=gs://${BUCKET}/twconvrsu_tf_v2
MODEL_DIR=${DATA_DIR}/models/${MODEL_NAME}
#BUCKET="you-bucket-name" # change to your bucket name

PACKAGE_PATH=trainer # this can be a gcs location to a zipped and uploaded package
TRAIN_FILES=${DATA_DIR}/train.tfrecords
VALID_FILES=${DATA_DIR}/valid.tfrecords
TEST_FILES=${DATA_DIR}/tests.tfrecords
PREDICT_FILES=${DATA_DIR}/example.tfrecords
VOCAB_FILE=${DATA_DIR}/vocabulary.txt
VOCAB_PROC=${DATA_DIR}/vocab_processor.bin
EMBEDDING_FILE=${DATA_DIR}/embeddings.vec

CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${TIER}_${CURRENT_DATE}
#JOB_NAME=tune_${MODEL_NAME}_${CURRENT_DATE} # for hyper-parameter tuning jobs

echo "Model: ${JOB_NAME}"

gcloud ml-engine jobs submit training ${JOB_NAME} \
        --stream-logs \
        --job-dir=${MODEL_DIR} \
        --runtime-version=1.7 \
        --region=${REGION} \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH}  \
        --config=config.yaml \
        -- \
        --train-files=${TRAIN_FILES} \
        --eval-files=${VALID_FILES} \
        --tests-files=${TEST_FILES} \
        --vocab-path=${VOCAB_FILE} \
        --vocab-proc=${VOCAB_PROC} \
        --job-dir=${MODEL_DIR} \
        --file-encoding=tf \
        --num-epochs=-1 \
        --train-batch-size=128 \
	    --train-steps=2000 \
        --eval-batch-size=128 \
        --eval-steps=10 \
        --eval-every-secs=600 \
        --max-content-len=1400 \
        --max-utterance-len=1400 \
        --embedding-size=300 \
        --num-distractors=9 \
        --learning-rate=0.001 \
        --train \
        --tests \
        #--predict
        #        --embedding-path=${EMBEDDING_FILE} \
        #--predict-files=${PREDICT_FILES} \

# notes:
# use --packages instead of --package-path if gcs location
# add --reuse-job-dir to resume training