#!/bin/bash

echo "Training local ML model"

MODEL_NAME="twconvrsu_v1" # change to your model name

PACKAGE_PATH=trainer
DATA_DIR=data/convusersec/twconvrsu_tf_v1
TRAIN_FILES=${DATA_DIR}/train.tfrecords
VALID_FILES=${DATA_DIR}/valid.tfrecords
TEST_FILES=${DATA_DIR}/tests.tfrecords
PREDICT_FILES=${DATA_DIR}/example.tfrecords
MODEL_DIR=${DATA_DIR}/models/${MODEL_NAME}
VOCAB_FILE=${DATA_DIR}/vocab_size.txt
VOCAB_PROC=${DATA_DIR}/vocab_processor.bin


gcloud ml-engine local train \
        --module-name=trainer.task \
        --package-path=${PACKAGE_PATH} \
        -- \
        --train-files=${TRAIN_FILES} \
        --eval-files=${VALID_FILES} \
        --tests-files=${TEST_FILES} \
        --vocab-path=${VOCAB_FILE} \
        --vocab-proc=${VOCAB_PROC} \
        --job-dir=${MODEL_DIR} \
        --file-encoding=tf \
        --train-batch-size=32 \
	    --train-steps=10 \
        --eval-batch-size=32 \
        --eval-steps=1 \
        --num-distractors=9 \
        --max-content-len=1400 \
        --max-utterance-len=1400 \
        --learning-rate=0.001 \
        --train \
        #--predict
        #--tests \
        #--predict-files=${PREDICT_FILES} \





#ls ${MODEL_DIR}/export/estimator
#MODEL_LOCATION=${MODEL_DIR}/export/estimator/$(ls ${MODEL_DIR}/export/estimator | tail -1)
#echo ${MODEL_LOCATION}
#ls ${MODEL_LOCATION}

# invoke trained model to make prediction given new data instances
#gcloud ml-engine local predict --model-dir=${MODEL_LOCATION} --example-instances=data/new-data.json