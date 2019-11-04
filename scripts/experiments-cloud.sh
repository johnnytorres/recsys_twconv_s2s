
# exit on error
set -e

echo "Submitting a Cloud ML Engine job..."

REGION="us-central1"
TIER="BASIC_GPU" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
MODEL_NAME="recsys_deepconv_tfidf" # change to your model name

PROJECT_ID=jtresearch
BUCKET=jtresearchbucket
DATA_DIR=gs://${BUCKET}/twconvrsu_tf_v1
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

gcloud auth activate-service-account --key-file=gcloud/iglesiaebg.json

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


# GCLOUD RUN

DATA_DIR=$BASE_DIR/twconv/2011_trec
DATA_STAGGING=${DATA_DIR}/staggingdata

RESULTS_DIR=$DATA_DIR/staggingresults/random
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  random

RESULTS_DIR=$DATA_DIR/staggingresults/tfidf
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  tfidf

RESULTS_DIR=$DATA_DIR/staggingresults/rnn
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  rnn \
  --train-files=train.tfrecords \
  --eval-files=valid.tfrecords \
  --test-files=test.tfrecords \
  --vocab-path=vocabulary.txt \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=20385 \
  --train-batch-size=64 \
  --num-epochs=15 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=300 \
  --rnn-dim=300 \
  --train \
  --test

RESULTS_DIR=$DATA_DIR/staggingresults/lstm
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  lstm \
  --train-files=train.tfrecords \
  --eval-files=valid.tfrecords \
  --test-files=test.tfrecords \
  --vocab-path=vocabulary.txt \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=20385 \
  --train-batch-size=64 \
  --num-epochs=15 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=300 \
  --rnn-dim=300 \
  --train \
  --test

RESULTS_DIR=$DATA_DIR/staggingresults/bilstm
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  bilstm \
  --train-files=train.tfrecords \
  --eval-files=valid.tfrecords \
  --test-files=test.tfrecords \
  --vocab-path=vocabulary.txt \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=20385 \
  --train-batch-size=64 \
  --num-epochs=15 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=300 \
  --rnn-dim=300 \
  --train \
  --test

RESULTS_DIR=$DATA_DIR/staggingresults/mf
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  mf \
  --train-files=train.tfrecords \
  --eval-files=valid.tfrecords \
  --test-files=test.tfrecords \
  --vocab-path=vocabulary.txt \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=20385 \
  --train-batch-size=64 \
  --num-epochs=100 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=8 \
  --train \
  --test

RESULTS_DIR=$DATA_DIR/staggingresults/nmf
mkdir -p ${RESULTS_DIR}
python -m twconvrecusers.task \
  --data-dir=${DATA_STAGGING} \
  --job-dir=${RESULTS_DIR} \
  nmf \
  --train-files=train.tfrecords \
  --eval-files=valid.tfrecords \
  --test-files=test.tfrecords \
  --vocab-path=vocabulary.txt \
  --num-distractors=9 \
  --max-content-len=120 \
  --max-utterance-len=120 \
  --train-size=20385 \
  --train-batch-size=64 \
  --num-epochs=50 \
  --eval-batch-size=128 \
  --learning-rate=0.0001 \
  --embedding-size=8 \
  --train \
  --test
