
# parameters for docker
IMAGE_REPO_NAME=twconvrecsys
IMAGE_REPO_TAG=latest
PROJECT_ID=jtresearch
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_REPO_TAG}

# parameters for experiments
DATASET=twconv_2011_trec # (ej twconv_2011_trec or twconv_2016_usersec)
SUBSET=sampledataconvs # specify subset (sample or alldata)
MODEL=rnn # specify the model (rnn, lstm, bilstm)
RUN="local" # local or gcloud
STORE="local" # store results local or gcloud
REUSE_JOBDIR=0
EMBEDDING_ENABLED=0
EMBEDDING_TRAINABLE=0

TRAIN_BATCH_SIZE=64
EVAL_BATCH_SIZE=128
NUM_EPOCHS=15
MAX_INPUT_LEN=120
MAX_SOURCE_LEN=120
MAX_TARGET_LEN=120
NUM_DISTRACTORS=9
EMBEDDING_SIZE=300
RNN_DIM=300
LEARNING_RATE=0.0001

export DATASET
export SUBSET
export MODEL
export RUN
export STORE
export REUSE_JOBDIR
export EMBEDDING_ENABLED
export EMBEDDING_TRAINABLE
export TRAIN_BATCH_SIZE
export EVAL_BATCH_SIZE
export NUM_EPOCHS
export MAX_INPUT_LEN
export MAX_SOURCE_LEN
export MAX_TARGET_LEN
export NUM_DISTRACTORS=9
export EMBEDDING_SIZE=300
export RNN_DIM=300
export LEARNING_RATE=0.0001


docker-build:
	echo "building wheel..."
	python setup.py sdist bdist_wheel
	echo "building docker..."
	docker build -f Dockerfile -t ${IMAGE_URI} .

run:
	./scripts/experiments.sh


