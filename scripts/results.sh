
# exit on error
set -e


gcloud auth activate-service-account --key-file=gcloud/credentials.json


BUCKET='mlresearchbucket'
DATASET='twconv_2011_trec_v9_alldataconvs'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}

mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv
gsutil cp ${GCLOUD_DIR}/lstm/predictions.csv  ${RESULTS_DIR}/lstm/predictions.csv
gsutil cp ${GCLOUD_DIR}/bilstm/predictions.csv  ${RESULTS_DIR}/bilstm/predictions.csv

BUCKET='mlresearchbucket'
DATASET='twconv_2011_trec_v9_alldatausers'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}

mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv
gsutil cp ${GCLOUD_DIR}/lstm/predictions.csv  ${RESULTS_DIR}/lstm/predictions.csv
gsutil cp ${GCLOUD_DIR}/bilstm/predictions.csv  ${RESULTS_DIR}/bilstm/predictions.csv

BUCKET='mlresearchbucket'
DATASET='twconv_2016_usersec_v9_alldataconvs'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}

mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv
gsutil cp ${GCLOUD_DIR}/lstm/predictions.csv  ${RESULTS_DIR}/lstm/predictions.csv
gsutil cp ${GCLOUD_DIR}/bilstm/predictions.csv  ${RESULTS_DIR}/bilstm/predictions.csv


BUCKET='mlresearchbucket'
DATASET='twconv_2016_usersec_v9_alldatausers'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}

mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv
gsutil cp ${GCLOUD_DIR}/lstm/predictions.csv  ${RESULTS_DIR}/lstm/predictions.csv
gsutil cp ${GCLOUD_DIR}/bilstm/predictions.csv  ${RESULTS_DIR}/bilstm/predictions.csv

# get results for embeddings

BUCKET='mlresearchbucket'
DATASET='twconv_2011_trec_v10_alldataconvs'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}

mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv
gsutil cp ${GCLOUD_DIR}/rnn_embeddings/predictions.csv  ${RESULTS_DIR}/rnn_embeddings/predictions.csv
gsutil cp ${GCLOUD_DIR}/rnn_embeddings_trainable/predictions.csv  ${RESULTS_DIR}/rnn_embeddings_trainable/predictions.csv


BUCKET='mlresearchbucket'
DATASET='twconv_2016_usersec_v10_alldataconvs'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}

mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn_embeddings/predictions.csv  ${RESULTS_DIR}/rnn_embeddings/predictions.csv
gsutil cp ${GCLOUD_DIR}/rnn_embeddings_trainable/predictions.csv  ${RESULTS_DIR}/rnn_embeddings_trainable/predictions.csv

# get results for test size

BUCKET='mlresearchbucket'
DATASET='twconv_2011_trec_v10_alldataconvs_test20'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}
mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv

BUCKET='mlresearchbucket'
DATASET='twconv_2011_trec_v10_alldataconvs_test30'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}
mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv

BUCKET='mlresearchbucket'
DATASET='twconv_2011_trec_v10_alldataconvs_test40'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}
mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv

BUCKET='mlresearchbucket'
DATASET='twconv_2016_usersec_v10_alldataconvs_test20'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}
mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv

BUCKET='mlresearchbucket'
DATASET='twconv_2016_usersec_v10_alldataconvs_test30'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}
mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv

BUCKET='mlresearchbucket'
DATASET='twconv_2016_usersec_v10_alldataconvs_test40'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}
mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv


#################################################################################
# generate reports

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec/cft \
	--job-dir=~/data/results/twconvrecsys/twconvrecsys_results_twconv_2011_trec_cftopics

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2016_usersec/cft \
	--job-dir=~/data/results/twconvrecsys/twconvrecsys_results_twconv_2016_usersec_cftopics


python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec/alldataconvs \
	--job-dir=~/data/results/twconvrecsys/twconv_2011_trec_v9_alldataconvs

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec/alldatausers \
	--job-dir=~/data/results/twconvrecsys/twconv_2011_trec_v9_alldatausers

python -m twconvrecsys.metrics.report \
	--data-dir=~/.keras/datasets/twconv_2016_usersec_v9 \
	--data-subdir=alldataconvs \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2016_usersec/alldatausers \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers


# report for embeddings

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec \
	--data-subdir=alldataconvs \
	--job-dir=~/data/results/twconvrecsys/twconv_2011_trec_v10_alldataconvs/rnn

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec \
	--data-subdir=alldataconvs \
	--job-dir=~/data/results/twconvrecsys/twconv_2011_trec_v10_alldataconvs/rnn_embeddings

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec \
	--data-subdir=alldataconvs \
	--job-dir=~/data/results/twconvrecsys/twconv_2011_trec_v10_alldataconvs/rnn_embeddings_trainable


# report for size

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec \
	--data-subdir=alldataconvs_test20 \
	--job-dir=~/data/results/twconvrecsys/twconv_2011_trec_v10_alldataconvs_test20/rnn
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec \
	--data-subdir=alldataconvs_test30 \
	--job-dir=~/data/results/twconvrecsys/twconv_2011_trec_v10_alldataconvs_test30/rnn
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec \
	--data-subdir=alldataconvs_test40 \
	--job-dir=~/data/results/twconvrecsys/twconv_2011_trec_v10_alldataconvs_test40/rnn

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2016_usersec \
	--data-subdir=alldataconvs_test20 \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v10_alldataconvs_test20/rnn
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2016_usersec \
	--data-subdir=alldataconvs_test30 \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v10_alldataconvs_test30/rnn
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2016_usersec \
	--data-subdir=alldataconvs_test40 \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v10_alldataconvs_test40/rnn

# reports by domains

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs_uKarlaMoralesR \
	--data-subdir=lstm \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs_uKarlaMoralesR/lstm
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs_uMashiRafael \
	--data-subdir=lstm \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs_uMashiRafael/lstm
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs_uaguschmer \
	--data-subdir=lstm \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs_uaguschmer/lstm
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs_uintersect \
	--data-subdir=lstm \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs_uintersect/lstm

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers_uKarlaMoralesR \
	--data-subdir=lstm \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers_uKarlaMoralesR/lstm
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers_uMashiRafael \
	--data-subdir=lstm \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers_uMashiRafael/lstm
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers_uaguschmer \
	--data-subdir=lstm \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers_uaguschmer/lstm
python -m twconvrecsys.metrics.report \
	--data-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers_uintersect \
	--data-subdir=lstm \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers_uintersect/lstm