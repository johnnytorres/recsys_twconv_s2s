
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
	--data-dir=~/data/twconv/twconv_2016_usersec/alldataconvs \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldataconvs/bilstm

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2016_usersec/alldatausers \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v9_alldatausers



