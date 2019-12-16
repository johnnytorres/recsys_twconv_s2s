
# exit on error
set -e


gcloud auth activate-service-account --key-file=gcloud/credentials.json

#gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/rnn/predictions.csv ~/data/twconv/2011_trec/staggingresults/rnn/predictions.csv
#gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/lstm/predictions.csv ~/data/twconv/2011_trec/staggingresults/lstm/predictions.csv
#gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/bilstm/predictions.csv ~/data/twconv/2011_trec/staggingresults/bilstm/predictions.csv
#gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/mf/predictions.csv ~/data/twconv/2011_trec/staggingresults/mf/predictions.csv
#gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/nmf/predictions.csv ~/data/twconv/2011_trec/staggingresults/nmf/predictions.csv
#
#
#gsutil cp -r gs://jtresearchbucket/twconvrecsys/twconv_2011_trec_v5/rnn \
#	~/data/results/twconvrecsys/twconv_2011_trec_v3

BUCKET='mlresearchbucket'
DATASET='twconv_2011_trec_v7'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}

mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv
gsutil cp ${GCLOUD_DIR}/lstm/predictions.csv  ${RESULTS_DIR}/lstm/predictions.csv
gsutil cp ${GCLOUD_DIR}/bilstm/predictions.csv  ${RESULTS_DIR}/bilstm/predictions.csv


python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec/alldata \
	--job-dir=~/data/results/twconvrecsys/twconv_2011_trec_v7

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2011_trec/cft \
	--job-dir=~/data/results/twconvrecsys/twconvrecsys_results_twconv_2011_trec_cftopics

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2016_usersec/cft \
	--job-dir=~/data/results/twconvrecsys/twconvrecsys_results_twconv_2016_usersec_cftopics


BUCKET='mlresearchbucket'
DATASET='twconv_2016_usersec_v8'
RESULTS_DIR=~/data/results/twconvrecsys/${DATASET}
GCLOUD_DIR=gs://${BUCKET}/twconvrecsys/${DATASET}

mkdir -p ${RESULTS_DIR}
gsutil cp ${GCLOUD_DIR}/rnn/predictions.csv  ${RESULTS_DIR}/rnn/predictions.csv
gsutil cp ${GCLOUD_DIR}/lstm/predictions.csv  ${RESULTS_DIR}/lstm/predictions.csv
gsutil cp ${GCLOUD_DIR}/bilstm/predictions.csv  ${RESULTS_DIR}/bilstm/predictions.csv


python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2016_usersec/alldata \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v8

python -m twconvrecsys.metrics.report \
	--data-dir=~/data/twconv/twconv_2016_usersec/cft \
	--job-dir=~/data/results/twconvrecsys/twconv_2016_usersec_v3/cft