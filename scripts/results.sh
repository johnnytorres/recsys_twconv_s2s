
# exit on error
set -e


gcloud auth activate-service-account --key-file=gcloud/credentials.json

gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/rnn/predictions.csv ~/data/twconv/2011_trec/staggingresults/rnn/predictions.csv
gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/lstm/predictions.csv ~/data/twconv/2011_trec/staggingresults/lstm/predictions.csv
gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/bilstm/predictions.csv ~/data/twconv/2011_trec/staggingresults/bilstm/predictions.csv
gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/mf/predictions.csv ~/data/twconv/2011_trec/staggingresults/mf/predictions.csv
gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/nmf/predictions.csv ~/data/twconv/2011_trec/staggingresults/nmf/predictions.csv


gsutil cp -r gs://jtresearchbucket/twconvrecsys/twconv_2011_trec_v3/rnn \
	~/data/results/twconvrecsys/twconv_2011_trec_v3


#mkdir ~/data/results/twconvrecsys/twconv_2011_trec_v4

DATASET='twconv_2011_trec_v5'
gsutil cp -r gs://mlresearchbucket/twconvrecsys/${DATASET}/rnn/predictions.csv \
	~/data/results/twconvrecsys/rnn
mkdir -p 	~/data/results/twconvrecsys/${DATASET}/lstm/
gsutil cp -r gs://mlresearchbucket/twconvrecsys/${DATASET}/lstm/predictions.csv \
	~/data/results/twconvrecsys/${DATASET}/lstm/
mkdir -p 	~/data/results/twconvrecsys/${DATASET}/bilstm/
gsutil cp -r gs://mlresearchbucket/twconvrecsys/${DATASET}/bilstm/predictions.csv \
	~/data/results/twconvrecsys/${DATASET}/bilstm/