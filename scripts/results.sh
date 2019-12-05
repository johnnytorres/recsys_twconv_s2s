
# exit on error
set -e


gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/rnn/predictions.csv ~/data/twconv/2011_trec/staggingresults/rnn/predictions.csv
gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/lstm/predictions.csv ~/data/twconv/2011_trec/staggingresults/lstm/predictions.csv
gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/bilstm/predictions.csv ~/data/twconv/2011_trec/staggingresults/bilstm/predictions.csv
gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/mf/predictions.csv ~/data/twconv/2011_trec/staggingresults/mf/predictions.csv
gsutil cp gs://jtresearchbucket/twconvrecsys/2011_trec/nmf/predictions.csv ~/data/twconv/2011_trec/staggingresults/nmf/predictions.csv


gsutil cp -r gs://jtresearchbucket/twconvrecsys/twconv_2011_trec_v3/rnn \
	~/data/results/twconvrecsys/twconv_2011_trec_v3/rnn