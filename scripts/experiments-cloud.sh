
# exit on error
set -e

runner=$1

echo "Submitting a Cloud ML Engine job..."

REGION="us-central1"
TIER="BASIC" # BASIC | BASIC_GPU | STANDARD_1 | PREMIUM_1
MODEL_NAME="twconvrecsys_tfidf" # change to your model name
BUCKET=jtresearchbucket
JOB_DIR=gs://${BUCKET}/

PACKAGE_PATH=twconvrecsys # this can be a gcs location to a zipped and uploaded package
CURRENT_DATE=`date +%Y%m%d_%H%M%S`
JOB_NAME=train_${MODEL_NAME}_${TIER}_${CURRENT_DATE}

echo "Model: ${JOB_NAME}"

if [ "${runner}" == "cloud" ]
then
	echo 'running in the cloud'
	gcloud auth activate-service-account --key-file=gcloud/iglesiaebg.json
	gcloud ai-platform jobs submit training ${JOB_NAME} \
					--stream-logs \
					--runtime-version=1.7 \
					--region=${REGION} \
					--module-name=${PACKAGE_PATH}.task \
					--package-path=${PACKAGE_PATH}  \
					--config=config.yaml \
					--staging-bucket=${JOB_DIR} \
					-- \
					--dataset-name=trec \
					--job-dir=${JOB_DIR} \
					tfidf
else
	echo 'running locally'
	gcloud ai-platform local train \
					--module-name=${PACKAGE_PATH}.task \
					--package-path=${PACKAGE_PATH}  \
					-- \
					--dataset-name=trec \
					--job-dir=${JOB_DIR} \
					tfidf
fi



