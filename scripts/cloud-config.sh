
# GCLOUD CONFIG

# GOOGLE CLOUD AUTHENTICATION
# install gcloud sdk https://cloud.google.com/sdk/docs/quickstart-macos
gcloud auth list
#gcloud auth login `ACCOUNT`
gcloud auth login iglesiaebg
# OR to change the current active account
#gcloud config set account `ACCOUNT`
gcloud config set account iglesiaebg
# GOOGLE CLOUD PROJECT
gcloud projects list
#gcloud config set project `PROJECT_NAME`
gcloud config set project jtresearch
# GOOGLE CLOUD TO RUN ML ENGINE
# to be able to run in google cloud, we need to configure authentication
# create user
#gcloud iam service-accounts create [ACCOUNT_NAME]
gcloud iam service-accounts create jtresearcher
#assign role
#gcloud projects add-iam-policy-binding [PROJECT_ID] --member "serviceAccount:[ACCOUNT_NAME]@[PROJECT_ID].iam.gserviceaccount.com" --role "roles/owner"
gcloud projects add-iam-policy-binding jtresearch --member "serviceAccount:jtresearcher@jtresearch.iam.gserviceaccount.com" --role "roles/owner"
#gcloud projects add-iam-policy-binding jtresearch --member "serviceAccount:jtresearcher@jtresearch.iam.gserviceaccount.com" --role "roles/storage.objectAdmin"
#get key
gcloud iam service-accounts keys create gcloud/jtresearcher.json --iam-account jtresearcher@jtresearch.iam.gserviceaccount.com
#config custom docker
gcloud auth configure-docker
