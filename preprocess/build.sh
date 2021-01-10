PROJECT_ID=$(gcloud config get-value project)


gcloud builds submit --config config.yml