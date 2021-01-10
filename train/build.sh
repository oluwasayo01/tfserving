PROJECT_ID=$(gcloud config get-value project)
IMAGE_NAME="cnn-train"
IMAGE_TAG="latest"

gcloud builds submit --config config.yml