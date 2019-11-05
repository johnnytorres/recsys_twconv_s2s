IMAGE_REPO_NAME=twconvrecsys
IMAGE_REPO_TAG=latest
PROJECT_ID=jtresearch
IMAGE_URI=gcr.io/${PROJECT_ID}/${IMAGE_REPO_NAME}:${IMAGE_REPO_TAG}

docker-build:
	echo "building wheel..."
	python setup.py sdist bdist_wheel
	echo "building docker..."
	docker build -f Dockerfile -t ${IMAGE_URI} .