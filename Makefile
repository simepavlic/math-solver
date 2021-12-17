docker-build: ## Build docker container
		docker build -t my-docker-api -f docker/Dockerfile .

docker-run: docker-build
		docker run -d --publish 5000:5000 my-docker-api