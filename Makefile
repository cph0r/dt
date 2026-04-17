install:
	pip install -e .

test:
	pytest

run:
	uvicorn app.main:app --reload

docker-build:
	docker build -t support-agentic-rag .
