.PHONY: download, ingest, format, qa, chat

format:
	poetry run black .
	poetry run isort .

# Download model
download:
	poetry run python download_model.py

# Run question answer over your docs
qa:
	poetry run python question_answer_docs.py

# ingest docs
ingest:
	poetry run python ingest.py

# Run chatbot without your docs
chat:
	poetry run python chat.py

# Run question answer over your docs
server:
	poetry run python question_answer_docs_server.py

simple_server:
	poetry run python simple_server.py


