.PHONY: setup synthetic-data data baseline model explain api demo test docker-build docker-run

setup:
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt

# Only needed if you don't have the real Kaggle CSV yet -- writes a
# placeholder dataset to data/raw/cs-training.csv so you can exercise
# the pipeline before downloading the real thing.
synthetic-data:
	python -m scripts.generate_synthetic_data

data:
	python -m src.data_prep

baseline:
	python -m src.train_baseline

model:
	python -m src.train_model

explain:
	python -m src.explain

api:
	uvicorn src.api:app --reload

demo:
	streamlit run streamlit_app.py

test:
	pytest tests/ -v

docker-build:
	docker build -t credit-risk-api .

docker-run:
	docker run -p 8000:8000 credit-risk-api
