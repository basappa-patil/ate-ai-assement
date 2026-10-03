.PHONY: setup generate-data train test run-app clean

setup:
	pip install -r requirements.txt

generate-data:
	python generate_test_data.py

train:
	python -m src.ate_ai.train --data data/raw/ATE_AI_Engineering_Assessment.csv

test:
	python -m pytest tests/ -v

run-app:
	streamlit run app/streamlit_app.py

clean:
	rm -rf data/raw/*.csv data/processed/*.csv models/*.joblib reports/*.json __pycache__ .pytest_cache
