RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/"
RESULTS_DATA_DIR = "data/results/"
S3_BUCKET_RAW="s3://2022-msia-423-akhatov-alisher/raw/"
S3_BUCKET_PROCESSED="s3://2022-msia-423-akhatov-alisher/processed/"
ARGS := $()

.PHONY: image app tests

image:
	docker build -f dockerfiles/Dockerfile -t final-project .

app:
	docker build -f dockerfiles/Dockerfile.app -t final-project-app .

tests:
	docker build -f dockerfiles/Dockerfile.test -t final-project-tests .



.PHONY: acquire create_db ingest preprocess generate_features train recommend evaluate full_pipeline full_ingest

acquire:
	docker run --mount type=bind,source="$(shell pwd)/",target=/app/ \
		   -e AWS_ACCESS_KEY_ID \
		   -e AWS_SECRET_ACCESS_KEY \
		   final-project run_acquire.py download $(ARGS) --local_dir=${RAW_DATA_DIR} \
		   --s3_dir=${S3_BUCKET_RAW}

create_db:
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
		   -e SQLALCHEMY_DATABASE_URI \
		   final-project run_acquire.py create_db

data/raw/BX-Book-Ratings.csv:
	docker run --mount type=bind,source="$(shell pwd)/",target=/app/ \
			   final-project run_acquire.py download --local_dir=${RAW_DATA_DIR} \
			   --s3_dir=${S3_BUCKET_RAW}
data/raw/BX-Books.csv:
	docker run --mount type=bind,source="$(shell pwd)/",target=/app/ \
			   final-project run_acquire.py download --local_dir=${RAW_DATA_DIR} \
			   --s3_dir=${S3_BUCKET_RAW}
data/raw/BX-Users.csv:
	docker run --mount type=bind,source="$(shell pwd)/",target=/app/ \
			   final-project run_acquire.py download --local_dir=${RAW_DATA_DIR} \
			   --s3_dir=${S3_BUCKET_RAW}

data/processed/clean_df.csv: run_acquire.py config/config.yaml data/raw/BX-Book-Ratings.csv data/raw/BX-Books.csv data/raw/BX-Users.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
		   -e AWS_ACCESS_KEY_ID \
		   -e AWS_SECRET_ACCESS_KEY \
		   final-project run_model.py preprocess --raw_dir=${RAW_DATA_DIR}\
		   --processed_dir=${PROCESSED_DATA_DIR} --s3_raw_dir=${S3_BUCKET_RAW}

preprocess: data/processed/clean_df.csv

data/processed/top_n_books.csv: run_model.py data/processed/clean_df.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
		   final-project run_model.py generate_features  --processed_dir=${PROCESSED_DATA_DIR}

generate_features: data/processed/top_n_books.csv

ingest: data/processed/top_n_books.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
		   -e SQLALCHEMY_DATABASE_URI \
		   final-project run_acquire.py ingest_db --processed_dir=${PROCESSED_DATA_DIR}

models/nearest_neighbors_train.joblib data/processed/test_df.csv data/processed/train_pivot_matrix.csv: run_model.py data/processed/clean_df.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
		   final-project run_model.py train --processed_dir=${PROCESSED_DATA_DIR}

train: models/nearest_neighbors_train.joblib data/processed/test_df.csv data/processed/train_pivot_matrix.csv

data/results/recommendations.txt: models/nearest_neighbors_train.joblib data/processed/train_pivot_matrix.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
		   final-project run_model.py recommend --processed_dir=${PROCESSED_DATA_DIR} --results_dir=${RESULTS_DATA_DIR}
recommend: data/results/recommendations.txt

data/results/model_metrics.txt: models/nearest_neighbors_train.joblib data/processed/test_df.csv data/processed/train_pivot_matrix.csv
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
		   final-project run_model.py evaluate --processed_dir=${PROCESSED_DATA_DIR} --results_dir=${RESULTS_DATA_DIR}

evaluate: data/results/model_metrics.txt


full_pipeline: acquire preprocess generate_features train recommend evaluate
full_ingest: preprocess generate_features ingest


.PHONY: run-app run-tests
run-app: models/nearest_neighbors_train.joblib data/processed/train_pivot_matrix.csv data/processed/clean_df.csv create_db full_ingest generate_features
	docker run --mount type=bind,source="$(shell pwd)",target=/app/ \
			-e SQLALCHEMY_DATABASE_URI \
			-e AWS_ACCESS_KEY_ID \
		    -e AWS_SECRET_ACCESS_KEY \
			-p 5002:5002 final-project-app

run-tests:
	docker run final-project-tests

kill-app:
	docker kill final-project-app

.PHONY: clean_all_data clean_db clean_processed clean_raw
clean_all_data:
	rm 'data/raw/BX-Book-Ratings.csv'
	rm 'data/raw/BX-Books.csv'
	rm 'data/raw/BX-Users.csv'
	rm 'data/processed/clean_df.csv'
	rm 'data/processed/test_df.csv'
	rm 'data/processed/top_n_books.csv'
	rm 'data/processed/train_pivot_matrix.csv'
	rm 'data/book_recommender.db'
	rm 'data/results/model_metrics.txt'
	rm 'data/results/recommendations.txt'


clean_db:
	rm 'data/book_recommender.db'

clean_processed:
	rm 'data/processed/train_pivot_matrix.csv'
	rm 'data/processed/clean_df.csv'
	rm 'data/processed/top_n_books.csv'
	rm 'data/processed/test_df.csv'

clean_raw:
	rm 'data/raw/BX-Book-Ratings.csv'
	rm 'data/raw/BX-Books.csv'
	rm 'data/raw/BX-Users.csv'
