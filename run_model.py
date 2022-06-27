import logging.config
import argparse

import yaml
import joblib

from src.preprocess import rename_and_select, counting_ratings, transform_to_pivot, \
    save_df, get_top_books, save_txt
from src.read_data import read_raw_files, read_dataframe
from src.model import recommend_books, train_model, split_test_train, evaluate_model

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("model-pipeline")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="pipeline for book recommender model")
    # parser for running the specific parts of the pipeline
    parser.add_argument("step",
                        default="preprocess",
                        help="Which step to run",
                        choices=["preprocess", "generate_features",
                                 "train", "recommend", "evaluate"])

    parser.add_argument("--raw_dir", default="data/raw/",
                        help="directory to where the cleaned data will be saved")

    parser.add_argument("--s3_raw_dir",
                        help="s3 directory to download raw data from")

    parser.add_argument("--processed_dir", default="data/processed/",
                        help="directory to where the cleaned data will be saved")

    parser.add_argument("--results_dir", default="data/results/",
                        help="directory where the model evaluation and "
                             "recommendation will be saved")

    parser.add_argument("--config", default="config/config.yaml",
                        help="Path to configuration file")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        yaml_config = yaml.load(f, Loader=yaml.FullLoader)
    logger.info("Configuration file loaded from %s", args.config)

    # running preprocess step of the pipeline
    if args.step == 'preprocess':
        # reading the raw data from s3
        df_rating, df_book, df_user = read_raw_files(args.s3_raw_dir,
                                                     yaml_config["raw_file_names"]["ratings_file"],
                                                     yaml_config["raw_file_names"]["books_file"],
                                                     yaml_config["raw_file_names"]["users_file"],
                                                     **yaml_config["preprocess"]["reading_params"])
        # renaming columns and selecting specific users and books based on thresholds
        merged_df = rename_and_select(df_rating, df_book, df_user,
                                      yaml_config["preprocess"]["rating_cols"],
                                      yaml_config["preprocess"]["books_cols"],
                                      yaml_config["preprocess"]["user_cols"],
                                      yaml_config["preprocess"]["users_threshold"])

        final_df = counting_ratings(merged_df, yaml_config["preprocess"]["rating_threshold"])
        save_df(final_df, args.processed_dir, yaml_config["clean_file_names"]["clean_data"])

    # running generate_features step of the pipeline
    if args.step == "generate_features":
        # sourcing cleaned dataframe from the given path
        clean_df = read_dataframe(args.processed_dir, yaml_config["clean_file_names"]["clean_data"])

        # extracting top 20 books df and saving
        top_n_books = get_top_books(clean_df, yaml_config["TOP_N_BOOKS"],
                                    **yaml_config["preprocess"]["group_by"])
        save_df(top_n_books, args.processed_dir, yaml_config["clean_file_names"]["top_n_books"])

    elif args.step == "train":

        # reading clean data from the given path
        clean_df = read_dataframe(args.processed_dir, yaml_config["clean_file_names"]["clean_data"])
        train_df, test_df = split_test_train(clean_df,
                                             yaml_config["model"]['test_ratio'],
                                             yaml_config["generate_features"]["rating_threshold"],
                                             yaml_config["model"]["random_seed"])
        # transforming dataframe to pivot matrix
        train_matrix = transform_to_pivot(train_df)

        # save the train matrix and test dataframes
        save_df(train_matrix, args.processed_dir, yaml_config["clean_file_names"]["train_book_mx"])
        save_df(test_df, args.processed_dir, yaml_config["clean_file_names"]["test_data"])

        # training a model with specific parameters passed from yaml file on full dataframe
        model = train_model(train_matrix, **yaml_config["model"]["model_params"])

        # saving the model to the given path in config.yaml
        joblib.dump(model, yaml_config["model"]["train_model_path"])
        logger.info("Model is saved to %s", yaml_config["model"]["train_model_path"])

    elif args.step == "recommend":

        # sourcing datasets used for recommendations
        train_matrix = read_dataframe(args.processed_dir,
                                      yaml_config["clean_file_names"]["train_book_mx"])
        # loading the trained model from the path given in yaml file
        trained_model = joblib.load(yaml_config["model"]["train_model_path"], mmap_mode="r")
        # running the function that gives back recommendations based on user picks
        recommended_books = recommend_books(train_matrix,
                                            yaml_config["model"]["sample_titles"],
                                            trained_model)

        logger.info("User picked books: %s", yaml_config["model"]["sample_titles"])
        logger.info("Recommended books based on user picks: %s", recommended_books)

        REC_BOOKS = "".join(recommended_books)

        save_txt(REC_BOOKS, args.results_dir,
                 yaml_config["result_file_names"]["recommendations"])

    elif args.step == "evaluate":
        # sourcing datasets for model training
        train_matrix = read_dataframe(args.processed_dir,
                                      yaml_config["clean_file_names"]["train_book_mx"])
        test_df = read_dataframe(args.processed_dir, yaml_config["clean_file_names"]["test_data"])
        # sourcing fitted model
        trained_model = joblib.load(yaml_config["model"]["train_model_path"], mmap_mode="r")

        # calculating mean average precision
        precision, recall = evaluate_model(train_matrix, test_df, trained_model,
                                           yaml_config["model"]["number_of_evals"])
        # converting numeric floats to strings
        final_results = f"precision: {precision}; recall: {recall}"
        # saving the results to the specific directory
        save_txt(final_results, args.results_dir, yaml_config["result_file_names"]["metrics"])
