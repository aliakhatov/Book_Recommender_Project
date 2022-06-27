import argparse
import logging.config
import yaml

from src.acquire_upload import upload_files_to_s3, download_from_web
from src.books_manager import BookManager, create_db
from config.flaskconfig import SQLALCHEMY_DATABASE_URI

logging.config.fileConfig("config/logging/local.conf")
logger = logging.getLogger("rds-pipeline")

if __name__ == "__main__":

    # Add parsers for both preprocessing the data, creating a database and adding the tables to it
    parser = argparse.ArgumentParser(
        description="Preprocess data, create and/or add data to database")

    parser.add_argument("step",
                        default="download",
                        help="Which step to run",
                        choices=["download", "create_db", "ingest_db"])

    parser.add_argument("--s3_dir",
                        help="s3 directory for uploading data to")

    parser.add_argument("--local_dir",
                        help="local data directory to store or upload the files")

    parser.add_argument("--processed_dir",
                        help="local data directory to store or upload the files")

    parser.add_argument("--engine_string", default=SQLALCHEMY_DATABASE_URI,
                        help="SQLAlchemy connection URI for database")

    parser.add_argument("--config",
                        default="config/config.yaml",
                        help="Directory of configuration file")

    parser.add_argument("--download_only", default=False, action="store_true",
                        help="If True, will not upload data to the S3 bucket. "
                             "If False, will upload data to S3")
    args = parser.parse_args()

    try:
        with open(args.config, "r") as f:
            yaml_config = yaml.safe_load(f)
    except yaml.error.YAMLError as e:
        logger.error("Error while loading configuration from %s", args.config)
    else:
        logger.info("Configuration file loaded from %s", args.config)

    if args.step == "download":
        # downloading the data from the web url
        download_from_web(args.local_dir, yaml_config["DEFAULT_WEB_URL"])
        if not args.download_only:
            # uploading the downloaded data to s3
            upload_files_to_s3(args.local_dir, args.s3_dir)

    elif args.step == "create_db":
        # creating database with the given SQLALCHEMY_DATABASE_URI
        create_db(args.engine_string)

    elif args.step == "ingest_db":

        # ingesting to database with given SQLALCHEMY_DATABASE_URI
        book_manager = BookManager(engine_string=args.engine_string)
        book_manager.persist_to_rds(args.processed_dir,
                                    yaml_config["clean_file_names"]["top_n_books"])
        book_manager.close()

