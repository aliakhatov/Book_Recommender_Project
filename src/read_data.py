import logging.config
from typing import Tuple

import pandas as pd
import botocore.exceptions

logger = logging.getLogger(__name__)


def read_raw_files(file_dir: str,
                   df_ratings: str,
                   df_books: str,
                   df_users: str,
                   **kwargs) -> \
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Used for reading raw datasets from the given directory
    Args:
        file_dir(`str`): s3 path to read the data from
        df_ratings(`str`): Ratings csv file name
        df_books(`str`): Books csv file name
        df_users(`str`): Users csv file name
        **kwargs (`dict`): specified parameters passed to read_csv
    Returns:
        ratings_df(`pd.DataFrame`): ratings dataframe
        books_df(`pd.DataFrame`): books dataframe
        users_df(`pd.DataFrame`): users dataframe
    """
    try:
        try:

            ratings_df = pd.read_csv(f"{file_dir}{df_ratings}",
                                     **kwargs)
            logger.info("%s file is successfully read", df_ratings)

        except FileNotFoundError:
            logger.error("Please double check provided file path: %s and filename:%s",
                         file_dir, df_ratings)
            raise

        try:
            books_df = pd.read_csv(f"{file_dir}{df_books}", **kwargs)

        except FileNotFoundError:
            logger.error("Please double check provided file path: %s and filename:%s",
                         file_dir, df_books)
            raise

        try:
            users_df = pd.read_csv(f"{file_dir}{df_users}", **kwargs)
            logger.info("%s file is successfully read", df_users)

        except FileNotFoundError:
            logger.error("Please double check provided file path: %s and filename:%s",
                         file_dir, df_users)
            raise

    except botocore.exceptions.NoCredentialsError:
        logger.warning("Please provide AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")

    else:
        return ratings_df, books_df, users_df


def read_dataframe(file_dir: str, file_name: str) -> pd.DataFrame:
    """ Reads a dataframe object and return that dataframe
    Args:
        file_dir(`str`): Directory of the file that's being read
        file_name(`str`): The name of the file that's being read
    Returns:
        dataframe(`pd.DataFrame`): the read dataframe
    """
    try:
        file_path = f"{file_dir}{file_name}"
        dataframe = pd.read_csv(file_path)
    except FileNotFoundError:
        logger.error("Please double check your path and the file name")
        raise
    except botocore.exceptions.NoCredentialsError:
        logger.error("Data download failed. Please provide AWS_ACCESS_KEY_ID "
                     "and AWS_SECRET_ACCESS_KEY")

    else:
        logger.info("%s%s dataframe is successfully read", file_dir, file_name)
        return dataframe
