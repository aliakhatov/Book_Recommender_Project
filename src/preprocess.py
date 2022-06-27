import logging.config
from typing import Union
import pandas as pd

logger = logging.getLogger(__name__)


def rename_and_select(ratings_df: pd.DataFrame, books_df: pd.DataFrame,
                      user_df: pd.DataFrame, ratings_cols: dict,
                      book_cols: dict, users_cols: dict,
                      user_threshold: int) -> pd.DataFrame:
    """Generates a dataframe that has only top users based on specified threshold
    Args:
        ratings_df (`pd.DataFrame`): the raw rating data
        books_df (`pd.DataFrame`): the raw books data
        user_df (`pd.DataFrame`): the raw user data
        book_cols (`dict`): Columns used for renaming for books df
        ratings_cols (`dict`): Columns used for renaming for ratings df
        users_cols (`dict`): Columns used for renaming for users df
        user_threshold (`dict`): Threshold used for filtering top users
    Returns:
        df_scale (`pd.DataFrame`): dataframe that's filtered based on users
    """

    if not isinstance(ratings_df, pd.DataFrame):
        logger.error("The argument `processed_df` is not dataframe,"
                     "it is %stype", type(ratings_df))
        raise TypeError(f"The argument `processed_df` is not dataframe,"
                        f"it is {type(ratings_df)} type")

    if not isinstance(books_df, pd.DataFrame):
        logger.error("The argument `processed_df` is not dataframe,"
                     "it is %s type", type(books_df))
        raise TypeError(f"The argument `processed_df` is not dataframe,"
                        f"it is {type(books_df)} type")

    if not isinstance(user_df, pd.DataFrame):
        logger.error("The argument `processed_df` is not dataframe,"
                     "it is %s type", type(user_df))
        raise TypeError(f"The argument `processed_df` is not dataframe,"
                        f"it is {type(user_df)} type")

    # renaming the columns for all the three dataframes
    try:
        books_df.rename(columns=book_cols, inplace=True)

    except KeyError:
        logger.error("Please make sure that %s are valid column names",
                     book_cols.keys())
        raise
    try:
        ratings_df.rename(columns=ratings_cols, inplace=True)

    except KeyError:
        logger.error("Please make sure that %s are valid column names",
                     ratings_cols.keys())
        raise
    try:
        user_df.rename(columns=users_cols, inplace=True)

    except KeyError:
        logger.error("Please make sure that %s are valid column names",
                     users_cols.keys())
        raise
    # selecting only users who looked at more than 150 books
    top_users = ratings_df["uid"].value_counts() >= user_threshold
    top_users_index = top_users[top_users].index

    # selecting all the user ratings who looked at more than 150 books
    selected_ratings = ratings_df[ratings_df['uid'].isin(top_users_index)]
    merged_df = selected_ratings.merge(books_df, on="isbn", how="inner", sort=False)
    logger.info("Successfully finished all the steps in selecting_top_users function")

    return merged_df


def counting_ratings(top_user_df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """ Generates a dataframe that has only books based on specified rating threshold
    Args:
        top_user_df (`pd.DataFrame`): merged dataframe
        threshold (`int`): rating threshold used for filtering

    Returns:
        final_df (`pd.DataFrame`): dataframe that has the books that are filtered
        based on specific rating threshold
    """
    if not isinstance(top_user_df, pd.DataFrame):
        logger.error("The argument `processed_df` is not dataframe,"
                     "it is %s type", type(top_user_df))
        raise TypeError(f"The argument `processed_df` is not dataframe,"
                        f"it is {type(top_user_df)} type")

    try:
        # counting number ratings each book has
        rating_count = top_user_df.groupby('title')['rating'].count().reset_index()

        # merging datasets to have counts
        final_df = top_user_df.merge(rating_count, on="title")
        final_df.rename(columns={"rating_y": "rating_count",
                                 "rating_x": "rating"}, inplace=True)
        # select only books which have count of more than the given threshold
        final_df = final_df[final_df["rating_count"] >= threshold]
        logger.info("Selected all the books with ratings higher than %d", threshold)
        # dropping duplicates
        final_df.drop_duplicates(["uid", "title"], inplace=True)

    except KeyError:
        logger.error("Please make sure that `title`, `uid` and `rating` "
                     "are valid columns/column names")
        raise

    else:
        logger.info("Successfully finished all the steps in counting_ratings function")
        return final_df


def get_top_books(clean_df: pd.DataFrame, top_n: int, **kwargs) -> pd.DataFrame:
    """ Function to extract top 20 books from the cleaned data based on rating and rating count
    Args:
        clean_df(`pd.DataFrame`): clean dataframe
        top_n (`int`): top n number of books selected
        **kwargs(`dict`): aggregation arguments

    Returns:
        top_20_df(`pd.DataFrame`) top 20 books dataframe
    """
    if not isinstance(clean_df, pd.DataFrame):
        logger.error("The argument `processed_df` is not dataframe, "
                     "it is %s type", type(clean_df))
        raise TypeError(f"The argument `processed_df` is not dataframe, "
                        f"it is {type(clean_df)} type")

    try:
        # grouping by title to find best books by rating and rating count
        grouped_df = clean_df.groupby(by=["title"]).agg(kwargs).reset_index()
        grouped_df.sort_values(by=["rating", "rating_count"], ascending=False, inplace=True)

        # selecting only top 20 unique values
        top_n_list = grouped_df.drop_duplicates(subset=["title"])\
                         .reset_index(drop=True).iloc[:top_n]

    except KeyError:
        logger.error("Please make sure that title, rating and rating count "
                     "are valid columns/column names")
        raise

    else:
        selected_df = clean_df[["author", "year", "image_l", "title"]]
        top_n_df = pd.merge(top_n_list, selected_df, on="title", how="left") \
            .drop_duplicates(subset=["title"]).reset_index(drop=True)

        # selecting only title, author, year and image url columns
        top_n_df['author'] = top_n_df['author'].str.title()
        top_n_df = top_n_df[["title", "author", "year", "image_l"]]

        logger.info("Successfully extracted top 20 books from the cleaned dataset")
        return top_n_df


def transform_to_pivot(clean_df: pd.DataFrame) -> pd.DataFrame:
    """ Function to transform cleaned dataframe to matrix form dataframe
    Args:
        clean_df(`pd.DataFrame`): clean dataframe that's used to convert the data into matrix

    Returns:
         book_matrix(`pd.DataFrame`)
    """
    if not isinstance(clean_df, pd.DataFrame):
        logger.error("The argument `processed_df` is not dataframe,"
                     "it is %s type", type(clean_df))
        raise TypeError(f"The argument `processed_df` is not dataframe,"
                        f"it is {type(clean_df)} type")

    try:
        book_matrix = clean_df.pivot_table(columns="uid",
                                           index="title",
                                           values="rating")
        book_matrix = book_matrix.fillna(0).reset_index()
    except KeyError:
        logger.error("Please double check that columns `uid`, "
                     "`rating` and `title` exist")

    else:
        logger.info("Successfully transformed cleaned data to sparse matrix form")
        return book_matrix


def save_df(processed_df: pd.DataFrame,
            output_dir: str = None,
            file_name: str = None) -> None:
    """ Used for saving the dataframe to csv file and to s3 directory
    Args:
    processed_df(pd.DataFrame): already processed dataframe
    output_dir (`str`): directory where the df will be saved (s3 or local)
    file_name (`str`): file name as specified in yaml file

    Returns:
         None
    """
    if not isinstance(processed_df, pd.DataFrame):
        logger.error("The argument `processed_df` is not dataframe,"
                     "it is %s type", type(processed_df))
        raise TypeError(f"The argument `processed_df` is not dataframe,"
                        f"it is {type(processed_df)} type")

    if file_name is None:

        processed_df.to_csv(output_dir, index=False)
        logger.info("The processed data is saved to %s", output_dir)

    else:
        full_path = f"{output_dir}{file_name}"
        processed_df.to_csv(full_path, index=False)
        logger.info("The processed data is saved to %s%s", output_dir, file_name)


def save_txt(a_text: Union[str, list, float],
             file_dir: str,
             file_name: str) -> None:
    """ Save the text to .txt, .csv or any other format
    Args:
        a_text(Union[`str`, `list`, `float`]): text that's being saved
        file_dir(`str`): Directory of the file that's being saved
        file_name(`str`): The name that's given to the file being saved

    Returns:
        None
    """
    try:
        with open(f"{file_dir}{file_name}", "w") as text_file:
            text_file.write(f"{a_text}")
            logger.info("Successfully saved %s", file_name)
    except OSError as error:
        logger.error("Please make sure that file directory and file"
                     "name you specified are correct: %s", error)
