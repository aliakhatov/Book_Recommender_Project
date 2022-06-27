import logging.config
from typing import List, Dict, Tuple

import pandas as pd
import sklearn.neighbors
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import numpy as np

logger = logging.getLogger(__name__)


def train_model(book_pivot: pd.DataFrame, **kwargs) -> sklearn.base.BaseEstimator:
    """ Helper function for fitting the KNN model and extracting results
    Args:
        book_pivot(`pd.DataFrame`): dataframe that has the sparse matrix with row indexes as titles
                           and columns as user's rating of  the book
        ** kwargs (:obj:`dict`): dictionary of parameters for Nearest neighbors model

    Returns:
        model(`sklearn.base.BaseEstimator`)

    """
    try:
        # fitting the knn based on given title
        book_pivot = book_pivot.set_index("title")
        book_sparse = csr_matrix(book_pivot)
        model = NearestNeighbors(**kwargs)
        model.fit(book_sparse)

    except TypeError:
        logger.error("Please double check the **kwargs argument."
                     "There could be unexpected arguments passed")
        raise
    else:
        logger.info("Model fitting step is finished")
        return model


def recommend_books(book_matrix: pd.DataFrame, titles_list: List,
                    knn_model: sklearn.base.BaseEstimator) -> List[str]:
    """ Main modeling function that's used to generate recommendations based on given book titles
    Args:
        book_matrix(`pd.DataFrame`): dataframe that has the sparse matrix with row indexes as titles
        and columns as user's rating of  the book
        titles_list(:obj:`list` of `str`): list of book titles that are returned from the user input
        knn_model(`sklearn.base.BaseEstimator` obj): model that's used to run the KNN
    Returns:
        unique_books(:obj:`list` of `str`)
    """
    # checking whether the passed object is a dataframe
    if not isinstance(book_matrix, pd.DataFrame):
        logger.error("The argument `book_matrix` is not dataframe, it is %s type",
                     type(book_matrix))
        raise TypeError(f"The argument `book_matrix` is not dataframe,"
                        f"it is {type(book_matrix)} type")

    if book_matrix.empty:
        logger.warning("`book_matrix` is empty. Please check whether it is supposed to be empty")

    # setting title column as index for the dataframe
    try:
        book_matrix = book_matrix.set_index("title")

    except KeyError:
        logger.error("`title` is not in the columns. Please make sure it is the right column name")

    # running the helper function for each book title
    recommended_list = [get_suggestions_helper(book_matrix, a_title, knn_model) \
                        for a_title in titles_list]
    recommended_list = list(np.concatenate(recommended_list))

    # selecting only unique books
    unique_books = list(set(recommended_list))

    # yields the elements in `unique_books` that are NOT in `title_list`
    unique_books = list(np.setdiff1d(unique_books, titles_list))
    logger.info("Book recommendations are successfully generated")

    return unique_books


def extract_info(title_list: List[str], clean_df: pd.DataFrame) -> List[Dict]:
    """ Used for extracting images, author and year of the recommended books
    Args:
        title_list(:obj:`list` of `str`): title list
        clean_df(`pd.DataFrame`): cleaned dataframe
    Returns:
        rec_books(:obj:`list` of `dict`)
    """
    rec_books = []

    # checking whether dataframe is empty
    if clean_df.empty:
        logger.warning("`clean_df` is empty. Please check whether "
                       "it is supposed to be empty")

    # looping through all the book titles to generate other information
    # such as image url, author and year
    for book_title in title_list:
        try:
            book_image = clean_df[clean_df['title'] == book_title][['title',
                                                                    'image_l',
                                                                    'author',
                                                                    'year']].reset_index(drop=True)
        except KeyError:
            logger.error("Please make sure that `title`, `image_l`, `author` and `year` exist")
            raise
        rec_books.append(book_image.iloc[0].to_dict())

    logger.info("Author, image_l and year variables for recommended "
                "list are successfully retrieved")
    return rec_books


def split_test_train(clean_df: pd.DataFrame, test_ratio: int,
                     rating_threshold: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """ Splitting the data to test and train sets
    Args:
        clean_df(`pd.DataFrame`): cleaned dataframe
        test_ratio(`int`): test split ration of the full dataset
        rating_threshold(`int`): threshold for selecting users for test and train split
        seed(`int`): random seed used for train test split reproducibility

    Returns:
            train_df (`pd.DataFrame`): Training predictor variables
            test_df (`pd.DataFrame`): Ground truth in training set
    """
    # setting random seed number
    np.random.seed(seed)

    # checking whether the passed object is a dataframe
    if not isinstance(clean_df, pd.DataFrame):
        logger.error("The argument `clean_df` is not dataframe, it is %s type",
                     type(clean_df))
        raise TypeError(f"The argument `processed_df` is not dataframe, "
                        f"it is {type(clean_df)} type")
    # selecting top users with the certain rating count threshold
    top_users_rated = clean_df['uid'].value_counts() >= rating_threshold
    top_users_rated_index = top_users_rated[top_users_rated].index
    selected_ratings_rated = clean_df[clean_df['uid'].isin(top_users_rated_index)]
    unique_id_num = len(selected_ratings_rated['uid'].unique())

    try:
        # setting test size of the test train split
        test_size = int(np.round(unique_id_num * test_ratio))
        unique_id = selected_ratings_rated['uid'].unique()

    except TypeError:
        logger.error("Please make sure that `test ratio` is numeric variable")
        raise
    # selecting random user ids from the all user ids
    random_ids = np.random.choice(unique_id, test_size, replace=False)

    # creating test and train dataframes
    train_df = clean_df[~clean_df['uid'].isin(random_ids)]
    test_df = clean_df[clean_df['uid'].isin(random_ids)]

    logger.info("Successfully created train and test dataframes")
    return train_df, test_df


def get_suggestions_helper(book_pivot: pd.DataFrame,
                           title: str, model: sklearn.base.BaseEstimator) -> List:
    """ Helper function for generating suggestions
    Args:
        book_pivot(pd.DataFrame): sparse matrix dataframe
        title(`str`): a single book title
        model(`sklearn.base.BaseEstimator` obj): model that's used to run the KNN
    Returns:
        rec_list(:obj:`list` of `str`)
    """
    if not isinstance(book_pivot, pd.DataFrame):
        logger.error("The argument `book_pivot` is not dataframe, it is %s type",
                     type(book_pivot))
        raise TypeError(f"The argument `book_pivot` is not dataframe,"
                        f"it is {type(book_pivot)} type")

    # here title does not exist in test data, so need to change it
    suggestions = model.kneighbors(book_pivot[book_pivot.index == title] \
                                   .values.reshape(1, -1),
                                   return_distance=False)
    logger.info("Suggestions are retrieved from knn model")

    # knn returns all the nearest neighbors and the point itself
    # so, selecting all the suggestions except the point itself
    suggestions = suggestions[0][1:]
    rec_list = book_pivot.index[suggestions]

    return rec_list.tolist()


def evaluate_model(train_matrix: pd.DataFrame, test_df: pd.DataFrame,
                   knn_model: sklearn.base.BaseEstimator,
                   num_of_evaluations: int) -> Tuple[float, float]:
    """ Evaluates the model using Mean Average Precision (MAP)
    Args:
        train_matrix(`pd.DataFrame`): train matrix set to run the model on
        test_df(`pd.DataFrame`): test set to calculate the MAP
        knn_model(`sklearn.base.BaseEstimator` obj): model that's used to run the KNN
        num_of_evaluations(`int`): number of titles selected for each user at random
    Returns:
        final_precision(`str`): precision metric for the model
        final_recall(`str`): recal metric for the model

    """
    train_matrix = train_matrix.set_index("title")
    precision_list = []
    recall_list = []
    test_id_list = test_df['uid'].unique()

    for an_id in test_id_list:
        # selecting unique titles
        unique_titles = test_df[test_df['uid'] == an_id]['title']
        unique_existing_titles = np.intersect1d(unique_titles, train_matrix.index.unique())
        random_title = np.random.choice(unique_existing_titles, num_of_evaluations, replace=False)

        # calling the model to extract recommendations
        recommended_list = [get_suggestions_helper(train_matrix, a_title, knn_model) \
                            for a_title in random_title]
        recommended_list = np.concatenate(recommended_list)

        # finding number of recommendations
        common_num = len(np.intersect1d(recommended_list, unique_titles))
        try:
            recall = common_num / len(unique_titles)
            precision = common_num / len(recommended_list)
            precision_list.append(precision)
            recall_list.append(recall)

        except ZeroDivisionError:
            logger.error("Please check that the list that are being passed are not empty")
    # calculating average precision and recall for the all the ids
    final_precision = float(np.mean(precision_list))
    final_recall = float(np.mean(recall_list))

    logger.info("Successfully calculated precision and recall!")
    return final_precision, final_recall
