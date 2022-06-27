import pandas as pd
import src.model
import pytest
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

values_in = [[204359, '0439136350', 6,
              'Harry Potter and the Prisoner of Azkaban (Book 3)',
              'J. K. Rowling', 1999, 'Scholastic',
              'http://images.amazon.com/images/P/0439136350.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/0439136350.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/0439136350.01.LZZZZZZZ.jpg',
              159],
             [187517, '0375504397', 0, 'Black House', 'Stephen King', 2001,
              'Random House Trade',
              'http://images.amazon.com/images/P/0375504397.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/0375504397.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/0375504397.01.LZZZZZZZ.jpg',
              82],
             [238120, '0380974509', 0, 'The Promise', 'Donna Boyd', 1999,
              'Avon Books',
              'http://images.amazon.com/images/P/0380974509.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/0380974509.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/0380974509.01.LZZZZZZZ.jpg',
              66],
             [225087, '0446527165', 10, 'Wish You Well', 'David Baldacci',
              2000, 'Warner Books',
              'http://images.amazon.com/images/P/0446527165.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/0446527165.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/0446527165.01.LZZZZZZZ.jpg',
              97],
             [264031, '0446675059', 0, 'The Honk and Holler Opening Soon',
              'Billie Letts', 1999, 'Warner Books',
              'http://images.amazon.com/images/P/0446675059.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/0446675059.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/0446675059.01.LZZZZZZZ.jpg',
              82]]

book_list = [[11676,
              'Politically Correct Bedtime Stories: Modern Tales for Our Life and Times',
              'James Finn Garner', 6],
             [16795, 'The Poisonwood Bible: A Novel', 'Barbara Kingsolver', 4],
             [35859, 'Bel Canto: A Novel', 'Ann Patchett', 9],
             [76352, 'Bel Canto: A Novel', 'Ann Patchett', 6],
             [11676, 'One for the Money (Stephanie Plum Novels (Paperback))',
              'Janet Evanovich', 8],
             [16795, 'One for the Money (Stephanie Plum Novels (Paperback))',
              'Janet Evanovich', 9],
             [35859, 'One for the Money (Stephanie Plum Novels (Paperback))',
              'Janet Evanovich', 10],
             [153662, 'One for the Money (Stephanie Plum Novels (Paperback))',
              'Janet Evanovich', 9],
             [11676, 'The Secret Garden', 'Frances Hodgson Burnett', 10],
             [16795, 'The Secret Garden', 'Frances Hodgson Burnett', 8],
             [153662, 'The Secret Garden', 'Frances Hodgson Burnett', 10],
             [11676, 'The Tao of Pooh', 'Benjamin Hoff', 8],
             [11676, 'Girl in Hyacinth Blue', 'Susan Vreeland', 7],
             [153662, 'Girl in Hyacinth Blue', 'Susan Vreeland', 9],
             [11676, 'Chocolat', 'Joanne Harris', 10],
             [16795, 'The Secret Life of Bees', 'Sue Monk Kidd', 10],
             [35859, 'The Secret Life of Bees', 'Sue Monk Kidd', 9],
             [11676,
              'Three To Get Deadly : A Stephanie Plum Novel (A Stephanie Plum Novel)',
              'Janet Evanovich', 8],
             [35859,
              'Three To Get Deadly : A Stephanie Plum Novel (A Stephanie Plum Novel)',
              'Janet Evanovich', 9],
             [11676, 'Lucky : A Memoir', 'Alice Sebold', 10]]


def test_recommend_books_happy():
    df_in = pd.DataFrame(book_list, columns=['uid', 'title', 'author', 'rating'])

    # creating a pivot table
    matrix = df_in.pivot_table(columns='uid', index='title', values='rating')
    matrix = matrix.fillna(0)
    book_sparse = csr_matrix(matrix)
    # fitting the model
    model_b = NearestNeighbors(algorithm='brute', n_neighbors=3)
    model_b.fit(book_sparse)

    picked_title = ['The Secret Garden']
    matrix.reset_index(inplace=True)
    result = src.model.recommend_books(matrix, picked_title, model_b)

    suggestion_true = ['Girl in Hyacinth Blue',
                       'One for the Money (Stephanie Plum Novels (Paperback))']
    assert suggestion_true == result


def test_recommend_books_unhappy():
    with pytest.raises(TypeError):
        not_df = "not a dataframe"
        a_model = NearestNeighbors()
        src.model.get_suggestions_helper(not_df, "some_title", a_model)


def test_split_test_train_happy():
    clean_df = pd.DataFrame(book_list, columns=['uid', 'title', 'author', 'rating'])
    test_ratio = 0.2
    threshold = 2
    rd_seed = 12345
    train_df, test_df = src.model.split_test_train(clean_df, test_ratio, threshold, rd_seed)

    train_true = [[16795, 'The Poisonwood Bible: A Novel', 'Barbara Kingsolver', 4],
                  [35859, 'Bel Canto: A Novel', 'Ann Patchett', 9],
                  [76352, 'Bel Canto: A Novel', 'Ann Patchett', 6],
                  [16795, 'One for the Money (Stephanie Plum Novels (Paperback))',
                   'Janet Evanovich', 9],
                  [35859, 'One for the Money (Stephanie Plum Novels (Paperback))',
                   'Janet Evanovich', 10],
                  [153662, 'One for the Money (Stephanie Plum Novels (Paperback))',
                   'Janet Evanovich', 9],
                  [16795, 'The Secret Garden', 'Frances Hodgson Burnett', 8],
                  [153662, 'The Secret Garden', 'Frances Hodgson Burnett', 10],
                  [153662, 'Girl in Hyacinth Blue', 'Susan Vreeland', 9],
                  [16795, 'The Secret Life of Bees', 'Sue Monk Kidd', 10],
                  [35859, 'The Secret Life of Bees', 'Sue Monk Kidd', 9],
                  [35859,
                   'Three To Get Deadly : A Stephanie Plum Novel (A Stephanie Plum Novel)',
                   'Janet Evanovich', 9]]

    train_true = pd.DataFrame(train_true, columns=['uid', 'title', 'author', 'rating'])
    test_true = [[11676,
                  'Politically Correct Bedtime Stories: Modern Tales for Our Life and Times',
                  'James Finn Garner', 6],
                 [11676, 'One for the Money (Stephanie Plum Novels (Paperback))',
                  'Janet Evanovich', 8],
                 [11676, 'The Secret Garden', 'Frances Hodgson Burnett', 10],
                 [11676, 'The Tao of Pooh', 'Benjamin Hoff', 8],
                 [11676, 'Girl in Hyacinth Blue', 'Susan Vreeland', 7],
                 [11676, 'Chocolat', 'Joanne Harris', 10],
                 [11676,
                  'Three To Get Deadly : A Stephanie Plum Novel (A Stephanie Plum Novel)',
                  'Janet Evanovich', 8],
                 [11676, 'Lucky : A Memoir', 'Alice Sebold', 10]]

    test_true = pd.DataFrame(test_true, columns=['uid', 'title', 'author', 'rating'])
    train_df.reset_index(inplace=True, drop=True)
    test_df.reset_index(inplace=True, drop=True)

    assert train_true.equals(train_df) and test_true.equals(test_df)


def test_split_test_train_unhappy():

    with pytest.raises(TypeError):
        clean_df = pd.DataFrame(book_list, columns=['uid', 'title', 'author', 'rating'])
        test_ratio = "0.2"
        threshold = 2
        rd_seed = 12345
        src.model.split_test_train(clean_df, test_ratio, threshold, rd_seed)


def test_extract_info_happy():

    titles_in = ['The Honk and Holler Opening Soon', 'Wish You Well']
    df_input = pd.DataFrame(values_in, index=[7407, 159656, 7655, 154969, 93967],
                            columns=['uid', 'isbn', 'rating', 'title', 'author', 'year', 'publisher',
                                     'image_s', 'image_m', 'image_l', 'rating_count'])

    output_true = [{'title': 'The Honk and Holler Opening Soon',
                    'image_l': 'http://images.amazon.com/images/P/0446675059.01.LZZZZZZZ.jpg',
                    'author': 'Billie Letts',
                    'year': 1999},
                   {'title': 'Wish You Well',
                    'image_l': 'http://images.amazon.com/images/P/0446527165.01.LZZZZZZZ.jpg',
                    'author': 'David Baldacci',
                    'year': 2000}]

    df_output = src.model.extract_info(titles_in, df_input)
    assert output_true == df_output


def test_extract_info_unhappy():

    with pytest.raises(KeyError):
        titles_in = ['The Honk and Holler Opening Soon', 'Wish You Well']
        df_input = pd.DataFrame(values_in, index=[7407, 159656, 7655, 154969, 93967],
                                columns=['uid', 'isbn', 'rating', 'blalba', 'author', 'year', 'publisher',
                                         'image_s', 'image_m', 'blabla2', 'rating_count'])
        src.model.extract_info(titles_in, df_input)


def test_train_model_happy():

    df_in = pd.DataFrame(book_list, columns=['uid', 'title', 'author', 'rating'])
    # creating a pivot table
    matrix = df_in.pivot_table(columns='uid', index='title', values='rating')
    matrix = matrix.fillna(0)
    book_sparse = csr_matrix(matrix)
    # fitting the model
    model_true = NearestNeighbors(algorithm='brute', n_neighbors=3, metric="cosine")
    model_true.fit(book_sparse)

    params = {"algorithm": 'brute', "metric": 'cosine', "n_neighbors": 3}
    matrix.reset_index(inplace=True)
    model_output = src.model.train_model(matrix, **params)
    assert model_output.algorithm == model_output.algorithm


def test_train_model_unhappy():
    with pytest.raises(TypeError):
        df_in = pd.DataFrame(book_list, columns=['uid', 'title', 'author', 'rating'])
        # creating a pivot table
        matrix = df_in.pivot_table(columns='uid', index='title', values='rating')
        matrix = matrix.fillna(0)
        params = {"algorithm": 'brute', "metric": 'cosine', "random_state": 3}
        matrix.reset_index(inplace=True)
        src.model.train_model(matrix, **params)
