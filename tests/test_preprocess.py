import pandas as pd
import src.preprocess
import pytest
import numpy as np

df_values = [[264317, 'B000234N76', 0, 'Falling Angels', 'Tracy Chevalier', 2001, 'E P Dutton',
              'http://images.amazon.com/images/P/B000234N76.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B000234N76.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B000234N76.01.LZZZZZZZ.jpg', 73],
             [271705, 'B0001PIOX4', 0, 'Fahrenheit 451', 'Ray Bradbury', 1993, 'Simon &amp; Schuster',
              'http://images.amazon.com/images/P/B0001PIOX4.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B0001PIOX4.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B0001PIOX4.01.LZZZZZZZ.jpg', 139],
             [244277, 'B0001FZGPI', 0, "The Bonesetter's Daughter", 'Amy Tan', '2001', 'Putnam Pub Group',
              'http://images.amazon.com/images/P/B0001FZGPI.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B0001FZGPI.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B0001FZGPI.01.LZZZZZZZ.jpg', 154],
             [204359, 'B0000T6KIM', 4, 'Faking It', 'Jennifer Crusie', 2002, "St. Martin's Press",
              'http://images.amazon.com/images/P/B0000T6KIM.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B0000T6KIM.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B0000T6KIM.01.LZZZZZZZ.jpg', 107],
             [69697, 'B0000T6KHI', 0, 'Three Fates', 'Nora Roberts', 2002, 'Putnam Pub Group',
              'http://images.amazon.com/images/P/B0000T6KHI.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B0000T6KHI.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B0000T6KHI.01.LZZZZZZZ.jpg', 88],
             [112001, 'B0000T6KHI', 10, 'Three Fates', 'Nora Roberts', 2002, 'Putnam Pub Group',
              'http://images.amazon.com/images/P/B0000T6KHI.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B0000T6KHI.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B0000T6KHI.01.LZZZZZZZ.jpg', 88],
             [73681, 'B00009NDAN', 9, 'Winter Solstice', 'Rosamunde Pilcher', 2000,
              "St. Martin's Press", 'http://images.amazon.com/images/P/B00009NDAN.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00009NDAN.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00009NDAN.01.LZZZZZZZ.jpg', 79],
             [274301, 'B00009NDAN', 10, 'Winter Solstice', 'Rosamunde Pilcher', 2000,
              "St. Martin's Press", 'http://images.amazon.com/images/P/B00009NDAN.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00009NDAN.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00009NDAN.01.LZZZZZZZ.jpg', 79],
             [208410, 'B00009EF82', 0, 'Hannibal', 'Thomas Harris', 1999, 'Delacorte Press',
              'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg', 206],
             [91203, 'B00009EF82', 0, 'Hannibal', 'Thomas Harris', 1999, 'Delacorte Press',
              'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg', 206],
             [168064, 'B00009EF82', 0, 'Hannibal', 'Thomas Harris', 1999, 'Delacorte Press',
              'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg', 206],
             [120565, 'B00009EF82', 9, 'Hannibal', 'Thomas Harris', 1999, 'Delacorte Press',
              'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg', 206],
             [80538, 'B00009EF82', 10, 'Hannibal', 'Thomas Harris', 1999, 'Delacorte Press',
              'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg', 206],
             [164533, 'B00008WFXL', 0, 'The Da Vinci Code', 'Dan Brown', '0', 'Doubleday',
              'http://images.amazon.com/images/P/B00008WFXL.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00008WFXL.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00008WFXL.01.LZZZZZZZ.jpg', 272],
             [87141, 'B00007CWQC', 0, 'The Villa', 'Nora Roberts', 2001, 'Putnam Pub Group',
              'http://images.amazon.com/images/P/B00007CWQC.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00007CWQC.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00007CWQC.01.LZZZZZZZ.jpg', 106],
             [13582, 'B00005NCS7', 0, 'Moonlight Becomes You', 'Mary Higgins Clark', 0, 'Simon &amp; Schuster',
              'http://images.amazon.com/images/P/B00005NCS7.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00005NCS7.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00005NCS7.01.LZZZZZZZ.jpg', 73],
             [78973, 'B00001U0CP', 8, 'Unnatural Exposure', 'Patricia Cornwell', 1997, 'Putnam Pub Group',
              'http://images.amazon.com/images/P/B00001U0CP.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00001U0CP.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00001U0CP.01.LZZZZZZZ.jpg', 99],
             [32440, 'B00001U0CP', 0, 'Unnatural Exposure', 'Patricia Cornwell', 1997, 'Putnam Pub Group',
              'http://images.amazon.com/images/P/B00001U0CP.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/B00001U0CP.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/B00001U0CP.01.LZZZZZZZ.jpg', 99], [
                 234828, 'B00001IVC7', 0, 'Desperation', 'Stephen King', 1996, 'Viking Press',
                 'http://images.amazon.com/images/P/B00001IVC7.01.THUMBZZZ.jpg',
                 'http://images.amazon.com/images/P/B00001IVC7.01.MZZZZZZZ.jpg',
                 'http://images.amazon.com/images/P/B00001IVC7.01.LZZZZZZZ.jpg', 67],
             [94242, '9505112076', 0, 'Matilda', 'Roald Dahl', 1996, 'Aguilar',
              'http://images.amazon.com/images/P/9505112076.01.THUMBZZZ.jpg',
              'http://images.amazon.com/images/P/9505112076.01.MZZZZZZZ.jpg',
              'http://images.amazon.com/images/P/9505112076.01.LZZZZZZZ.jpg', 66]]

df_index = [101912, 99401, 63214, 67536, 199463, 199464, 152812, 152813, 93616, 93613, 93615,
            93614, 93612, 5376, 199349, 56192, 116777, 116776, 160982, 29419]

index_df = [0, 1, 2]

df_columns = ['uid', 'isbn', 'rating', 'title', 'author', 'year', 'publisher',
              'image_s', 'image_m', 'image_l', 'rating_count']

ratings_input = pd.DataFrame([[276725, '034545104X', 10],
                              [276725, '0155061224', 5],
                              [276727, '0446520802', 4]],
                             index=index_df,
                             columns=['User-ID', 'ISBN', 'Book-Rating'])

books_input = pd.DataFrame([['034545104X', 'Flesh Tones: A Novel', 'M. J. Rose', 2002,
                             'Ballantine Books',
                             'http://images.amazon.com/images/P/034545104X.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/034545104X.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/034545104X.01.LZZZZZZZ.jpg'],
                            ['0446520802', 'The Notebook', 'Nicholas Sparks', 1996,
                             'Warner Books',
                             'http://images.amazon.com/images/P/0446520802.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/0446520802.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/0446520802.01.LZZZZZZZ.jpg'],
                            ['0155061224', 'Rites of Passage', 'Judith Rae', '2001', 'Heinle',
                             'http://images.amazon.com/images/P/0155061224.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/0155061224.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/0155061224.01.LZZZZZZZ.jpg']],
                           index=index_df,
                           columns=['ISBN', 'Book-Title', 'Book-Author',
                                    'Year-Of-Publication', 'Publisher',
                                    'Image-URL-S', 'Image-URL-M', 'Image-URL-L'])

users_input = pd.DataFrame([[1, 'nyc, new york, usa', np.nan],
                            [2, 'stockton, california, usa', 18.0],
                            [3, 'moscow, yukon territory, russia', np.nan]],
                           index=index_df,
                           columns=['User-ID', 'Location', 'Age'])

book_col = {'ISBN': 'isbn',
            'Book-Title': 'title',
            'Book-Author': 'author',
            'Year-Of-Publication': 'year',
            'Publisher': 'publisher',
            'Image-URL-S': 'image_s',
            'Image-URL-M': 'image_m',
            'Image-URL-L': 'image_l'}

rating_col = {'User-ID': 'uid',
              'ISBN': 'isbn',
              'Book-Rating': 'rating'}

user_col = {'User-ID': 'uid',
            'Location': 'location',
            'Age': 'age'}


def test_get_top_books_happy():
    df_input = pd.DataFrame(data=df_values, columns=df_columns, index=df_index)

    df_true = pd.DataFrame([['Winter Solstice', 'Rosamunde Pilcher', 2000,
                             'http://images.amazon.com/images/P/B00009NDAN.01.LZZZZZZZ.jpg'],
                            ['Three Fates', 'Nora Roberts', 2002,
                             'http://images.amazon.com/images/P/B0000T6KHI.01.LZZZZZZZ.jpg'],
                            ['Faking It', 'Jennifer Crusie', 2002,
                             'http://images.amazon.com/images/P/B0000T6KIM.01.LZZZZZZZ.jpg'],
                            ['Unnatural Exposure', 'Patricia Cornwell', 1997,
                             'http://images.amazon.com/images/P/B00001U0CP.01.LZZZZZZZ.jpg'],
                            ['Hannibal', 'Thomas Harris', 1999,
                             'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg']],
                           index=[0, 1, 2, 3, 4],
                           columns=['title', 'author', 'year', 'image_l'])

    df_test = src.preprocess.get_top_books(df_input, 5, **{"rating": "mean", "rating_count": "mean"})
    df_test['year'] = df_test['year'].astype('int64')

    assert df_test.equals(df_true)


def test_get_top_books_unhappy():
    df_columns = ['uid', 'isbn', 'rating', 'blabla', 'author', 'year', 'publisher',
                  'image_s', 'image_m', 'image_l', 'rating_count']

    df_input = pd.DataFrame(data=df_values, columns=df_columns, index=df_index)

    with pytest.raises(KeyError):
        src.preprocess.get_top_books(df_input, 5, **{"rating": "mean", "rating_count": "mean"})


def test_rename_and_select_happy():
    # ratings_input_local = pd.DataFrame([[276725, '034545104X', 10],
    #                                     [276725, '0155061224', 5],
    #                                     [276727, '0446520802', 4]],
    #                                    index=index_df,
    #                                    columns=['User-NEWID', 'ISBN', 'Book-Rating'])

    df_test = src.preprocess.rename_and_select(ratings_input, books_input, users_input,
                                               rating_col, book_col, user_col, 2)

    df_true = pd.DataFrame([[276725, '034545104X', 10, 'Flesh Tones: A Novel', 'M. J. Rose',
                             2002, 'Ballantine Books',
                             'http://images.amazon.com/images/P/034545104X.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/034545104X.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/034545104X.01.LZZZZZZZ.jpg'],
                            [276725, '0155061224', 5, 'Rites of Passage', 'Judith Rae',
                             '2001', 'Heinle',
                             'http://images.amazon.com/images/P/0155061224.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/0155061224.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/0155061224.01.LZZZZZZZ.jpg']],
                           index=[0, 1],
                           columns=['uid', 'isbn', 'rating', 'title', 'author', 'year', 'publisher',
                                    'image_s', 'image_m', 'image_l'])

    assert df_test.equals(df_true)


def test_rename_and_select_unhappy():
    with pytest.raises(KeyError):
        ratings_input_local = pd.DataFrame([[276725, '034545104X', 10],
                                            [276725, '0155061224', 5],
                                            [276727, '0446520802', 4]],
                                           index=index_df,
                                           columns=['User-NEWID', 'ISBN', 'Book-Rating'])

        src.preprocess.rename_and_select(ratings_input_local, books_input, users_input,
                                         rating_col, book_col, user_col, 2)


def test_counting_ratings_happy():

    threshold = 3
    df_input = pd.DataFrame(data=df_values, columns=df_columns, index=df_index).drop('rating_count', axis=1)
    df_input['year'] = df_input['year'].astype('int64')

    df_true = pd.DataFrame([[208410, 'B00009EF82', 0, 'Hannibal', 'Thomas Harris', 1999,
                             'Delacorte Press',
                             'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg',
                             5],
                            [91203, 'B00009EF82', 0, 'Hannibal', 'Thomas Harris', 1999,
                             'Delacorte Press',
                             'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg',
                             5],
                            [168064, 'B00009EF82', 0, 'Hannibal', 'Thomas Harris', 1999,
                             'Delacorte Press',
                             'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg',
                             5],
                            [120565, 'B00009EF82', 9, 'Hannibal', 'Thomas Harris', 1999,
                             'Delacorte Press',
                             'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg',
                             5],
                            [80538, 'B00009EF82', 10, 'Hannibal', 'Thomas Harris', 1999,
                             'Delacorte Press',
                             'http://images.amazon.com/images/P/B00009EF82.01.THUMBZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.MZZZZZZZ.jpg',
                             'http://images.amazon.com/images/P/B00009EF82.01.LZZZZZZZ.jpg',
                             5]],
                           index=[8, 9, 10, 11, 12],
                           columns=['uid', 'isbn', 'rating', 'title', 'author', 'year', 'publisher',
                                    'image_s', 'image_m', 'image_l', 'rating_count'])

    df_output = src.preprocess.counting_ratings(df_input, 3)
    assert df_true.equals(df_output)


def test_counting_ratings_unhappy():
    with pytest.raises(KeyError):
        col_names = ['uid', 'isbn', 'rating', 'blabla', 'author', 'year', 'publisher',
                     'image_s', 'image_m', 'image_l', 'rating_count']

        df_input = pd.DataFrame(data=df_values, columns=col_names, index=df_index).drop('rating_count', axis=1)
        src.preprocess.counting_ratings(df_input, 3)
