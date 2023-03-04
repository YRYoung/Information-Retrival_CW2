#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
Using the vocabulary of terms identified in Task 1 (you will need to choose between removing or keeping stop words)
Build an inverted index for the collection so that you can retrieve passages in an efficient way.
"""
import os.path
import pickle

import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from tqdm.auto import tqdm

from cw1.task1 import preprocessing

verbose = __name__ == '__main__'


def read_passages_csv(data_location='candidate-passages-top1000.tsv'):
    df = pd.read_csv(data_location,
                     sep='\t', header=None, usecols=[1, 3],
                     names=['pid', 'content']).drop_duplicates()
    return df.reset_index(drop=True)


def read_all_csv(data_location='candidate-passages-top1000.tsv'):
    df = pd.read_csv(data_location,
                     sep='\t', header=None,
                     names=['qid', 'pid', 'query', 'passage']).drop_duplicates()
    return df.reset_index(drop=True)


def generate_indexes(dataframe, tokens, verbose=verbose):
    """
    Generate a matrix
    with size of (vocabulary size(how many tokens), passages size(how many passages))
    matrix[i,j] is the number of times the token vocab_dict[i] appears in the jth passage.
    """

    vocab_size = len(tokens)
    vocab_dict = dict(zip(tokens[:, 0], range(vocab_size)))
    passages_size = dataframe.shape[0]
    inverted_indexes = lil_matrix((vocab_size, passages_size))
    error_list = set()
    pbar = tqdm(dataframe.iterrows(), total=passages_size, desc='Invert indexing',
                unit='doc') if verbose else dataframe.iterrows()
    for index, passage in pbar:
        word_counter = preprocessing(passage.content)

        for word, count in word_counter:
            try:
                inverted_indexes[vocab_dict[word], index] = count
            except KeyError:
                error_list.add(word)
    if verbose and len(error_list) != 0:
        print(f'Words not in vocab: {error_list}')
    return csr_matrix(inverted_indexes)

# passages_dataframe = read_passages_csv(data_location=f'{data_path}/dataset/candidate_passages_top1000.tsv')
# file_name = f'{data_path}/temp/passages_indexes.pkl'
# passages_indexes = generate_indexes(passages_dataframe)

# if os.path.exists(file_name):
#     with open(file_name, 'rb') as file:
#         passages_indexes = pickle.load(file)
# else:
#     passages_indexes = generate_indexes(passages_dataframe)
#     with open(file_name, 'wb') as file:
#         pickle.dump(passages_indexes, file)
#
# if __name__ == '__main__':
#     print('indexing complete')
