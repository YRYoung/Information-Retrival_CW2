#!/usr/bin/env python
# -*- coding: UTF-8 -*-


import warnings

from tqdm import trange

warnings.filterwarnings(action='ignore', category=DeprecationWarning)

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
from scipy.sparse.sparsetools import csr_scale_rows, csr_scale_columns

from task2 import generate_indexes, read_all_csv, passages_indexes, passages_dataframe

__all__ = ['passages_indexes', 'queries_indexes',
           'passages_dataframe', 'queries_dataframe', 'candidates_passages_dataframe',
           'Help', 'select_first100']


class Help:

    @staticmethod
    def scale_csr(mat: csr_matrix, scaler):
        result = mat.tocsr(copy=True)
        m, n = mat.shape
        assert m != n, f'matrix is a square with shape {(m, n)}'

        scaler = np.array(scaler).reshape(-1)
        s_size = scaler.shape[0]
        assert s_size in [m, n], f'scaler has size {s_size}'

        func = csr_scale_rows if s_size == m else csr_scale_columns
        func(result.shape[0],
             result.shape[1],
             result.indptr,
             result.indices,
             result.data, scaler)

        return result

    @staticmethod
    def sparse_add_vec(mat: csr_matrix, vec):
        vec = np.array(vec).reshape(-1)
        assert mat.shape[1] == vec.shape[0]
        mat = mat.tocsc()
        mat.data = mat.data * np.repeat(vec, mat.indptr[1:] - mat.indptr[:-1])
        return mat

    @staticmethod
    def cosine_similarity(mat1: csr_matrix, mat2: csr_matrix):
        result = (mat1.T @ mat2).toarray()
        norm_1 = norm(mat1, axis=0).reshape(-1, 1)
        norm_2 = norm(mat2, axis=0).reshape(1, -1)

        return result / (norm_1 * norm_2)


def read_queries_csv(data_location='test-queries.tsv'):
    return pd.read_csv(data_location,
                       sep='\t', header=None,
                       names=['qid', 'content']
                       ).drop_duplicates().reset_index(drop=True)


def get_tf(inverted_indexes):
    # (1, doc_n)
    doc_len = inverted_indexes.sum(axis=0)  # 一篇文章总共多少次
    # (vocab_n, doc_n) * (1, doc_n)
    return Help.scale_csr(inverted_indexes, 1 / doc_len)


def generate_tf_idf(tf, idf):
    return Help.scale_csr(tf, idf)


def get_idf(inverted_indexes, np_log=np.log, add_half=False):
    n = inverted_indexes.shape[1]
    non_zeros = inverted_indexes.indptr[1:] - inverted_indexes.indptr[:-1]
    return np_log((n - non_zeros + .5) / (non_zeros + .5)) if add_half else np_log(n / non_zeros)


def select_first100(scores, remove_negative=True):
    result = np.zeros((200 * 100, 3)) * np.nan
    for i in trange(200):
        qid = queries_dataframe.loc[i].qid
        candidates_pids = candidates_passages_dataframe[candidates_passages_dataframe.qid == qid].pid.values
        candidates_pids_idxs = passages_dataframe[passages_dataframe.pid.isin(candidates_pids)].index.values

        score = scores[i, candidates_pids_idxs]


        first_100 = np.argsort(score)[::-1][:100]

        indexes = first_100[score[first_100] > 0] if remove_negative else first_100

        pids = candidates_pids[indexes]

        result_idx = i * 100
        result[result_idx:result_idx + 100, 0] = qid
        result[result_idx:result_idx + len(indexes), 1] = pids
        result[result_idx:result_idx + len(indexes), 2] = score[indexes]

    result = pd.DataFrame(result, columns=['qid', 'pid', 'score']).dropna()
    result[['pid', 'qid']] = result[['pid', 'qid']].astype(int)
    return result


def get_p_length_normalized(inverted_indexes_p):
    doc_len = inverted_indexes_p.sum(axis=0)
    avdl = doc_len.mean()  # average document(passage) length
    return doc_len / avdl


def get_bm25(tf_p, tf_q, idf, p_len_normalized, k1=1.2, k2=100, b=.75):
    K = k1 * ((1 - b) + b * p_len_normalized)  # different for every passage

    temp0 = ((k1 + 1) * tf_p)
    temp1 = Help.sparse_add_vec(tf_p, K)
    temp1.data = 1 / temp1.data
    S1 = temp0.multiply(temp1)

    left = Help.scale_csr(S1, idf)

    temp0 = ((k2 + 1) * tf_q)
    temp1 = tf_q.copy()
    temp1.data = 1 / (temp1.data + k2)
    right = temp0.multiply(temp1)

    return right.T @ left


verbose = __name__ == '__main__'


def ifprint(s, **kwargs):
    if verbose:
        print(s, **kwargs)


ifprint('Loading dataframes from files')
queries_dataframe = read_queries_csv()
queries_indexes = generate_indexes(queries_dataframe)
candidates_passages_dataframe = read_all_csv()

if __name__ == '__main__':
    # 1. Extract IDF
    ifprint('Extract IDF')
    idf = get_idf(passages_indexes)

    # 2. Extract TF-IDF of passages
    ifprint('Extract TF-IDF of passages')
    passages_tf = get_tf(passages_indexes)
    passages_tfidf = generate_tf_idf(passages_tf, idf)

    # 3. Using idf_psgs, extract TF-IDF of queries.
    ifprint('Extract the TF-IDF of queries')

    queries_tf = get_tf(queries_indexes)
    queries_tfidf = generate_tf_idf(queries_tf, idf)

    # 4. Use a basic vector space model with TF-IDF and cosine similarity
    ifprint('Calculate cosine similarity scores')
    similarity_scores = Help.cosine_similarity(queries_tfidf, passages_tfidf)

    # 5. retrieve at most 100 passages from the 1000 passages for each query
    # no headers, expected to have 19,290 rows
    ifprint('Retrieve passages for each query', end=' ')
    similarity_result = select_first100(similarity_scores) # (200, 182469)
    ifprint(f'rows:{similarity_result.shape[0]}')

    # 6. Store the outcomes in a file named tfidf.csv
    ifprint('Store results')
    similarity_result.to_csv('tfidf.csv', header=False, index=False)

    # 7. Use inverted index to implement BM25
    # while setting k1 = 1.2, k2 = 100, and b = 0.75.
    ifprint('Calculate BM25 scores')
    bm25_scores = get_bm25(tf_p=passages_tf, tf_q=queries_tf, idf=idf,
                           p_len_normalized=get_p_length_normalized(passages_indexes)).toarray()

    # 8. Retrieve at most 100 passages from within the 1000 passages for each query.
    ifprint('Retrieve passages for each query', end=' ')
    bm25_result = select_first100(bm25_scores)
    ifprint(f'rows:{bm25_result.shape[0]}')

    # 9. Store the outcomes in a file named bm25.csv
    ifprint('Store results')
    bm25_result.to_csv('bm25.csv', header=False, index=False)

    ifprint('------complete------')
