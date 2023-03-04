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


# __all__ = ['Help', 'select_first100']


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


def select_first_n(scores, passages_dataframe, queries_dataframe, pairs_dataframe,
                   file_path, remove_negative=True, first_n=1000):
    size = len(queries_dataframe)
    result = np.zeros((first_n, 3)) * np.nan
    for i in trange(size):
        result *= np.nan
        qid = queries_dataframe.loc[i].qid
        candidates_pids = pairs_dataframe[pairs_dataframe.qid == qid].pid.values
        candidates_pids_idxs = passages_dataframe[passages_dataframe['pid'].isin(candidates_pids).values].index.values
        # passages_dataframe[passages_dataframe.pid.isin(candidates_pids)].index.values

        score = scores[i, candidates_pids_idxs]

        first_n_idx = np.argsort(score)[::-1][:first_n]

        indexes = first_n_idx[score[first_n_idx] > 0] if remove_negative else first_n_idx

        pids = candidates_pids[indexes]

        result[:, 0] = qid
        result[: len(indexes), 1] = pids
        result[: len(indexes), 2] = score[indexes]

        df = pd.DataFrame(result, columns=['qid', 'pid', 'score']).dropna()
        df[['pid', 'qid']] = df[['pid', 'qid']].astype(int)
        df.to_csv(file_path, header=False, index=False, mode='a+')


def get_p_length_normalized(inverted_indexes_p):
    doc_len = inverted_indexes_p.sum(axis=0)
    avdl = doc_len.mean()  # average document(passage) length
    return doc_len / avdl


class BM25Score:
    def __init__(self, tf_p, tf_q, idf, p_len_normalized, k1=1.2, k2=100, b=.75):
        # setting k1 = 1.2, k2 = 100, and b = 0.75.

        self.tf_p = tf_p  # (vocab_n, doc_n)
        self.idf = idf  # (vocab_n, )

        self.K = k1 * ((1 - b) + b * p_len_normalized)  # (1, doc_n)
        self.temp0 = ((k1 + 1) * tf_p)  # (vocab_n, doc_n)

        temp0 = ((k2 + 1) * tf_q)
        temp1 = tf_q.copy()
        temp1.data = 1 / (temp1.data + k2)
        self.right = temp0.multiply(temp1).T

    def __getitem__(self, item):
        i, j = item
        temp0 = self.temp0[:, j]  # (vocab_n, j_len)
        temp1 = 1 / (self.tf_p[:, j] + self.K[:, j])  # (vocab_n, j_len)

        S1 = temp0.multiply(temp1)  # (vocab_n, j_len)

        left = Help.scale_csr(S1, self.idf)

        return (self.right[i] @ left).toarray().reshape(-1)


def get_bm25(tf_p, tf_q, idf, p_len_normalized):
    return BM25Score(tf_p, tf_q, idf, p_len_normalized)


verbose = __name__ == '__main__'


def ifprint(s, **kwargs):
    if verbose:
        print(s, **kwargs)
