"""
Evaluating Retrieval Quality. (20 marks)
---------------------------------------------------------------------
Implement methods to compute the average precision and NDCG metrics.
Compute the performance of using BM25 as the retrieval model on the validation data (validation_data.tsv ) using these metrics.
Your marks for this part will mainly depend on the implementation of metrics 
(as opposed to your implementation of BM25,
since we already focused on that as part of the first assignment).
---------------------------------------------------------------------
Report: mAP@3, 10, 100 and NDCG@3, 10, 100
mAP should be a single value regardless of individual query
"""
import warnings

import numpy as np

import pandas as pd

from cw1.task1 import *
from cw1.task2 import *
from cw1.task3 import *
from tqdm import trange
data_path = './data'


def get_tokens(load=True, use_stop_words=False):
    vocab_file_name = [f'{data_path}/temp/vocab.npy', f'{data_path}/temp/vocab_no_sw.npy']
    if not load:
        vocab_txt = f'{data_path}/part2/validation_data.tsv'

        tokens = np.array(preprocessing(read_txt(vocab_txt), remove_stop_words=False, verbose=True))
        np.save(vocab_file_name[0], tokens)

        tokens_no_sw = remove_stop_words(tokens)
        np.save(vocab_file_name[1], tokens_no_sw)
    else:
        tokens = np.load(vocab_file_name[0])
        tokens_no_sw = np.load(vocab_file_name[1])
    return tokens if use_stop_words else tokens_no_sw


def get_dfs(csv_path=f'{data_path}/dataset/candidate_passages_top1000.tsv'):
    def read_csv(csv_path):
        df = pd.read_csv(csv_path,
                         sep='\t', header=0,
                         names=['qid', 'pid', 'query', 'passage', 'relevancy']).drop_duplicates()
        return df.reset_index(drop=True)

    df = read_csv(csv_path)

    p_df = df[['pid', 'passage']].drop_duplicates().reset_index(drop=True)
    p_df.columns = [['pid', 'content']]

    q_df = df[['qid', 'query']].drop_duplicates().reset_index(drop=True)
    q_df.columns = [['qid', 'content']]
    return df, p_df, q_df


def get_indexes(tokens, p_df, q_df, load=True, verbose=True,
                file_path=(f'{data_path}/temp/var_passages_idx.pkl', f'{data_path}/temp/var_queries_idx.pkl')):
    if load:
        with open(file_path[0], 'rb') as file:
            p_idx = pickle.load(file)
        with open(file_path[1], 'rb') as file:
            q_idx = pickle.load(file)
    else:

        p_idx = generate_indexes(p_df, tokens, verbose=verbose)
        q_idx = generate_indexes(q_df, tokens, verbose=verbose)
        with open(file_path[0], 'wb') as file:
            pickle.dump(p_idx, file)

        with open(file_path[1], 'wb') as file:
            pickle.dump(q_idx, file)

    return p_idx, q_idx


def get_bm25_var(p_idx, q_idx, p_df, q_df, df, load=False):
    file_path = f'{data_path}/temp/bm25.csv'
    if not load:
        print('Calculate BM25 scores')
        bm25_scores = get_bm25(tf_p=p_idx, tf_q=q_idx,
                               idf=get_idf(p_idx, add_half=True),
                               p_len_normalized=get_p_length_normalized(p_idx))

        select_first_n(bm25_scores, p_df, q_df, df, file_path=file_path)
        print('------complete------')
    return pd.read_csv(file_path, header=None, names=['qid', 'pid', 'score'])


def eval_scores(scores, df, queries_df, log=np.log2):
    size = len(queries_df)
    ndcg = np.zeros(size)
    precisions = np.zeros(size)
    for i in trange(size):
        qid = queries_df.loc[i].qid
        query_df = scores[scores['qid'] == qid].reset_index()

        relevant_pid = df[(df['qid'] == qid) & (df.relevancy == 1)].pid.values
        relevant_idx = query_df[query_df.pid.isin(relevant_pid)].index.values

        # assert len(relevant_pid) == 1, f'i = {i}, qid = {qid}'
        if len(relevant_idx) == 0:
            continue
        with warnings.catch_warnings(record=True) as w:
            dcg = 1 / log(2 + relevant_idx)
            if len(w) > 0:
                print(i)
        best = 1 / log(2 + np.arange(len(relevant_pid)))
        ndcg[i] = np.sum(dcg) / np.sum(best)

        precision = (np.arange(len(relevant_idx)) + 1) / (relevant_idx + 1)
        precisions[i] = np.sum(precision) / len(relevant_pid)

    eval_df = queries_df.qid.copy()
    eval_df['precision'] = precisions
    eval_df['NDCG'] = ndcg
    return eval_df


if __name__ == '__main__':
    tokens = get_tokens(load=True)
    all_df, passages_df, queries_df = get_dfs(csv_path=f'{data_path}/part2/validation_data.tsv')
    passages_indexes, queries_indexes = get_indexes(tokens, passages_df, queries_df, load=True)
    bm25_scores = get_bm25_var(passages_indexes, queries_indexes, passages_df, queries_df, all_df, load=True)

    eval_df = eval_scores(bm25_scores, all_df, queries_df)
    eval_df.to_csv(f'{data_path}/temp/eval_bm25_val.csv', header=True, index=False)

    relevant_pairs = all_df[all_df.relevancy == 1][['qid', 'pid']]
    relevant_pairs_scores = pd.merge(bm25_scores, relevant_pairs, on=['qid', 'pid'])
    print(relevant_pairs_scores.score.mean())

    irrelevant_pairs = all_df[all_df.relevancy != 1][['qid', 'pid']]
    irrelevant_pairs_scores = pd.merge(bm25_scores, irrelevant_pairs, on=['qid', 'pid'])
    print(irrelevant_pairs_scores.score.mean())
