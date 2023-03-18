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

https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832
https://towardsdatascience.com/mean-average-precision-at-k-map-k-clearly-explained-538d8e032d2
"""
import torch
from huepy import *

import pandas as pd
import numpy as np
from utils import val_raw_df

__all__ = [ 'eval_per_query', 'init_evaluator']


def init_evaluator(at: list = [3, 10, 100], x_val_handler=None, prepare_x=True):
    val_df = pd.read_parquet(val_raw_df)
    del val_df['query']
    del val_df['passage']
    del val_df['qid']
    del val_df['pid']
    del val_df['p_idx']
    if prepare_x:
        x_val = torch.load('./data/val_embeddings.pth')[0]

        if x_val_handler is not None:
            x_val = x_val_handler(x_val)

    _, count_repeats = np.unique(val_df.q_idx.values, return_counts=True)
    p_idx = np.hstack([np.arange(count) for count in count_repeats])
    num_queries = len(count_repeats)

    precisions = np.zeros((len(at), num_queries))
    ndcgs = np.zeros((len(at), num_queries))

    def eval_one(predict):
        val_df['predict_relevancy'] = predict(x_val) if prepare_x else predict

        df = val_df.sort_values(by=['q_idx', 'predict_relevancy'], ascending=[True, False])
        del val_df['predict_relevancy']

        df.loc[:, 'p_idx'] = p_idx
        retrieved_relevant = df[df['relevancy'] == 1][['q_idx', 'p_idx']]

        _, idx, counts = np.unique(retrieved_relevant['q_idx'], return_counts=True, return_index=True)

        for i, (start, l) in enumerate(zip(idx, counts)):
            rel_idx = retrieved_relevant[start:start + l]['p_idx'].values + 1
            precision, ndcg = eval_per_query(rel_idx, at)
            precisions[:, i] = precision
            ndcgs[:, i] = ndcg

        avg_precision = np.mean(precisions, axis=1)
        avg_ndcg = np.mean(ndcgs, axis=1)

        [print(blue(italic(f'mAP @ {now}: {value}'))) for now, value in zip(at, avg_precision)]
        [print(orange(italic(f'NDCG @ {now}: {value}'))) for now, value in zip(at, avg_ndcg)]

        return avg_precision, avg_ndcg

    return lambda predictor: eval_one(predictor)


def eval_per_query(relev_rank, at: list[int], log=np.log):
    relev_rank.sort()
    dcg = 1 / log(1 + relev_rank)
    # todo: np.cumsum(a)
    precisions, ndcg = np.zeros(len(at)), np.zeros(len(at))

    for j, now in enumerate(at):
        relev_rank_now = relev_rank[relev_rank <= now]
        num_relev = len(relev_rank_now)
        if num_relev == 0:
            continue

        precision = (np.arange(num_relev) + 1) / relev_rank_now
        precisions[j] = np.sum(precision) / num_relev

        ideal_dgc = np.sum(1 / log(2 + np.arange(len(relev_rank))))
        ndcg[j] = np.sum(dcg[:now]) / ideal_dgc

    return precisions, ndcg


