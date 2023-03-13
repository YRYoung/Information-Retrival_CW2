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

from huepy import *
from pandas import DataFrame


from cw1.task1 import *
from cw1.task2 import *
from cw1.task3 import *

data_path = './data'
output_path = f'{data_path}/temp0'

__all__ = ['eval_scores', 'read_csv', 'data_path']


def read_csv(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path,
                     sep='\t', header=0,
                     names=['qid', 'pid', 'query', 'passage', 'relevancy']).drop_duplicates()
    return df.reset_index(drop=True)


def to_dataframes(csv_path) -> tuple[DataFrame, DataFrame, DataFrame]:
    df = read_csv(csv_path)

    p_df = df[['pid', 'passage']].drop_duplicates().reset_index(drop=True)
    p_df.columns = [['pid', 'content']]

    q_df = df[['qid', 'query']].drop_duplicates().reset_index(drop=True)
    q_df.columns = [['qid', 'content']]
    return df, p_df, q_df


def eval_per_query(relevant_idx, at: list[int], log=np.log):
    dcg = 1 / log(1 + relevant_idx)
    precisions, ndcg = np.zeros(len(at)), np.zeros(len(at))

    for j, now in enumerate(at):
        relv_retd_idx = relevant_idx[relevant_idx <= now]
        total_relv_retd = len(relv_retd_idx)
        if not relv_retd_idx:
            continue

        precision = (np.arange(total_relv_retd) + 1) / relv_retd_idx
        precisions[j] = np.sum(precision) / total_relv_retd

        ideal_dgc = np.sum(1 / log(2 + np.arange(total_relv_retd)))
        ndcg[j] = np.sum(dcg[:now]) / ideal_dgc

    return precisions, ndcg


def eval_scores(scores, df, queries_df, log=np.log2, at: list[int] = [3, 10, 100]) -> DataFrame:
    # a query may correspond to 2 passages, but the relevancy is always 1 or 0
    size = len(queries_df)
    ndcg = np.zeros((len(at), size))
    precisions = np.zeros((len(at), size))

    for i in trange(size):
        qid = queries_df.loc[i].qid
        query_df = scores[scores['qid'] == qid].reset_index()

        # true relevant
        relevant_pid = df[(df['qid'] == qid) & (df.relevancy == 1)].pid.values
        # e.g., [2, 33]
        relevant_idx = query_df[query_df.pid.isin(relevant_pid)].index.values + 1

        precisions[:, i], ndcg[:, i] = eval_per_query(relevant_idx, at=at, log=log)

    eval_df = queries_df.qid.copy()

    for j, now in enumerate(at):
        eval_df[f'precision@{now}'] = precisions[j, :]
        eval_df[f'NDCG@{now}'] = ndcg[j, :]

    avg_precision = np.mean(precisions, axis=1)
    avg_ndcg = np.mean(ndcg, axis=1)

    [print(blue(italic(f'Average Precision @ {now}: {value}'))) for now, value in zip(at, avg_precision)]
    print('-' * 40)
    [print(orange(italic(f'Average NDCG @ {now}: {value}'))) for now, value in zip(at, avg_ndcg)]

    return eval_df


def _get_tokens(load=True, use_stop_words=False, vocab_txt=f'{data_path}/part2/validation_data.tsv'):
    vocab_file_name = [f'{output_path}/vocab.npy', f'{output_path}/vocab_no_sw.npy']
    if not load:

        tokens = np.array(preprocessing(read_txt(vocab_txt), remove_stop_words=False, verbose=True))
        np.save(vocab_file_name[0], tokens)

        tokens_no_sw = remove_stop_words(tokens)
        np.save(vocab_file_name[1], tokens_no_sw)
    else:
        tokens = np.load(vocab_file_name[0])
        tokens_no_sw = np.load(vocab_file_name[1])
    return tokens if use_stop_words else tokens_no_sw


def _get_indexes(tokens, p_df, q_df, load=True, verbose=True,
                 file_path=(f'{output_path}/var_passages_idx.pkl',
                            f'{output_path}/var_queries_idx.pkl')):
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


def _get_bm25_var(p_idx, q_idx, p_df, q_df, df, load=False, first_n=100):
    file_path = f'{output_path}/bm25.csv'
    if not load:
        print('Calculate BM25 scores')
        bm25_scores = get_bm25(tf_p=p_idx, tf_q=q_idx,
                               idf=get_idf(p_idx, add_half=True),
                               p_len_normalized=get_p_length_normalized(p_idx))

        select_first_n(bm25_scores, p_df, q_df, df, file_path=file_path, first_n=first_n)
        print('------complete------')
    return pd.read_csv(file_path, header=None, names=['qid', 'pid', 'score'])


if __name__ == '__main__':
    tokens = _get_tokens(load=False, vocab_txt=f'{data_path}/part2/train_data.tsv')
    all_df, passages_df, queries_df = to_dataframes(csv_path=f'{data_path}/part2/train_data.tsv')
    passages_indexes, queries_indexes = _get_indexes(tokens, passages_df, queries_df, load=False,
                                                     file_path=(f'{output_path}/train_passages_idx.pkl',
                                                                f'{output_path}/train_queries_idx.pkl'))
    bm25_scores = _get_bm25_var(passages_indexes, queries_indexes,
                                passages_df, queries_df, all_df,
                                load=False, first_n=100)

    eval_df = eval_scores(bm25_scores, all_df, queries_df)
    eval_df.to_csv(f'{output_path}/eval_bm25_train.csv', header=True, index=False)
