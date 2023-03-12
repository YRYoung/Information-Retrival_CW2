import numpy as np
import pandas as pd

from eval import data_path, read_csv

output_path = f'{data_path}/temp1'
val_tsv = f'{data_path}/part2/validation_data.tsv'
train_tsv = f'{data_path}/part2/train_data.tsv'
sample_tsv = f'{data_path}/part2/sample_data.tsv'


def add_q_idx():
    df = read_csv(train_tsv)
    df1 = df.sort_values(by=['qid', 'relevancy'], ascending=[True, False])
    _, count_repeats = np.unique(df1.qid.values, return_counts=True)

    df1['q_idx'] = np.repeat(np.arange(len(count_repeats)), count_repeats)

    df1['p_idx'] = np.hstack([np.arange(bb) for bb in count_repeats])
    df1.to_parquet(out_file, compression='gzip')


def clean():

