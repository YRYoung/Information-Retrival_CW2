import numpy as np
import pandas as pd

from eval import data_path, read_csv

output_path = f'{data_path}/temp1'
val_tsv = f'{data_path}/part2/validation_data.tsv'
train_tsv = f'{data_path}/part2/train_data.tsv'
sample_tsv = f'{data_path}/part2/sample_data.tsv'


def add_q_idx():
    df = read_csv(csv_path=train_tsv)
    df1 = df.sort_values(by=['qid', 'pid'])
    _, b = np.unique(df1.qid.values, return_counts=True)
    df1['p_idx'] = np.repeat(np.arange(len(b)), b)

    df1.to_parquet(f'{output_path}/train_data_raw.parquet.gzip', compression='gzip')
def clean():

