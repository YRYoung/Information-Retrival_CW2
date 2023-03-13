import numpy as np
import pandas as pd
import torch
from flair.data import Sentence

from eval import read_csv

data_path = '../data'

output_path = f'{data_path}/temp1'
val_tsv = f'{data_path}/part2/validation_data.tsv'
train_tsv = f'{data_path}/part2/train_data.tsv'
sample_tsv = f'{data_path}/part2/sample_data.tsv'

out_file_train = f'{output_path}/train_data_raw.parquet.gzip'


def add_q_idx(dataframe=read_csv(train_tsv),
              save=out_file_train):
    df1 = dataframe.sort_values(by=['qid', 'relevancy'], ascending=[True, False])
    _, count_repeats = np.unique(df1.qid.values, return_counts=True)

    df1['q_idx'] = np.repeat(np.arange(len(count_repeats)), count_repeats)

    df1['p_idx'] = np.hstack([np.arange(bb) for bb in count_repeats])
    df1.to_parquet(save, compression='gzip')


def clean(dataframe=pd.read_parquet(out_file_train),
          save=f'{output_path}/train_data_del9.parquet.gzip'):
    choose = dataframe['relevancy'] == 0
    bg_df = dataframe[choose]
    target_df = dataframe[~choose]

    selected_bg_df = bg_df.sample(n=len(bg_df) // 10, random_state=1)
    selected_df = pd.concat([selected_bg_df, target_df])
    selected_df = selected_df.sort_values(by=['qid', 'relevancy', 'p_idx'], ascending=[True, False, True])

    selected_df.to_parquet(save, compression='gzip')


# @lru_cache(maxsize=None)
def _processing(all_df, embedding):
    avg_embedding = []

    for content in ['query', 'passage']:
        print(f'\t embedding {content}', end='')
        sentences = [Sentence(p) for _, p in all_df[content].iterrows()]
        embedding.embed(sentences)

        result = torch.stack([torch.stack(
            [token.embedding for token in sent.tokens]).mean(dim=0) for sent in sentences])

        avg_embedding.append(result)  # (N, 300)
        print(f': {avg_embedding[-1].shape}')

    x = torch.stack(avg_embedding, dim=2)  # (N, 300, 2) float32
    y = torch.tensor(all_df.relevancy.values, dtype=torch.float32)  # (N, ) float32

    return x, y