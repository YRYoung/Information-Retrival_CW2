import random

import numpy as np
import pandas as pd
import torch
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from icecream import ic
from tqdm import tqdm

from eval import read_csv

data_path = './data'

output_path = f'{data_path}/temp1'
val_tsv = f'{data_path}/part2/validation_data.tsv'
train_tsv = f'{data_path}/part2/train_data.tsv'
sample_tsv = f'{data_path}/part2/sample_data.tsv'
df_path = f'{data_path}/dataframes'
train_raw_df = f'{df_path}/train_data_raw.parquet.gzip'
val_raw_df = f'{df_path}/val_data_raw.parquet.gzip'



def add_q_idx(dataframe,
              save=train_raw_df):
    df1 = dataframe.sort_values(by=['qid', 'relevancy'], ascending=[True, False])
    _, count_repeats = np.unique(df1.qid.values, return_counts=True)

    df1['q_idx'] = np.repeat(np.arange(len(count_repeats)), count_repeats)

    df1['p_idx'] = np.hstack([np.arange(count) for count in count_repeats])
    df1.to_parquet(save, compression='gzip')


def clean(dataframe,
          save=f'{data_path}/raw_dataframes/train_data_cleaned.parquet.gzip'):
    # delete less than 1000, reserving targets (+293):
    _, count_repeats = np.unique(dataframe.qid.values, return_counts=True)
    drop_q_idx = np.where(count_repeats == 1000)[0]
    dataframe = dataframe[dataframe.q_idx.isin(drop_q_idx) | dataframe.relevancy == 1]

    # re-index queries
    del dataframe['q_idx']
    _, count_repeats = np.unique(dataframe.qid.values, return_counts=True)
    # dataframe['q_idx'] = np.repeat(np.arange(len(count_repeats)), count_repeats).copy()
    dataframe.assign(q_idx=np.repeat(np.arange(len(count_repeats)), count_repeats))

    dataframe.to_parquet(save, compression='gzip')


def subsample(dataframe: pd.DataFrame, save):
    choose = dataframe['relevancy'] == 0
    bg_df = dataframe[choose]
    target_df = dataframe[~choose]

    selected_bg_df = bg_df.sample(n=len(bg_df) // 100, random_state=1)
    selected_df = pd.concat([selected_bg_df, target_df])
    selected_df = selected_df.sort_values(by=['qid', 'relevancy', 'p_idx'], ascending=[True, False, True])

    selected_df.to_parquet(save, compression='gzip')


def _processing(all_df, embedding):
    avg_embedding = []

    for content in ['query', 'passage']:
        print(f'\t embedding {content}', end='')
        sentences = [Sentence(p) for p in all_df[content].tolist()]
        embedding.embed(sentences)

        result = torch.stack([torch.stack(
            [token.embedding for token in sent.tokens]).mean(dim=0) for sent in sentences])

        avg_embedding.append(result)  # (N, 300)
        print(f': {avg_embedding[-1].shape}')

    x = torch.stack(avg_embedding, dim=2)  # (N, 300, 2) float32
    y = torch.tensor(all_df.relevancy.values, dtype=torch.float32)  # (N, ) float32

    return x, y


def embed_all(dataframe, embedding, save_path):
    x, y = _processing(dataframe, embedding)
    torch.save([x, y], save_path)


def embed_queries(raw_df: pd.DataFrame, embedding, save_path=f'{data_path}/p_embeddings', passage=True):
    id, content = ('pid', 'passage') if passage else ('qid,''query')

    sub_df = raw_df[[id, content]].drop_duplicates()
    del raw_df
    pbar = tqdm(sub_df.itertuples(), total=len(sub_df), unit=content)

    data={}
    previous_pid = 1
    for row in pbar:
        pid = row[1]
        sentence = Sentence(row[2])

        embedding.embed(sentence)
        data[pid] = torch.stack([token.embedding for token in sentence.tokens]).mean(dim=0)

        if pid % 100000 == 0:
            file_name = f'{save_path}/{previous_pid}_{pid}.pth'
            torch.save(data, file_name)
            data = {}
            previous_pid = pid
            pbar.set_postfix({'file_name': file_name})




if __name__ == "__main__":
    embed_all(dataframe=pd.read_parquet(val_raw_df),
              embedding=WordEmbeddings('en'),
              save_path=f'{data_path}/val_embeddings.pth')

    # ic(clean(pd.read_parquet(train_raw_df), save=f'{df_path}/train_data_cleaned.parquet.gzip'))
    #
    # subsample(pd.read_parquet(f'{df_path}/train_data_cleaned.parquet.gzip'),
    #           save=f'{df_path}/train_debug.parquet.gzip')
    # embed_queries(pd.read_parquet(train_raw_df), embedding=WordEmbeddings('en'))