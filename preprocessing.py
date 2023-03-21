import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import cw1.task1
from utils import train_raw_df, val_raw_df

data_path = './data'

output_path = f'{data_path}/temp1'
df_path = f'{data_path}/dataframes'


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
    from flair.data import Sentence

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


def embed_queries(raw_df: pd.DataFrame, embedding, save_path,
                  passage=True, embedded_pids=None):
    from flair.data import Sentence
    id, content = ('pid', 'passage') if passage else ('qid', 'query')

    sub_df = raw_df[[id, content]].drop_duplicates()
    del raw_df
    pbar = tqdm(sub_df.itertuples(), total=len(sub_df), unit=content)

    data = {}
    previous_pid = 1
    for row in pbar:
        pid = row[1]
        if embedded_pids is not None and pid in embedded_pids:
            continue
        sent_str = row[2]
        # if tokenize_callback is not None and callable(tokenize_callback):
        #     sent_str = tokenize_callback(sent_str, return_countter=False)

        sentence = Sentence(sent_str)

        embedding.embed(sentence)
        data[pid] = torch.stack([token.embedding for token in sentence.tokens]).mean(dim=0)

        if passage and pid % 100000 == 0:
            file_name = f'{save_path}/{previous_pid}.pth'
            torch.save(data, file_name)
            data = {}
            previous_pid = pid
            pbar.set_postfix({'file_name': file_name})

    final_filename = f'{save_path}/{previous_pid}.pth' if passage else save_path
    torch.save(data, final_filename)


def read_csv(csv_path) -> pd.DataFrame:
    df = pd.read_csv(csv_path,
                     sep='\t', header=0,
                     names=['qid', 'pid', 'query', 'passage', 'relevancy']).drop_duplicates()
    return df.reset_index(drop=True)


if __name__ == "__main__":
    from flair.embeddings import WordEmbeddings

    # embed_all(dataframe=pd.read_parquet(val_raw_df),
    #           embedding=WordEmbeddings('en'),
    #           save_path=f'{data_path}/val_embeddings.pth')
    embedding = WordEmbeddings('en')
    # ic(clean(pd.read_parquet(train_raw_df), save=f'{df_path}/train_data_cleaned.parquet.gzip'))

    # subsample(pd.read_parquet(f'{df_path}/train_data_cleaned.parquet.gzip'),
    #           save=f'{df_path}/train_debug.parquet.gzip')

    embed_queries(pd.read_parquet(val_raw_df), embedding=embedding,
                  passage=True, save_path=f'{data_path}/val_p_embeddings',
                  embedded_pids=pd.read_parquet(train_raw_df).pid.drop_duplicates())
                  # tokenize_callback=cw1.task1.preprocessing)
