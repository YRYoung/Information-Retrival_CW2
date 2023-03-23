import os
import time
from functools import wraps

import torch
from tqdm import tqdm

data_path = './data'

val_tsv = f'{data_path}/part2/validation_data.tsv'
train_tsv = f'{data_path}/part2/train_data.tsv'

dataframe_folder = f'{data_path}/dataframes'

train_raw_df = f'{dataframe_folder}/train_data_raw.parquet.gzip'
val_raw_df = f'{dataframe_folder}/val_data_raw.parquet.gzip'
train_debug_df = f'{dataframe_folder}/train_debug.parquet.gzip'

queries_embeddings = f'{data_path}/q_embeddings.pth'
passages_embeddings = f'{data_path}/p_embeddings'

val_passages_embeddings = f'{data_path}/val_p_embeddings'

map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def timeit(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f'Executing {func.__name__}')
        start = time.perf_counter()

        result = func(*args, **kwargs)

        end = time.perf_counter()
        print(f'------- {end - start:.6f}s -------')
        return result

    return wrapper


def load_passages_tensors(folder=passages_embeddings, first=None):
    pth_files = os.listdir(folder)

    pth_files.sort(key=lambda x: int(x[:-4]))
    if first is not None:
        pth_files = pth_files[:first]
    p_tensors_all = {}
    [p_tensors_all.update(torch.load(
        f'{folder}/{name}',
        map_location=map_location)) for name in tqdm(pth_files)]
    return p_tensors_all
