import pandas as pd
import torch
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from torch.utils.data import Dataset
from icecream import ic


def processing(row: pd.Series, embedding: WordEmbeddings):
    sentences = [Sentence(s) for s in [row['query'], row['passage']]]
    embedding.embed(sentences)
    # (2, 300)
    x = torch.stack([torch.stack(
        [token.embedding for token in sent.tokens]).mean(dim=0) for sent in sentences])

    y = torch.tensor(row.relevancy, dtype=torch.float32)  # (1) float32

    return x, y


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        super(Dataset).__init__()

        print(f'processing data: ', end='')

        self.dataframe = dataframe
        self.N = len(self.dataframe)
        ic(self.N)

        self.embedding = WordEmbeddings('en')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # all_df = self.dataframe[self.dataframe['qidx'] == idx]
        return processing(self.dataframe.iloc[idx], self.embedding)
