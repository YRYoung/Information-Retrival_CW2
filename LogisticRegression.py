"""
Logistic Regression (LR)(25 marks)
---------------------------------------
Represent passages and query based on a word embedding method,
(such as Word2Vec, GloVe, FastText, or ELMo).

Compute query (/passage) embeddings by averaging embeddings of all the words in that query (/passage).

With these query and passage embeddings as input, implement a logistic regression model
to assess relevance of a passage to a given query.
---------------------------------------
Describe how you perform input processing & representation or features used.

Using the metrics you have implemented in the previous part,
report the performance of your model based on the validation data.

Analyze the effect of the learning rate on the model training loss.

(All implementations for logistic regression algorithm must be your own for this part.)

"""

import numpy as np
import pandas as pd
import torch
from icecream import ic
from tqdm import tqdm

from eval import init_evaluator
from utils import queries_embeddings, train_raw_df, load_passages_tensors, map_location


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def BCEloss(y, h):
    return y * np.log(h) + (1 - y) * np.log(1 - h)


class LogisticRegression():
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations

    def _init_weights(self, w=None, b=None):
        self.w = w if w is not None else np.zeros(600)
        self.b = b if b is not None else 0
        self.losses = np.zeros(self.n_iterations)

    def fit(self, dataloader, evaluator=None):
        self._init_weights()
        if evaluator is not None:
            self.accuracies = []

        # gradient descent
        for epoch in range(self.n_iterations):
            ic(epoch)

            pbar = tqdm(enumerate(dataloader), unit='batch', total=len(dataloader))
            for i_batch, (x_batch, y_batch) in pbar:
                loss = self._fit_batch(x_batch, y_batch)

                pbar.set_postfix({'loss': loss})
                self.losses[epoch] += loss
            self.losses /= i_batch + 1
            if evaluator is not None:
                self.accuracies.append(evaluator(self.forward))

    def _fit_batch(self, x, y):
        #         weights=np.int(y==1)
        n = x.shape[0]
        h = self.forward(x)
        tmp = h - y

        dw = x.T.dot(tmp)
        db = np.einsum('i->', tmp)

        scaler = self.learning_rate / n
        self.w -= dw * scaler
        self.b -= db * scaler

        return - np.einsum('i->', BCEloss(y, h)) / n

    def forward(self, x):
        res = sigmoid(x.dot(self.w) + self.b)
        return res

    def save(self, path):
        return torch.save({'w': self.w, "b": self.b,
                           'losses': self.losses, 'accuracies': self.accuracies}, path)

    def load(self, path):
        value = torch.load(path)
        self._init_weights(w=value['w'], b=value['b'])
        self.losses = value['losses']
        self.accuracies = value['accuracies']


class DataLoader:
    def __init__(self, dataframe: pd.DataFrame, batch_size, p_tensors):
        self.current_pth = -1
        self.p_tensors = p_tensors
        self.q_tensors = torch.load(queries_embeddings, map_location=map_location)
        self.df = dataframe.sort_values(by=['pid'])[['qid', 'pid', 'relevancy']]
        self.N = len(dataframe)
        self.batch_size = batch_size
        self.num_batches = self.N // self.batch_size + 1
        ic(self.N, self.num_batches, self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for start in range(0, self.N, self.batch_size):
            end = min(start + self.batch_size, self.N)
            this_batch_size = end - start
            df = self.df.iloc[start:end]
            df.reset_index(drop=True, inplace=True)

            queries = torch.zeros((this_batch_size, 300))
            passages = torch.zeros((this_batch_size, 300))
            for i, row in df.iterrows():
                queries[i, :] = self.q_tensors[row.qid]
                passages[i, :] = self.p_tensors[row.pid]

            x = torch.stack([queries, passages], dim=2).numpy().reshape(-1, 600)
            y = df.relevancy.values.reshape(-1)
            yield x, y


if __name__ == '__main__':
    epoch = 50
    learning_rate = 0.005
    dataloader = DataLoader(pd.read_parquet(train_raw_df), 1024, load_passages_tensors())
    model = LogisticRegression(learning_rate=learning_rate, n_iterations=epoch)
    # model.load('./lrmodel.pth')
    evaluator = init_evaluator(
        x_val_handler=lambda x: x.numpy().reshape(-1, 600))
    model.save(f'./all_{epoch}_{str(learning_rate)}.pth')

    model.fit(dataloader, evaluator)
