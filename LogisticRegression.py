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

from NN.CustomDataset import CustomDataset
from utils import queries_embeddings, load_passages_tensors, \
    train_raw_df


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def BCEloss(y, h):
    return y * np.log(h) + (1 - y) * np.log(1 - h)


class LogisticRegression:
    def __init__(self, learning_rate, n_iterations):

        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.accuracies = []
        self.continue_training = False
        self.losses = np.array([])
        self.w = None
        self.b = None
        self.history = None

    def _init_weights(self):
        if self.w is None and self.b is None:
            self.w = np.zeros(600)
            self.b = 0

        self.losses = np.concatenate([self.losses, np.zeros(self.n_iterations)])

    def fit(self, dataloader, evaluator=None):
        start_epoch = len(self.losses)

        self._init_weights()

        # gradient descent
        for epoch in range(start_epoch, start_epoch + self.n_iterations):
            ic(epoch)

            pbar = tqdm(enumerate(dataloader), unit='batch', total=len(dataloader))
            for i_batch, (x_batch, y_batch) in pbar:
                loss = self._fit_batch(x_batch, y_batch)

                pbar.set_postfix({'loss': loss})
            self.losses[epoch] = loss

            if evaluator is not None:
                self.accuracies.append(evaluator(self.forward))

        self.get_history()
        print("done")

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
        return torch.save({'w': self.w, "b": self.b, 'history': self.history,
                           'losses': self.losses, 'accuracies': self.accuracies}, path)

    def load(self, path):
        value = torch.load(path)
        self.w = value['w']
        self.b = value['b']
        self.losses = value['losses']
        self.losses = self.losses[self.losses != 0]
        self.accuracies = value['accuracies']
        self.history = value['history']
        self.continue_training = True

    def get_history(self):
        result_df = pd.DataFrame(self.losses, columns=['Loss'])
        result_df.loc[:, ['mAP@3', 'mAP@10', 'mAP@100']] = [a[0] for a in self.accuracies]
        result_df.loc[:, ['NDCG@3', 'NDCG@10', 'NDCG@100']] = [a[1] for a in self.accuracies]
        self.history = result_df.iloc[:, [1, 2, 3, 4, 5, 6, 0]]
        return self.history


class DataLoader:
    def __init__(self, batch_size: int, passages_per_query: int, p_tensors=None, dataframe=None, q_tensors=None, ):

        if p_tensors is None:
            p_tensors = load_passages_tensors()

        if q_tensors is None:
            q_tensors = torch.load(queries_embeddings, map_location=torch.device('cpu'))

        if dataframe is None:
            dataframe = pd.read_parquet(train_raw_df)

        self.dataset = ValidationDataset(all_dataframe=dataframe,
                                         val_p_tensors=p_tensors,
                                         queries_tensors=q_tensors,
                                         passages_per_query=passages_per_query)

        _, counts = np.unique(dataframe.qid.values, return_counts=True)
        self.valid_q_i = np.where(counts > passages_per_query)[0]
        ic(self.valid_q_i)
        self.passages_per_query = passages_per_query
        self.num_queries = len(self.valid_q_i)

        self.batch_size = batch_size
        self.num_batches = self.num_queries // self.batch_size + 1

        ic('DataLoader', self.num_queries, self.num_batches, self.batch_size)

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        for start in range(0, self.num_queries, self.batch_size):
            end = min(start + self.batch_size, self.num_queries)
            this_batch_size = end - start

            x = np.zeros((this_batch_size * self.passages_per_query, 2, 300))
            y = np.zeros(this_batch_size * self.passages_per_query)
            for indice, q_idx in enumerate(self.valid_q_i[start:end]):
                xx, yy = self.dataset[q_idx]
                idx_start = indice * self.passages_per_query
                idx_end = idx_start + self.passages_per_query

                x[idx_start:idx_end, ...] = xx
                y[idx_start:idx_end] = yy

            yield x.reshape(-1, 600), y
