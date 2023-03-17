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
import os

import numpy as np
import pandas as pd
import torch
from icecream import ic
from tqdm import tqdm

from utils import queries_embeddings, train_debug_df, passages_embeddings, map_location


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def BCEloss(y, h):
    return y * np.log(h) + (1 - y) * np.log(1 - h)


class LogisticRegression():
    def __init__(self, learning_rate, n_iterations):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.accuracies = []
        self.continue_training = False
        self.losses = np.array([])
        self.w = None
        self.b = None

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
    def __init__(self, dataframe: pd.DataFrame, batch_size, p_tensors):
        self.current_pth = -1
        self.p_tensors = p_tensors
        self.q_tensors = torch.load(queries_embeddings, map_location=map_location)
        self.df = dataframe.sort_values(by=['qid'])[['qid', 'pid', 'relevancy']]
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

    pth_files = os.listdir(passages_embeddings)
    pth_files.sort(key=lambda x: int(x[:-4]))
    p_tensors_all = {}
    [p_tensors_all.update(torch.load(
        f'{passages_embeddings}/{name}',
        map_location=torch.device('cpu'))) for name in tqdm(pth_files)]

    import eval

    evaluator = eval.init_evaluator(
        x_val_handler=lambda x: x.numpy().reshape(-1, 600))
    dataloader = DataLoader(pd.read_parquet(train_debug_df), 1024, p_tensors_all)

    for lr in [0.02, 0.005, 0.1]:
        model = LogisticRegression(learning_rate=lr, n_iterations=400)

        model.fit(dataloader, evaluator)

        name = f'./debug_400_{lr:.3f}'
        model.save(f'{name}.pth')

        dff = model.get_history()
        dff.to_parquet(f'{name}.dataframe')
