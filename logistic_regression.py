"""
Logistic Regression (LR)(25 marks)
---------------------------------------
Represent passages and query based on a word embedding method,
(such as Word2Vec, GloVe, FastText, or ELMo).

Compute query (/passage) embeddings by averaging embeddings of all the words in that query (/passage).

With these query and passage embeddings as input, implement a logistic regression model
to assess relevance of a passage to a given query.

Describe how you perform input processing & representation or features used.

Using the metrics you have implemented in the previous part,
report the performance of your model based on the validation data.

Analyze the effect of the learning rate on the model training loss.

(All implementations for logistic regression algorithm must be your own for this part.)

"""
import re
import sys

import pandas as pd
import torch
from flair.data import Sentence
from flair.embeddings import WordEmbeddings
from icecream import ic
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from eval import *

output_path = f'{data_path}/temp1'
val_tsv = f'{data_path}/part2/validation_data.tsv'
train_tsv = f'{output_path}/train_data_del9.parquet.gzip'
sample_tsv = f'{data_path}/part2/sample_data.tsv'



def processing(row: pd.Series, embedding: WordEmbeddings):
    sentences = [Sentence(s) for s in [row['query'], row['passage']]]
    embedding.embed(sentences)
    # (2, 300)
    x = torch.stack([torch.stack(
        [token.embedding for token in sent.tokens]).mean(dim=0) for sent in sentences])

    y = torch.tensor(row.relevancy, dtype=torch.float32)  # (1) float32

    return x, y


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=300):
        super(LogisticRegression, self).__init__()
        self.linear0 = nn.Linear(300, 1)
        self.linear1 = nn.Linear(2, 1)

    def forward(self, x):
        out = self.linear0(x).squeeze()
        out = self.linear1(out).squeeze()
        return torch.sigmoid(out)


class CustomDataset(Dataset):
    def __init__(self, file_path):
        super(Dataset).__init__()
        self.name = re.search(r'(?<=/)\w+(?=_data)', file_path).group()
        print(f'processing {self.name} data', end='')

        self.dataframe = pd.read_parquet(file_path)
        self.N = len(self.dataframe)
        ic(self.N)

        self.embedding = WordEmbeddings('en')

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # all_df = self.dataframe[self.dataframe['qidx'] == idx]
        return processing(self.dataframe.iloc[idx], self.embedding)


def train_test_batch(model, dataloader, optimizer, criterion, train, writer, total_batch):
    name = 'train' if train else 'eval'
    losses = torch.zeros(len(dataloader))
    model.train(train)
    pbar = tqdm(enumerate(dataloader),
                unit='batch', total=len(dataloader), desc=name)
    for i_batch, (x_batch, y_batch) in pbar:

        if train:
            optimizer.zero_grad()

        pred = model.forward(x_batch)
        loss = criterion(pred, y_batch)
        losses[i_batch] = loss.detach()

        if train:
            loss.backward()
            optimizer.step()

        writer.add_scalar(f'Batch/Loss/{name}', losses[i_batch], total_batch[int(not train)])
        pbar.set_postfix({'loss': loss.detach()})
        total_batch[int(not train)] += 1

    return losses


def train_regression_model(data_path, num_epochs=100, lr=1e-2, batch_size=512):
    # init tensorboard writer
    writer = SummaryWriter()
    total_batch = [0, 0]

    # init dataset and dataloader
    full_dataset = CustomDataset(data_path)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    ic(train_size, test_size)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # init the model
    model = LogisticRegression()
    optimizer = optim.SGD(model.parameters(), lr=lr)  # create an SGD optimizer for the model parameters
    criterion = nn.BCELoss()
    for epoch in range(num_epochs):
        print(f'epoch {epoch}:')
        train_losses = train_test_batch(model, train_loader, optimizer, criterion=criterion,
                                        train=True, writer=writer, total_batch=total_batch)

        test_losses = train_test_batch(model, test_loader, optimizer, criterion=criterion,
                                       train=False, writer=writer, total_batch=total_batch)

        avg_train_loss, avg_test_loss = train_losses.mean(), test_losses.mean()
        writer.add_scalar(f'Epoch/Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Epoch/Loss/val', avg_test_loss, epoch)

        print('\ntrain loss = {}\ttest loss = {}'.format(avg_train_loss, avg_test_loss))
        torch.save(model, f'{output_path}/model_{epoch}.pth')
    return model.eval()  # return trained model


if getattr(sys, 'gettrace', None):
    print('Debugging')
if __name__ == "__main__":
    train_regression_model(train_tsv)
