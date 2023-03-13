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
import sys

import pandas as pd
import torch
from icecream import ic
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from CustomDataset import CustomDataset

data_path = '../data'

output_path = f'{data_path}/temp1'
val_tsv = f'{data_path}/part2/validation_data.tsv'
train_tsv = f'{output_path}/train_data_del9.parquet.gzip'
sample_tsv = f'{data_path}/part2/sample_data.tsv'


class LogisticRegression(nn.Module):
    def __init__(self, input_dim=300):
        super(LogisticRegression, self).__init__()
        self.linear0 = nn.Linear(300, 1)
        self.linear1 = nn.Linear(2, 1)

    def forward(self, x):
        out = self.linear0(x).squeeze()
        out = self.linear1(out).squeeze()
        return torch.sigmoid(out)


def train_test_batch(model, dataloader, optimizer, criterion, train, writer, total_batch):
    name = 'train' if train else 'eval'
    losses = torch.zeros(len(dataloader)).to(device)
    model.train(train)
    pbar = tqdm(enumerate(dataloader),
                unit='batch', total=len(dataloader), desc=name)
    for i_batch, (x_batch, y_batch) in pbar:
        torch.cuda.empty_cache()
        y_batch = y_batch.to(device)
        x_batch.to(device)

        if train:
            optimizer.zero_grad()

        pred = model.forward(x_batch)
        loss = criterion(pred, y_batch)
        losses[i_batch] = loss.detach()

        if train:
            loss.backward()
            optimizer.step()

        writer.add_scalar(f'Batch/Loss/{name}', losses[i_batch], total_batch[int(not train)])
        # pbar.set_postfix({'loss': loss.detach()})
        total_batch[int(not train)] += 1

    return losses


def train_model(dataframe, num_epochs=10, lr=5e-3, batch_size=512, load_from=-1, logdir=None):
    # init dataset and dataloader
    full_dataset = CustomDataset(dataframe)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    ic(train_size, test_size)
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    loader_config = {"batch_size": batch_size, "shuffle": True, "num_workers": 4,
                     }
    train_loader = DataLoader(train_dataset, **loader_config)
    test_loader = DataLoader(test_dataset, **loader_config)

    # init the model
    model = LogisticRegression()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # init tensorboard writer
    total_batch = [0, 0]

    if load_from >= 0:
        logdir = load_model(model, optimizer, epoch=load_from, total_batch=total_batch)

    model.to(device)
    writer = SummaryWriter(logdir)

    criterion = nn.BCELoss()
    for epoch in range(load_from + 1, num_epochs):
        print(f'epoch {epoch}:')
        train_losses = train_test_batch(model, train_loader, optimizer, criterion=criterion,
                                        train=True, writer=writer, total_batch=total_batch)

        avg_train_loss = train_losses.mean()
        torch.cuda.empty_cache()

        writer.add_scalar(f'Epoch/Loss/train', avg_train_loss, epoch)
        print('\ntrain loss = {}'.format(avg_train_loss))

        if (epoch + 1) % 1 == 0:
            test_losses = train_test_batch(model, test_loader, optimizer, criterion=criterion,
                                           train=False, writer=writer, total_batch=total_batch)
            avg_test_loss = test_losses.mean()
            writer.add_scalar('Epoch/Loss/val', avg_test_loss, epoch)
            print('\nval loss = {}'.format(avg_test_loss))
            torch.cuda.empty_cache()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_batch': total_batch,
            'logdir': writer.log_dir,
        }, f'{output_path}/model_{epoch}.pth')
    return model.eval()  # return trained model


def eval_model(eval_dataframe, name, batch_size=512):
    # load model
    epoch = 3
    model = LogisticRegression()

    checkpoint = torch.load(f'{output_path}/model_{epoch}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()

    # init dataset and dataloader
    full_dataset = CustomDataset(eval_dataframe, name)
    loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)

    # eval
    pbar = tqdm(enumerate(loader), unit='batch', total=len(loader), desc=name)
    for i_batch, (x_batch, y_batch) in pbar:
        pred = model.forward(x_batch)

        # writer.add_scalar(f'Evaluation/{name}', losses[i_batch], total_batch[int(not train)])


def load_model(model, optimizer, epoch, total_batch):
    checkpoint = torch.load(f'{output_path}/model_{epoch}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    total_batch[:] = checkpoint['total_batch']
    return checkpoint['logdir']


if getattr(sys, 'gettrace', None):
    print('Debugging')
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'training in {device}')
    train_model(pd.read_parquet(train_tsv),
                logdir='runs/lr_5e_3')
