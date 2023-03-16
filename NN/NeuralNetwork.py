"""
Neural Network Model (NN) (30 marks)

Using the same training data representation from the previous question
build a neural network based model that can re-rank passages.

You may use existing packages, namely Tensorflow or PyTorch in this subtask
Justify your choice by describing why you chose a particular architecture and how it fits to our problem.
You are allowed to use different types of neural network architectures
(e.g. feed forward, convolutional, recurrent and/or transformer based neural networks)

Using the metrics you have implemented in the first part,
report the performance of your model on the validation data.

Describe how you perform input processing, as well as the representation/features used.
Your marks for this part will depend on the appropriateness of the model you have chosen for the task,
as well as the representations/features used in training.

"""
import sys

import numpy as np
import pandas as pd
import torch
from huepy import *
from icecream import ic
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from CustomDataset import CustomDataset
from eval import eval_per_query, init_evaluator
from utils import data_path, val_raw_df, timeit

output_path = f'{data_path}/temp1'
val_tsv = f'{data_path}/part2/validation_data.tsv'
train_tsv = f'{output_path}/train_data_del9.parquet.gzip'
sample_tsv = f'{data_path}/part2/sample_data.tsv'


class CustomNetwork(nn.Module):
    def __init__(self, input_dim=300):
        super(CustomNetwork, self).__init__()
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
    train_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # init the model
    model = CustomNetwork()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # init tensorboard writer
    total_batch = [0, 0]

    if load_from >= 0:
        logdir = load_model(model, epoch=load_from, optimizer=optimizer, total_batch=total_batch)

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

            torch.cuda.empty_cache()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'total_batch': total_batch,
            'logdir': writer.log_dir,
        }, f'{output_path}/model_{epoch}.pth')
    return model.eval()  # return trained model


@timeit
def load_model(model, epoch: int, run_name: str, optimizer=None, total_batch=None):
    checkpoint = torch.load(f'{output_path}/{run_name}/model_{epoch}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if total_batch is not None:
        total_batch[:] = checkpoint['total_batch']
    return checkpoint['logdir']


if getattr(sys, 'gettrace', None):
    print('Debugging')
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'training in {device}')
    # train_model(pd.read_parquet(train_tsv), logdir='runs/lr_5e_3')
    at = [3, 10, 100]
    evaluator = init_evaluator(x_val_handler=
                               lambda x: x.detach().to(device).permute(0, 2, 1),
                               at=at)
    # '5e_3', '1e_3',
    for run_name in ['1e_2']:
        for epoch in range(7):
            model = CustomNetwork()
            logdir = load_model(model, epoch, run_name)
            model.to(device).eval()


            def predict_callback(x_val):
                return model.forward(x_val).cpu().detach().numpy()


            avg_precision, avg_ndcg = evaluator(predict_callback)
            writer = SummaryWriter(logdir)
            [writer.add_scalar(f'Epoch/mAP@{now}', value, epoch) for now, value in zip(at, avg_precision)]
            [writer.add_scalar(f'Epoch/NDCG@{now}', value, epoch) for now, value in zip(at, avg_ndcg)]
