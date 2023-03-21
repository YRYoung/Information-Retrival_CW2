import os
import sys

import pandas as pd
import torch
import yaml
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from CustomDataset import ValidationDataset
from NN.torchModelv2 import PytorchCNN, MultiMarginRankingLoss
from utils import data_path, train_raw_df, map_location

models_save_dir = f'{data_path}/models'
log_root_dir = f'{data_path}/logs'


def collate(batch):
    x, y = zip(*batch)
    q_x, p_x = zip(*x)
    p_x = torch.stack(p_x, dim=0)
    q_x = torch.cat(q_x, dim=0)
    y = torch.stack(y, dim=0)
    return (q_x, p_x), y


def train_epoch(model, dataloader, optimizer, criterion, writer, total_batch):
    losses = torch.zeros(len(dataloader)).to(map_location)
    model.train()
    pbar = tqdm(enumerate(dataloader),
                unit='batch', total=len(dataloader))
    for i_batch, (x_batch, y_batch) in pbar:
        torch.cuda.empty_cache()

        y_batch = y_batch.view(-1, 1)
        x_batch = x_batch.view(-1, 2, 300)

        y_batch = y_batch.to(map_location, non_blocking=True)
        x_batch = x_batch.to(map_location, non_blocking=True)

        optimizer.zero_grad()

        pred = model.forward(x_batch)
        # ic(pred)
        loss = criterion(pred, y_batch)
        losses[i_batch] = loss.detach()

        loss.backward()
        optimizer.step()

        writer.add_scalar(f'Batch/Loss/train', losses[i_batch], total_batch[0])
        pbar.set_postfix({'loss': loss.detach()})
        total_batch[0] += 1

    return losses


def train_epoch_2(model, dataloader, optimizer, criterion, writer, total_batch):
    model.train()
    pbar = tqdm(enumerate(dataloader),
                unit='batch', total=len(dataloader))
    loss = torch.nan
    for i_batch, ((query, passage), y) in pbar:
        torch.cuda.empty_cache()

        query = query.to(map_location, non_blocking=True)
        passage = passage.to(map_location, non_blocking=True)
        y = y.to(map_location, non_blocking=True)

        optimizer.zero_grad()

        pred = model.forward(query, passage)
        loss = criterion(pred, y)

        loss.backward()
        optimizer.step()

        writer.add_scalar(f'Batch/Loss/train', loss.detach(), total_batch[0])
        pbar.set_postfix({'loss': loss.detach()})
        total_batch[0] += 1

    return loss


def train_model(config, model, optimizer, total_batch, loader, num_epochs=10, evaluator=None, val_freq=1000, save_freq=5):
    writer = SummaryWriter(config['general']['logDir'])
    writer.add_text('learning rate', f"{config['training']['learning_rate']}")

    criterion = MultiMarginRankingLoss()

    for epoch in range(config['training']['init_epoch'] + 1, num_epochs):
        print(f'epoch {epoch}:')
        train_losses = train_epoch_2(model, loader, optimizer, criterion=criterion,
                                     writer=writer, total_batch=total_batch)

        epoch_loss = train_losses.mean()
        torch.cuda.empty_cache()

        writer.add_scalar(f'Epoch/Loss/train', epoch_loss, epoch)
        print('\ntrain loss = {}'.format(epoch_loss))

        if (epoch + 1) % val_freq == 0:
            model.eval()
            avg_precision, avg_ndcg = evaluator(model.forward)

            _at = ['@3', '@10', '@100']
            [writer.add_scalars(f'Epoch/Val/mAP{s}', value) for s, value in zip(_at, avg_precision)]
            [writer.add_scalars(f'Epoch/Val/NDCG{s}', value) for s, value in zip(_at, avg_ndcg)]

            torch.cuda.empty_cache()

        if (epoch + 1) % save_freq == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'total_batch': total_batch,
            }, f"{model_file_name(epoch, config)}.pth")
    return model.eval()  # return trained model


def init_config():
    with open('./NN/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # seed = config['general']['seed']

    os.makedirs(models_save_dir, exist_ok=True)
    os.makedirs(log_root_dir, exist_ok=True)

    config['general']['checkpointsDir'] = f"{models_save_dir}/{config['training']['cv']}"
    config['general']['logDir'] = f"{log_root_dir}/{config['training']['cv']}"

    os.makedirs(config['general']['checkpointsDir'], exist_ok=True)
    os.makedirs(config['general']['logDir'], exist_ok=True)

    return config


def load_model(config):
    # init the model
    model = PytorchCNN(conf=config)
    train_config = config['training']
    optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'])
    total_batch = [0, 0]

    if train_config['init_epoch'] >= 0:
        checkpoint = torch.load(
            f"{model_file_name(train_config['init_epoch'], config)}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        total_batch[:] = checkpoint['total_batch']

    model.to(map_location)

    return model, optimizer, total_batch


def model_file_name(epoch, config):
    return f"{config['general']['checkpointsDir']}/model_{epoch}"


if getattr(sys, 'gettrace', None):
    print('Debugging')
if __name__ == "__main__":
    print(f'training in {map_location}')

    config = init_config()

    df = pd.read_parquet(train_raw_df)

    full_dataset = ValidationDataset(all_dataframe=df,
                                     return_tensors='tuple', fake_tensor=True,
                                     val_p_tensors=None, queries_tensors=None,
                                     passages_per_query=config['training']['passages_per_query'])

    dataloader = DataLoader(full_dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=True, num_workers=0,
                            pin_memory=False, collate_fn=collate)

    model, optimizer, total_batch = load_model(config)
    # evaluator = init_evaluator(x_val_handler=lambda x: x.reshape(-1, 2, 300).to(map_location),
    #                            at=[3, 10, 100])

    train_model(config=config, model=model, optimizer=optimizer, total_batch=total_batch, loader=dataloader,
                num_epochs=10, evaluator=None)