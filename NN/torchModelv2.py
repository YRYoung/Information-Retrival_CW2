import os

import torch
import torch.nn as nn
from icecream import ic

from NN.DocModel import DocCNN
from utils import map_location


class PytorchCNN(nn.Module):
    def __init__(self, conf):
        super(PytorchCNN, self).__init__()
        self.config = conf
        self.cnn_dense_unit_ = self.config['CNN']['denseUnit']

        self.docCNN = DocCNN(conf, input_shape=(1, 300),
                             output_shape=self.cnn_dense_unit_[0])

        if conf['training']['2CNN']:
            self.final_layers = DocCNN(conf, input_shape=(2, self.cnn_dense_unit_[0]),
                                       output_shape=1)
        else:
            self.final_layers = nn.Sequential(
                nn.Linear(self.cnn_dense_unit_[0], self.cnn_dense_unit_[1]),
                self.docCNN.activation,
                nn.Linear(self.cnn_dense_unit_[1], 1),
                nn.Sigmoid()
            )

    def allto(self, *args, **kwargs):
        self.to(*args, **kwargs)
        self.docCNN.to(*args, **kwargs)
        return self

    def forward(self, query, passage):
        batch_size, passages_per_query, _ = passage.shape
        passage = passage.reshape(-1, 300)
        query = query.reshape(-1, 300)
        feature = torch.cat([query, passage], dim=0).reshape(-1, 1, 300)
        feature = self.docCNN.forward(feature)

        query = feature[:batch_size, ...].reshape(batch_size, 1, -1)
        passage = feature[batch_size:, ...].reshape(batch_size, passages_per_query, -1)

        if self.config['training']['2CNN']:
            query = query.repeat(1, passages_per_query, 1)
            result = torch.concatenate([query, passage], dim=2).reshape(-1, 2, self.cnn_dense_unit_[0])
            result = self.final_layers(result)

    
        else:
            result = (query - passage).reshape(-1, self.cnn_dense_unit_[0])
            result = self.final_layers(result)

        rank = self.final_layers(rank)

        return rank.reshape(batch_size, passages_per_query)

    # def predict(self,query,passage):


class MultiMarginRankingLoss(nn.Module):
    def __init__(self, config):
        super(MultiMarginRankingLoss, self).__init__()

        self._rankloss = nn.MarginRankingLoss()
        self.config = config
        if config['training']['bce']:
            self._bceloss = nn.BCELoss()
            self.bce_w = config['training']['bce']

    def list_mle(self, y_pred, y_true, k=None):
        # y_pred : batch x n_items
        # y_true : batch x n_items
        if k is not None:
            sublist_indices = (y_pred.shape[1] * torch.rand(size=k)).long()
            y_pred = y_pred[:, sublist_indices]
            y_true = y_true[:, sublist_indices]

        _, indices = y_true.sort(descending=True, dim=-1)

        pred_sorted_by_true = y_pred.gather(dim=1, index=indices)

        cumsums = pred_sorted_by_true.exp().flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

        listmle_loss = torch.log(cumsums + 1e-10) - pred_sorted_by_true

        return listmle_loss.sum(dim=1).mean()

    def forward(self, pred, y):
        num_q, num_p = y.shape
        loss = torch.zeros(num_q)
        for i in range(num_q):
            rlv_idx = torch.argwhere(y[i, ...] == 1)
            comparer = pred[i, rlv_idx].reshape(-1)

            for j in range(comparer.shape[0]):
                loss[i] += self._rankloss(comparer[j].repeat(num_p), pred[i], (y[i, ...] == 1).float())
        result = 1 - loss.mean()

        if self.config['training']['bce']:
            result = (1 - self.bce_w) * result + self.bce_w * self._bceloss(pred, y)
        return result


if __name__ == '__main__':
    os.chdir('..')

    import yaml

    from torchsummary import summary

    with open('./NN/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    # input = (1, 300)
    model = PytorchCNN(config).allto('cuda')
    summary(model, [(1, 300), (5, 300)], device='cuda')
    from torchview import draw_graph

    bs = 2
    model_graph = draw_graph(model, input_size=[(1, 1, 300), (1, 15, 300)], device='cuda', save_graph=True)
