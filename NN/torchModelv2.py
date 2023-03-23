import os

import torch
import torch.nn as nn

from NN.DocModel import DocCNN


class PytorchCNN(nn.Module):
    def __init__(self, conf):
        super(PytorchCNN, self).__init__()
        self.docCNN = DocCNN(conf, input_shape=(1, 300),
                             output_shape=conf['CNN']['denseUnit'][0])

        self.config = conf

        if conf['training']['2CNN']:
            self.final_layers = DocCNN(conf, input_shape=(2, conf['CNN']['denseUnit'][0]),
                                       output_shape=1)
        else:
            self.final_layers = nn.Sequential(
                nn.Linear(conf['CNN']['denseUnit'][0], conf['CNN']['denseUnit'][1]),
                self.docCNN.activation,
                nn.Linear(conf['CNN']['denseUnit'][1], 1),
                nn.Softsign()
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
            query = query.repeat(1, passages_per_query, -1)
            rank = torch.concatenate([query, passage], dim=2).reshape(-1, 2, self.config['CNN']['denseUnit'][0])
        else:

            rank = (query - passage).reshape(-1, self.config['CNN']['denseUnit'][0])

        rank = self.final_layers(rank)

        return rank.reshape(batch_size, passages_per_query)

    # def predict(self,query,passage):


class MultiMarginRankingLoss(nn.Module):
    def __init__(self, config):
        super(MultiMarginRankingLoss, self).__init__()

        self._rankloss = nn.MarginRankingLoss()
        self.config = config
        if config['training']['mse']:
            self._mseloss = nn.BCEWithLogitsLoss()
            self.w = config['training']['mse']

    def forward(self, pred, y):
        num_q, num_p = y.shape
        loss = torch.zeros(num_q)
        for i in range(num_q):
            rlv_idx = torch.argwhere(y[i, ...] == 1)

            comparer = pred[i, rlv_idx].reshape(-1)
            for j in range(comparer.shape[0]):
                loss[i] += self._rankloss(comparer[j].repeat(num_p), pred[i], (y[i, ...] != 1).float())

        if self.config['training']['mse']:
            return - (1 - self.w) * loss.mean() + self.w * self._mseloss(pred, y)
        else:
            return - loss.mean()


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
