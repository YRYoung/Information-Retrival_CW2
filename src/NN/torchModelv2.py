import os

import torch
import torch.nn as nn

from src.NN.DocModel import DocCNN
from src.utils import map_location


class PytorchCNN(nn.Module):
    def __init__(self, conf):
        super(PytorchCNN, self).__init__()
        self.config = conf
        self.softmax = nn.Softmax(dim=2)
        self.cnn_dense_unit_ = self.config['CNN']['denseUnit']

        self.docCNN = DocCNN(conf, input_shape=(1, 300),
                             output_shape=self.cnn_dense_unit_[0])

        if conf['training']['2CNN']:
            self.final_layers = DocCNN(conf, input_shape=(1, self.cnn_dense_unit_[0]),
                                       output_shape=1)
        else:
            self.final_layers = nn.Sequential(
                nn.Linear(self.cnn_dense_unit_[0], self.cnn_dense_unit_[1]),
                self.docCNN.activation,
                nn.Linear(self.cnn_dense_unit_[1], 1),
                nn.Sigmoid()
            )

        self.query_coder = nn.Sequential(
            nn.Linear(300, self.config['Query_Sequential'][0]),
            self.docCNN.activation,
            nn.Linear(self.config['Query_Sequential'][0], self.config['Query_Sequential'][1]),
            self.docCNN.activation,
            nn.Linear(self.config['Query_Sequential'][1], self.cnn_dense_unit_[0]),
            nn.Sigmoid(),
        )

        if self.config['training']['attention'][0]:
            d_inner_hid = 512
            self.w_1 = nn.Conv1d(300, d_inner_hid, 1)
            self.w_2 = nn.Conv1d(d_inner_hid, 300, 1)
            self.layer_norm = nn.LayerNorm(300)
            self.relu = nn.ReLU()

    def forward(self, query, passage):
        batch_size, passages_per_query, _ = passage.shape
        if self.config['training']['attention'][0]:
            attention = torch.bmm(passage, passage.transpose(1, 2)) / (self.cnn_dense_unit_[0] ** 0.5)
            attention = self.softmax(attention)
            result = torch.bmm(attention.transpose(1, 2), passage)
            # add & norm
            result = self.layer_norm(passage + result)

            # feed forward
            output = self.relu(self.w_1(result.transpose(1, 2)))
            output = self.w_2(output).transpose(2, 1)
            output = nn.Dropout()(output)
            passage = self.layer_norm(passage + output)

        passage = passage.reshape(-1, 1, 300)
        query = query.reshape(-1, 300)

        # feature = torch.cat([query, passage], dim=0).reshape(-1, 1, 300)

        passage = self.docCNN.forward(passage).reshape(batch_size, passages_per_query, -1)

        query = self.query_coder(query).reshape(batch_size, 1, -1)
        # passage = feature[batch_size:, ...].reshape(batch_size, passages_per_query, -1)

        result = (query - passage) if self.config['training']['cat_qp'] == 'diff' else (query * passage)

        # if self.config['training']['attention'][1]:
        #     attention = torch.bmm(passage, passage.transpose(1, 2)) / (self.cnn_dense_unit_[0] ** 0.5)
        #     attention = self.softmax(attention)
        #     result = torch.bmm(attention, result)

        result_shape = (-1, 1, self.cnn_dense_unit_[0]) if self.config['training']['2CNN'] else (
            -1, self.cnn_dense_unit_[0])
        result = self.final_layers(result.reshape(result_shape))

        return result.reshape(batch_size, passages_per_query)

    # def predict(self,query,passage):


class MultiMarginRankingLoss(nn.Module):
    def __init__(self, config):
        super(MultiMarginRankingLoss, self).__init__()

        self._rankloss = nn.MarginRankingLoss(reduction='sum')
        self.config = config
        if config['training']['bce']:
            self._bceloss = nn.BCELoss()
            self.bce_w = config['training']['bce']

    def forward(self, pred, y):
        num_q, num_p = y.shape
        loss = torch.zeros(num_q, device=map_location).float()
        for i in range(num_q):
            rlv_idx = torch.argwhere(y[i, ...] == 1)
            comparer = pred[i, rlv_idx].reshape(-1)

            for j in range(comparer.shape[0]):
                loss[i] += self._rankloss(comparer[j].repeat(num_p), pred[i], (y[i, ...] != 1).float())
        result = loss.mean()

        if self.config['training']['bce']:
            result = (1 - self.bce_w) * result + self.bce_w * self._bceloss(pred, y)
        return result


if __name__ == '__main__':
    os.chdir('../..')

    import yaml

    from torchsummary import summary

    with open('./NN/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)[0]
    # input = (1, 300)
    model = PytorchCNN(config).to('cuda')
    summary(model, [(1, 300), (5, 300)], device='cuda')
    from torchview import draw_graph

    bs = 2
    model_graph = draw_graph(model, input_size=[(1, 1, 300), (1, 15, 300)], device='cuda', save_graph=True)
