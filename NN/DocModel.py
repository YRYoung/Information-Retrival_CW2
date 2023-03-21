import os

import torch.nn as nn


class DocCNN(nn.Module):
    def __init__(self, config, input_shape=(2, 300), output_shape=1):
        super(DocCNN, self).__init__()

        self.config = config

        # Define the CNN layers

        self.activation = nn.LeakyReLU(.2) if config['CNN']['activation'] == 'leaky_relu' else nn.ReLU()

        self.cnn_units = nn.ModuleList()
        conv_input_shape = [input_shape[0]] + config['CNN']['layers']
        for i in range(len(config['CNN']['layers'])):

            self.cnn_units.append(nn.Conv1d(conv_input_shape[i], config['CNN']['layers'][i],
                                            kernel_size=config['CNN']['kernels'][i],
                                            stride=config['CNN']['stride'][i], padding='same'))
            if config['CNN']['batchNorm'][i] == 1:
                self.cnn_units.append(nn.BatchNorm1d(config['CNN']['layers'][i]))

            self.cnn_units.append(self.activation)

            if config['CNN']['maxPool'][i] == 1:
                self.cnn_units.append(nn.MaxPool1d(kernel_size=2))

        # Define the dense layers
        length = config['CNN']['layers'][-1] * input_shape[1] // sum(config['CNN']['maxPool']) ** 2

        self.dense_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(length, config['CNN']['denseUnit'][0]),
            self.activation,
            nn.Dropout(config['training']['dropRate'])
        )
        if config['CNN']['denseUnit'][0] != output_shape:
            self.dense_layers.append(nn.Linear(config['CNN']['denseUnit'][0], output_shape))

    def forward(self, x):
        for layer in self.cnn_units:

                x = layer(x)
        x = self.dense_layers(x)
        return x


if __name__ == '__main__':
    import yaml

    from torchsummary import summary

    from utils import map_location

    os.chdir('..')
    with open('./NN/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = DocCNN(config, (2, 300)).to(map_location)
    summary(model, (2, 300), device='cuda')
    from torchview import draw_graph
    #
    batch_size = 2
    model_graph = draw_graph(model, input_size=(batch_size, 2, 300), device=map_location, save_graph=True)
