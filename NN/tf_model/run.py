import os
import pickle
import sys

import pandas as pd
import yaml

from NN.tf_model import NeuralNetwork
from LogisticRegression import DataLoader
from eval import init_evaluator
from utils import data_path, train_debug_df, load_passages_tensors

output_path = f'{data_path}/temp1'
val_tsv = f'{data_path}/part2/validation_data.tsv'
train_tsv = f'{output_path}/train_data_del9.parquet.gzip'
sample_tsv = f'{data_path}/part2/sample_data.tsv'

epochs = 300
batch_size = 64


def main_train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    with open('./NN/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # set up variables
    checkpoint_dir = config['general']['checkpoint_dir']
    seed = config['general']['seed']

    os.makedirs(checkpoint_dir, exist_ok=True)

    # initialise save file
    # f = h5py.File(checkpoint_dir + "result_file.hdf5", "w")

    # init evaluate
    evaluator = init_evaluator(at=[3, 10, 100], prepare_x=False)

    with open('./data/val_embeddings_np.pkl', 'rb') as f:
        x_val, y_val = pickle.load(f)

    # init train
    x_df = pd.read_parquet(train_debug_df)
    dataloader = DataLoader(x_df, len(x_df), load_passages_tensors())
    _, (train_x, train_y) = [(x, y) for x, y in enumerate(dataloader)][0]

    for cv in range(config['training']['cv']):
        seed += cv

        CNN = NeuralNetwork.Network(config, evaluator)

        if config['training']['train']:
            CNN.train_model(X_train=train_x,
                            y_train=train_y,
                            X_val=x_val,
                            y_val=y_val,
                            epochs=epochs,
                            batch_size=batch_size,
                            checkpoint_dir=checkpoint_dir,
                            cv_order=cv)

        else:
            CNN.compile_model(checkpoint_dir=checkpoint_dir, cv=cv)


if __name__ == '__main__':
    sys.path.append('./cw2/NN')
    os.chdir('../..')
    main_train()
