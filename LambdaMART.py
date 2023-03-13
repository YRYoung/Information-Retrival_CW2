"""
LambdaMART Model (LM) (25 marks)
-------------------------------------------------
https://www.analyticsvidhya.com/blog/2019/02/flair-nlp-library-python/
https://xgboost.readthedocs.io/en/stable/python/python_intro.html#data-interface
Use the LambdaMART learning to rank algorithm (a variant of LambdaRank we have learned in the class)
from XGBoost gradient boosting library to learn a model that can re-rank passages.

command XGBoost to use LambdaMART algorithm for ranking
by setting the appropriate value to the objective parameter as described in the documentation

carry out hyperparameter tuning in this task
--------------------------------------------------
Report:
    - describe the methodology used in deriving the best performing model.
    - report the performance of your model on the validation data with metrics from eval.py
    - Describe:
        1. how you perform input processing
        2. the representation/features used as input

"""

from huepy import *
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from icecream import ic
from sklearn.model_selection import train_test_split
from xgboost import Booster

from eval import eval_per_query, init_evaluator

from utils import train_embeddings_folder, val_raw_df, timeit, data_path


@timeit
def load_data(train_pth=f'{train_embeddings_folder}/train_{1}.pth'):
    data = torch.load(train_pth)
    x = data[0].detach().numpy()
    y = data[1].detach().numpy()

    # Splitting training set
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=0, test_size=0.1)

    x_train = x_train.reshape(-1, 600)
    x_valid = x_valid.reshape(-1, 600)

    # XGBoost compatible data
    return xgb.DMatrix(x_train, y_train), xgb.DMatrix(x_valid, label=y_valid)


@timeit
def train_model(dtrain, dvalid):
    # defining parameters
    params = {
        'eta': 0.1,
        'max_depth': 200,
        'objective': 'rank:ndcg',
        'sampling_method': 'gradient_based',
        #     'subsample': 0.9,
        'eval_metric': 'ndcg-'
    }

    # Training the model
    xgb_model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        maximize=True,
        evals=[(dvalid, 'val'), (dtrain, 'train')],
        early_stopping_rounds=30
    )
    return xgb_model


if __name__ == '__main__':
    xgb.config_context(verbosity=1)
    # dtrain, dvalid = load_data()
    # model = train_model(dtrain, dvalid)

    model = Booster()
    model.load_model(f"{data_path}/temp2/xgboost.model")
    init_evaluator(x_val_handler=lambda x: x.detach().numpy().reshape(-1, 600))(model)
