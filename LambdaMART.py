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
import pandas as pd
import torch
import xgboost as xgb
from sklearn import model_selection
from xgboost import Booster

from LogisticRegression import DataLoader
from eval import init_evaluator
from utils import timeit, data_path, queries_embeddings, train_raw_df, load_passages_tensors, train_debug_df, val_raw_df


if __name__ == '__main__':
    fixed_params = {
        # Number of gradient boosted trees. Equivalent to number of boosting rounds.

        'objective': 'rank:ndcg',
        'sampling_method': 'gradient_based',  # Used only by gpu_hist tree method.
        'eval_metric': 'ndcg-',
        'tree_method': 'gpu_hist',

    }

    param_dicts = {

        'max_depth': [7, 8, 9],
        'learning_rate ': [0.01, 0.05],
        'n_estimators': [5, 6, 7, 8, 9],
        'booster': ['gbtree', 'dart'],
        'gamma': [0, 1, 2, 3]

    }.update(fixed_params)

    x_df = pd.read_parquet(train_debug_df)
    dataloader = DataLoader(x_df, len(x_df), load_passages_tensors())
    _,(train_x, train_y )= [(x, y) for x, y in enumerate(dataloader)][0]
    del dataloader
    val_df = pd.read_parquet(val_raw_df)
    del val_df['query']
    del val_df['passage']
    del val_df['pid']
    del val_df['p_idx']
    x_val = torch.load('./data/val_embeddings.pth')[0]

    model = xgb.XGBRanker(fixed_params)
    clf = model_selection.GridSearchCV(model, param_dicts, verbose=1,
                                       n_jobs=2)

    clf.fit(X=train_x, y=train_y, qid=dataloader.df.qid,
            eval_set=(x_val, val_df['relevancy']),
            eval_qid=val_df.qid)
    print(clf.best_score_)
    print(clf.best_params_)

    # model = Booster()
    # model.load_model(f"{data_path}/temp2/xgboost.model")
    init_evaluator(x_val_handler=lambda x: x.detach().numpy().reshape(-1, 600))(model)
