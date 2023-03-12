"""
LambdaMART Model (LM) (25 marks)
-------------------------------------------------
https://www.analyticsvidhya.com/blog/2019/02/flair-nlp-library-python/
https://xgboost.readthedocs.io/en/stable/python/python_intro.html#data-interface
Use the LambdaMART learning to rank algorithm (a variant of LambdaRank we have learned in the class)
from XGBoost gradient boosting library to learn a model that can re-rank passages.

command XGBoost to use LambdaMART algorithm for ranking
by setting the appropriate value to the objective parameter as described in the documentation

carry out hyper-parameter tuning in this task
--------------------------------------------------
Report:
    - describe the methodology used in deriving the best performing model.
    - report the performance of your model on the validation data with metrics from eval.py
    - Describe:
        1. how you perform input processing
        2. the representation/features used as input

"""

import xgboost as xgb

### Splitting training set ###
# x_train, x_valid, y_train, y_valid = train_test_split(train, target,
#                                                       random_state=42,
#                                                           test_size=0.3)

def custom_eval(preds, dtrain):
    labels = dtrain.get_label().astype(np.int)
    preds = (preds >= 0.3).astype(np.int)
    return [('f1_score', f1_score(labels, preds))]

### XGBoost compatible data ###
dtrain = xgb.DMatrix(x_train,y_train)
dvalid = xgb.DMatrix(x_valid, label = y_valid)

### defining parameters ###
params = {
          'colsample': 0.9,
          'colsample_bytree': 0.5,
          'eta': 0.1,
          'max_depth': 8,
          'min_child_weight': 6,
          'objective': 'binary:logistic',
          'subsample': 0.9
          }

### Training the model ###
xgb_model = xgb.train(
                      params,
                      dtrain,
                      feval= custom_eval,
                      num_boost_round= 1000,
                      maximize=True,
                      evals=[(dvalid, "Validation")],
                      early_stopping_rounds=30
                      )