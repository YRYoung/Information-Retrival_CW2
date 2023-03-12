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