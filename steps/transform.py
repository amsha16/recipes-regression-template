"""
This module defines the following routines used by the 'transform' step of the regression recipe:

- ``transformer_fn``: Defines customizable logic for transforming input data before it is passed
  to the estimator during model inference.
"""
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer


def transformer_fn():
    """
    Returns an *unfitted* transformer that defines ``fit()`` and ``transform()`` methods.
    The transformer's input and output signatures should be compatible with scikit-learn
    transformers.
    """
    import sklearn

    function_transformer_params = (
        {} if sklearn.__version__.startswith("1.0")
        else {"feature_names_out": "one-to-one"}
    )

    categorical_features = ['sex', 'smoker', 'region']
    
    categorical_transformer = Pipeline(
        steps = [('encoder_cat', OneHotEncoder(
            handle_unknown = 'ignore', drop='first',sparse_output=False))
        ]
    )

    return Pipeline(
        steps=[('encoder_ct', ColumnTransformer(
            transformers = [('cat', categorical_transformer,categorical_features)], 
            remainder = StandardScaler()))
        ]
    )
