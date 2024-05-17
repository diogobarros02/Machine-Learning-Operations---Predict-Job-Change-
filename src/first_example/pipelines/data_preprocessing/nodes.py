"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from .utils import *


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import sklearn
import mlflow


def clean_data(
    data: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """Does dome data cleaning.
    Args:
        data: Data containing features and target.
    Returns:
        data: Cleaned data
    """
    #remove some outliers
    df_transformed = data.copy()

    describe_to_dict = df_transformed.describe().to_dict()

    # for cols in ["age"]:
    #     Q1 = df_transformed[cols].quantile(0.25)
    #     Q3 = df_transformed[cols].quantile(0.75)
    #     IQR = Q3 - Q1     

    # filter = (df_transformed[cols] >= Q1 - 1.5 * IQR) & (df_transformed[cols] <= Q3 + 1.5 *IQR)
    # df_transformed = df_transformed.loc[filter]

    #df_transformed["age"] = np.nan
    df_transformed.fillna(-9999,inplace=True)

    describe_to_dict_verified = df_transformed.describe().to_dict()

    return df_transformed, describe_to_dict, describe_to_dict_verified 


def feature_engineer( data: pd.DataFrame) -> pd.DataFrame:
    
    le = LabelEncoder()

    df = campaign_(data)
    df = age_(df)
    df = balance_(df)
    if "y" in df.columns:
        df["y"] = df["y"].map({"no":0, "yes":1})
    
    #new profiling feature
    # In this step we should start to think on feature store
    df["mean_balance_bin_age"] = df.groupby("bin_age")["balance"].transform("mean")
    df["std_balance_bin_age"] = df.groupby("bin_age")["balance"].transform("std")
    df["z_score_bin_age"] = (df["mean_balance_bin_age"] - df["balance"])/(df["std_balance_bin_age"])
    #df['day_of_week'] = le.fit_transform(df['day_of_week'])
    df['month'] = le.fit_transform(df['month'])
    
    
    
    numerical_features = df.select_dtypes(exclude=['object','string','category']).columns.tolist()
    categorical_features = df.select_dtypes(include=['object','string','category']).columns.tolist()
    #Exercise create an assert for numerical and categorical features
    
    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    OH_cols= pd.DataFrame(OH_encoder.fit_transform(df[categorical_features]))

    # Adding column names to the encoded data set.
    OH_cols.columns = OH_encoder.get_feature_names_out(categorical_features)

    # One-hot encoding removed index; put it back
    OH_cols.index = df.index

    # Remove categorical columns (will replace with one-hot encoding)
    num_df = df.drop(categorical_features, axis=1)

    # Add one-hot encoded columns to numerical features
    df_final = pd.concat([num_df, OH_cols], axis=1)


    log = logging.getLogger(__name__)
    log.info(f"The final dataframe has {len(df_final.columns)} columns.")

    return df_final

