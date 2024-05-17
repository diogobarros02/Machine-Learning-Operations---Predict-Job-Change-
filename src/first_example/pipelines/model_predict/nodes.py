
import pandas as pd
import logging
from typing import Dict, Tuple, Any
import numpy as np  
import pickle
from ..data_preprocessing.nodes import clean_data, feature_engineer

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

logger = logging.getLogger(__name__)

def model_predict(X: pd.DataFrame,
                  model: pickle.Pickler) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Predict using the trained model.

    Args:
    --
        X (pd.DataFrame): Serving observations.
        model (pickle): Trained model.

    Returns:
    --
        scores (pd.DataFrame): Dataframe with new predictions.
    """

    # Predict
    y_pred = model.predict(X)

    # Create dataframe with predictions
    X['y_pred'] = y_pred
    
    # Create dictionary with predictions
    describe_servings = X.describe().to_dict()

    logger.info('Service predictions created.')
    logger.info('#servings: %s', len(y_pred))
    return X, describe_servings