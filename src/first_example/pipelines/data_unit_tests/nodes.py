"""
This is a boilerplate pipeline
generated using Kedro 0.18.8
"""

import logging
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import great_expectations as ge

def unit_test(
    data: pd.DataFrame,
): 

    pd_df_ge = ge.from_pandas(data)

    assert pd_df_ge.expect_column_values_to_be_of_type("duration", "int64").success == True
    assert pd_df_ge.expect_column_values_to_be_of_type("marital", "str").success == True
    assert pd_df_ge.expect_table_column_count_to_equal(17).success == True

    log = logging.getLogger(__name__)
    log.info("Data passed on the unit data tests")

    return 0