import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES
from src.data import _add_target_encoding


def test_target_encoding_is_out_of_fold_and_multiclass():
    row_count = 60
    train = pd.DataFrame(
        {
            column: [f"{column}_{index}" for index in range(row_count)]
            for column in CATEGORICAL_FEATURES
        }
    )
    test = pd.DataFrame(
        {column: [f"unseen_{column}"] for column in CATEGORICAL_FEATURES}
    )
    labels = pd.Series(np.tile([0, 1, 2], row_count // 3))

    encoded_train, encoded_test = _add_target_encoding(
        train.copy(), test.copy(), labels
    )

    encoded_columns = [
        column
        for column in encoded_train.columns
        if "_te_" in column
    ]
    assert len(encoded_columns) == len(CATEGORICAL_FEATURES) * 3
    assert not encoded_train[encoded_columns].isna().any().any()
    assert not encoded_test[encoded_columns].isna().any().any()
    # Every category is unique, so an OOF row must fall back to the class prior.
    assert np.allclose(encoded_train[encoded_columns].to_numpy(), 1 / 3)
