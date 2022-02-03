import torcharrow.dtypes as dt
import torcharrow as ta
from typing import List
import unittest
import random
import pyarrow.parquet as pq
from torcharrow.functional import functional
import math
import numpy as np
import numpy.testing
import os

# Based on https://github.com/facebookresearch/torchrec/blob/9a8a4ad631bbccd7cd8166b7e6d7607e2560d2bd/torchrec/datasets/criteo.py#L37-L46
INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26
DEFAULT_LABEL_NAME = "label"
DEFAULT_INT_NAMES: List[str] = [f"int_{idx}" for idx in range(INT_FEATURE_COUNT)]
DEFAULT_CAT_NAMES: List[str] = [f"cat_{idx}" for idx in range(CAT_FEATURE_COUNT)]

DTYPE = dt.Struct(
    [
        dt.Field(DEFAULT_LABEL_NAME, dt.int8),
        dt.Field(
            "dense_features",
            dt.Struct(
                [
                    dt.Field(int_name, dt.Int32(nullable=True))
                    for int_name in DEFAULT_INT_NAMES
                ]
            ),
        ),
        dt.Field(
            "sparse_features",
            dt.Struct(
                [
                    dt.Field(cat_name, dt.Int32(nullable=True))
                    for cat_name in DEFAULT_CAT_NAMES
                ]
            ),
        ),
    ]
)

class CriteoIntegrationTest(unittest.TestCase):
    NUM_ROWS = 128
    RAW_ROWS = None
    TEMPORARY_PARQUETY_FILE = "_test_criteo.parquet"

    def setUp(self):
        # Generate some random data
        random.seed(42)

        rows = []
        for i in range(type(self).NUM_ROWS):
            label = 1 if random.randrange(100) < 4 else 0 
            dense_features_struct = tuple(
                    random.randrange(1000) if random.randrange(100) < 80 else None for j in range(INT_FEATURE_COUNT)
                )
            sparse_features_struct = tuple(
                    random.randrange(-2 ** 31, 2 ** 31) if random.randrange(100) < 98 else None for j in range(CAT_FEATURE_COUNT)
                )

            rows.append((label, dense_features_struct, sparse_features_struct))

        type(self).RAW_ROWS = rows
        df = ta.DataFrame(rows, dtype=DTYPE)

        pq.write_table(df.to_arrow(), type(self).TEMPORARY_PARQUETY_FILE)
    
    def tearDown(self) -> None:
        os.remove(type(self).TEMPORARY_PARQUETY_FILE)

    @staticmethod
    def preproc(df: ta.IDataFrame) -> ta.IDataFrame:
        # 1. fill null values
        df["dense_features"] = df["dense_features"].fill_null(0)
        df["sparse_features"] = df["sparse_features"].fill_null(0)

        # 2. apply log(x+3) on dense features
        df["dense_features"] = (df["dense_features"] + 3).log()

        # 3. Pack each categorical feature as an single element array
        sparse_features = df["sparse_features"]
        for field in DEFAULT_CAT_NAMES:
            sparse_features[field] = functional.array_constructor(sparse_features[field])

        # FIXME: we shouldn't need to "put back the struct column"
        df["sparse_features"] = sparse_features

        df["label"] = df["label"].cast(dt.int32)

        return df


    def test_criteo_transform(self):
        # Read data from Parquet file
        table = pq.read_table(type(self).TEMPORARY_PARQUETY_FILE)
        df = ta.from_arrow(table)

        self.assertEqual(df.dtype, DTYPE)
        self.assertEqual(list(df), type(self).RAW_ROWS)

        df = type(self).preproc(df)

        # Check result
        self.assertEqual(df["label"].dtype, dt.int32)

        expected_labels = []
        expected_dense_features = []
        expected_sparse_features = []
        for (label, dense_features, sparse_features) in type(self).RAW_ROWS:
            expected_labels.append(label)
            expected_dense_features.append(tuple(np.log(np.array([v or 0 for v in dense_features], dtype=np.float32) + 3)))
            expected_sparse_features.append(tuple(
                [v or 0] for v in sparse_features
            ))

        self.assertEqual(list(df["label"]), expected_labels)
        numpy.testing.assert_array_almost_equal(
            np.array(list(df["dense_features"])), np.array(expected_dense_features)
        )
        self.assertEqual(list(df["sparse_features"]), expected_sparse_features)

        # TODO: do to_tensor and test the result


if __name__ == "__main__":
    unittest.main()
