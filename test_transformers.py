import unittest
import pandas as pd
import numpy as np
from pandas.util.testing import assert_frame_equal
from transformers import ClassifierProjectionFeature, Dummifier


class TestTranformers(unittest.TestCase):
    def test_classifier_projection_feature(self):
        """Test ClassifierProjectionFeature transformer"""
        # Given
        df = pd.DataFrame({'city': ['Paris'] * 6 + ['London'] * 4,
                           'sex': ['M', 'W'] * 5})
        target = pd.Series(['a', 'b', 'c', 'a', 'b'] * 2)
        expected_res = pd.DataFrame({'city': ['Paris'] * 6 + ['London'] * 4,
                                     'sex': ['M', 'W'] * 5,
                                     'city_target_response': [.5] * 6 + [.25] * 4,
                                     'city_count': [6] * 6 + [4] * 4})

        # When
        df_res = ClassifierProjectionFeature(projection_threshold=0.5) \
            .fit_transform(df, target)

        # Then
        self.assertTrue(df_res.sort_index(axis=1)
                        .equals(expected_res.sort_index(axis=1)))

    def test_dummifier(self):
        """Test Dummifier transformer"""

        # Given
        df = pd.DataFrame({"city": ["Paris", "London", "Paris", "Paris", "Madrid"],
                           "sex": ["M", "F", "M", "F", "F"]})
        df_2 = pd.DataFrame({"city": ["Madrid", "London", np.nan, np.nan, "Lisbonne"],
                           "sex": ["M", "F", "M", np.nan, "F"]})
        expected_res = pd.DataFrame({"city_Paris": [1, 0, 1, 1, 0],
                                     "city_London": [0, 1, 0, 0, 0],
                                     "city_Madrid": [0, 0, 0, 0, 1],
                                     "city_nan": [0] * 5,
                                     "sex_M": [1, 0, 1, 0, 0],
                                     "sex_F": [0, 1, 0, 1, 1],
                                     "sex_nan": [0] * 5})
        expected_res_2 = pd.DataFrame({"city_Paris": [0, 0, 0, 0, 0],
                                     "city_London": [0, 1, 0, 0, 0],
                                     "city_Madrid": [1, 0, 0, 0, 0],
                                     "city_nan": [0, 0, 1, 1, 0],
                                     "sex_M": [1, 0, 1, 0, 0],
                                     "sex_F": [0, 1, 0, 0, 1],
                                     "sex_nan": [0, 0, 0, 1, 0]})
        dummifier = Dummifier()

        # When
        df_res = dummifier.fit_transform(df)
        df_res_2 = dummifier.transform(df_2)

        # Then
        assert_frame_equal(df_res.sort_index(axis=1),
                           expected_res.sort_index(axis=1),
                           check_dtype=False)
        assert_frame_equal(df_res_2.sort_index(axis=1),
                           expected_res_2.sort_index(axis=1),
                           check_dtype=False)
