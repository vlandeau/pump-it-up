import time
from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np
from sklearn.ensemble.iforest import IsolationForest
from scipy import stats

VERBOSE = False


def _log_time(func):
    def func_with_logs(*args, **kwargs):
        if VERBOSE:
            func_name = func.__name__
            class_name = args[0].__class__.__name__
            start_time = time.time()
            print "Starting method %s of class %s" \
                  % (func_name, class_name)
        res = func(*args, **kwargs)
        if VERBOSE:
            elapsed_time = time.time() - start_time
            print "Method %s of class %s took %s seconds to perform" \
                  % (func_name, class_name, elapsed_time)
        return res

    return func_with_logs


class TimeStampConverter(TransformerMixin, BaseEstimator):
    def __init__(self, cols_to_convert, date_format='%Y-%m-%d'):
        self.cols_to_convert = cols_to_convert
        self.date_format = date_format

    def fit(self, df, y=None):
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        for col in self.cols_to_convert:
            df_copy[col] = pd.to_datetime(df_copy[col], format=self.date_format)
        return df_copy


class TypeConverter(TransformerMixin, BaseEstimator):
    def __init__(self, cols_to_convert, final_type=object):
        self.cols_to_convert = cols_to_convert
        self.final_type = final_type

    def fit(self, df, y=None):
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        for col in self.cols_to_convert:
            df_copy[col] = df_copy[col].astype(self.final_type)
        return df_copy


class RareValuesRemover(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.01, default_value="OTHER", cols=None):
        self.threshold = threshold
        self.default_value = default_value
        self.cols = cols

    def fit(self, df, y=None):
        if self.cols is None:
            self.cols = df.select_dtypes(include=["object"]).columns
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        for col in self.cols:
            counts = df_copy[col].value_counts(normalize=True)
            vals_to_drop = counts[counts < self.threshold].index.tolist()
            df_copy.loc[df_copy[col].isin(vals_to_drop), col] = self.default_value
        return df_copy


class Dummifier(TransformerMixin, BaseEstimator):
    def __init__(self, columns_to_dummify=None):
        self.columns_to_dummify = columns_to_dummify

    def fit(self, df, y=None):
        self.dummified_columns = []
        if self.columns_to_dummify is None:
            self.columns_to_dummify = df.select_dtypes(include=[object, "category"]) \
                .columns
        for col in self.columns_to_dummify:
            self.dummified_columns += map(lambda x: col + '_' + str(x),
                                          df[col].unique().tolist() + ['nan'])
        return self

    @_log_time
    def transform(self, df):
        dummified_columns_res = []
        for col in self.columns_to_dummify:
            dummified_columns_res += map(lambda x: col + '_' + str(x),
                                         df[col].unique().tolist())
        df_res = pd.get_dummies(df,
                                columns=self.columns_to_dummify,
                                dummy_na=True)
        for col in self.dummified_columns:
            if col not in df_res.columns:
                df_res[col] = 0
        return df_res.drop(axis=1,
                           labels=set(dummified_columns_res) - set(self.dummified_columns))


class DateInfoGetter(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.date_cols = []
        self.dates_min = {}

    def fit(self, df, y=None):
        self.date_cols = df.select_dtypes(include=[np.dtype("datetime64[ns]")]).columns
        for col in self.date_cols:
            self.dates_min[col] = df[col].min()
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        for col in self.date_cols:
            df_copy[col + "_weekday"] = df_copy[col].apply(lambda x: str(x.weekday()))
            df_copy[col + "_month"] = df_copy[col].apply(lambda x: str(x.month))
            df_copy[col + "_year"] = df_copy[col].apply(lambda x: str(x.year))
            df_copy[col + "_since_beginning"] = ((df_copy[col] - self.dates_min[col])
                                                 .apply(lambda x: x.days))
            del df_copy[col]
        return df_copy


class PandasDfToNpArrayConverter(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.features = []

    def fit(self, df, y=None):
        self.features = df.sort_index(axis=1).columns
        return self

    @_log_time
    def transform(self, df):
        return df.sort_index(axis=1).as_matrix()


class ClassifierProjectionFeature(TransformerMixin, BaseEstimator):
    def __init__(self, projection_threshold=0.01):
        self.projection_threshold = projection_threshold
        self.target = None
        self.projection_dfs = None
        self.columns_to_project = []

    @_log_time
    def fit(self, df, y=None):
        self.target = y.copy()
        mask_mode = (self.target == stats.mode(self.target)[0][0])
        self.target[mask_mode] = 1
        self.target[~mask_mode] = 0
        self.target = self.target.astype(np.int64)
        self.columns_to_project = self._categorical_features_with_rare_modalities(df)
        projection_target = 'projection_target'
        df[projection_target] = self.target
        self.projection_dfs = {}
        for col in self.columns_to_project:
            projection_df = pd.DataFrame()
            projection_df[col + '_target_response'] = df[[projection_target, col]] \
                .groupby(col).mean()['projection_target']
            projection_df[col + '_count'] = df[[projection_target, col]] \
                .groupby(col).count()['projection_target']
            self.projection_dfs[col] = projection_df.reset_index(level=0)
        df.drop(projection_target, axis=1, inplace=True)
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        for col in self.columns_to_project:
            df_copy = df_copy.merge(self.projection_dfs[col], on=col, how='left')
        return df_copy

    def _categorical_features_with_rare_modalities(self, df):
        categorical_features_with_rare_modalities = []
        for col in df.select_dtypes(include=[object]).columns:
            counts = df[col].value_counts(normalize=True)
            if (counts < self.projection_threshold).sum() > 0:
                categorical_features_with_rare_modalities.append(col)
        return categorical_features_with_rare_modalities


class ProjectionFeatureAdder(TransformerMixin, BaseEstimator):
    def __init__(self, target_series=None, projection_threshold=0.01):
        self.target_series = target_series
        self.projection_threshold = projection_threshold
        self.columns_to_project = []
        self.target_modes = []
        self.projection_values = dict()

    @_log_time
    def fit(self, df, y=None):
        self.columns_to_project = self._categorical_features_with_rare_modalities(df)
        print self.columns_to_project
        self.target_modes = self.target_series.drop_duplicates()
        df_copy = df.copy()
        df_copy['target'] = self.target_series
        len_df = len(df_copy)
        for col in self.columns_to_project:
            for target_mode in self.target_modes:
                dict_values = dict()
                for value in df[col].unique():
                    dict_values[value] = len(df_copy[(df_copy[col] == value)
                                                     & (df_copy['target'] == target_mode)]) / len_df
                self.projection_values[col + '_' + target_mode + '_projection'] = dict_values
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        for col in self.columns_to_project:
            for target_mode in self.target_modes:
                projection_col = col + '_' + target_mode + '_projection'
                df_copy[projection_col] = df_copy[col]
        df_copy[projection_col].replace(self.projection_values, inplace=True)
        return df_copy

    def _categorical_features_with_rare_modalities(self, df):
        categorical_features_with_rare_modalities = []
        for col in df.select_dtypes(include=[object]).columns:
            counts = df[col].value_counts(normalize=True)
            if len(counts[counts < self.projection_threshold]) > 0:
                categorical_features_with_rare_modalities.append(col)
        return categorical_features_with_rare_modalities


class ReplaceNegativeByNan(TransformerMixin, BaseEstimator):
    def __init__(self, columns_to_transform=[], filter_function=(lambda x: x <= 0)):
        self.columns_to_transform = columns_to_transform
        self.filter_function = filter_function

    def fit(self, df, y=None):
        self.columns_to_transform = self._filter_existing_columns(self.columns_to_transform, df)
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        for col in self.columns_to_transform:
            df_copy.loc[df_copy[col].apply(self.filter_function), col] = np.nan
        return df_copy

    def _filter_existing_columns(self, column_list, df):
        intersection = set(column_list).intersection(set(df.columns))
        return list(intersection)


class NanIndicatorAdder(TransformerMixin, BaseEstimator):
    def __init__(self, columns_to_transform=[]):
        self.columns_to_transform = columns_to_transform

    def fit(self, df, y=None):
        columns_with_na = df.columns[df.isnull().any()].tolist()
        if len(self.columns_to_transform) > 0:
            self.columns_to_transform = list(set(self.columns_to_transform).intersection(set(columns_with_na)))
        else:
            self.columns_to_transform = columns_with_na
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        for col in self.columns_to_transform:
            nan_indicator_col = col + "_with_nan"
            df_copy[nan_indicator_col] = 0
            mask_nan = df_copy[col].isnull()
            df_copy.loc[mask_nan, nan_indicator_col] = 1
        return df_copy


class CategoricalFeatureImputer(TransformerMixin, BaseEstimator):
    def __init__(self):
        self.categorical_columns = []

    def fit(self, df, y=None):
        self.categorical_columns = df.select_dtypes(include=[object]).columns
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        for col in self.categorical_columns:
            df_copy[col].fillna("UNKNOWN", inplace=True)
        return df_copy


class NumericalFeatureImputation(TransformerMixin, BaseEstimator):
    def __init__(self, strategy="median"):
        self.strategy = strategy
        self.numerical_columns = []

    def fit(self, df, y=None):
        self.numerical_columns = df.select_dtypes(include=[int, float]).columns
        return self

    @_log_time
    def transform(self, df):
        df_copy = df.copy()
        if self.strategy == "median":
            fill_na_dict = {c: df_copy[c].median() for c in self.numerical_columns}
        elif self.strategy == "mean":
            fill_na_dict = {c: df_copy[c].median() for c in self.numerical_columns}
        else:
            raise Exception("Unknown strategy " + self.strategy)
        df_copy.fillna(fill_na_dict, inplace=True)
        return df_copy


class OutlierRemoval(TransformerMixin, BaseEstimator):
    def __init__(self, n_estimators=150, n_jobs=4, anomaly_threshold=0.5):
        self.isolation_forest = IsolationForest(n_estimators=n_estimators,
                                                n_jobs=n_jobs)
        self.anomaly_threshold = anomaly_threshold
        self.target = None
        self.outlier_mask = None

    @_log_time
    def fit(self, df, y=None):
        self.isolation_forest.fit(df)
        return self

    @_log_time
    def transform(self, df):
        self.outlier_mask = self.isolation_forest.predict(df) < self.anomaly_threshold
        return df.loc[self.outlier_mask, :]
