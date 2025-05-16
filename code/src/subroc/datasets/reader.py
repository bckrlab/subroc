import pandas as pd
import numpy as np
from enum import Enum

from subroc.datasets.metadata import DatasetName, meta_dict
from subroc.quality_functions.base_qf import PredictionType


class DatasetStage(Enum):
    RAW = 0
    PROCESSED_PYTHON = 1
    PROCESSED_RAPIDMINER = 2
    PROCESSED_MODEL_READY = 3
    PROCESSED_MODEL_PREDICTED = 4
    PROCESSED_PERMUTED_MODEL_PREDICTED = 5


def _read_csv(path, delimiter=None, number_of_columns=None, skip_rows=None):
    if number_of_columns is not None:
        names = ["att" + str(i) for i in range(1, number_of_columns + 1)]
    else:
        names = None

    data = pd.read_csv(
        path,
        delimiter=delimiter,
        names=names,
        skipinitialspace=True,
        skiprows=skip_rows)

    return data


def _force_nominal_columns(data, forced_nominal_attributes):
    if forced_nominal_attributes is not None:
        for attribute_name in forced_nominal_attributes:
            data.loc[:, attribute_name] = data.loc[:, attribute_name].apply(str)

    return data


def _force_numerical_columns(data, forced_numerical_attributes):
    if forced_numerical_attributes is not None:
        data[forced_numerical_attributes] = data[forced_numerical_attributes].apply(pd.to_numeric, errors="coerce")

    return data


def _replace_values(data, replacement_dict):
    if replacement_dict is not None:
        for attribute_name in replacement_dict.keys():
            for original_value in replacement_dict[attribute_name].keys():
                data.loc[:, attribute_name] = data.loc[:, attribute_name].replace(
                    original_value,
                    replacement_dict[attribute_name][original_value])

    return data


def _read_raw_train(dataset_meta, dataset_path):
    number_of_columns = None
    if not dataset_meta.read_raw_column_names:
        number_of_columns = dataset_meta.number_of_columns

    data = _read_csv(
        dataset_path + ".data",
        number_of_columns=number_of_columns,
        skip_rows=dataset_meta.skip_rows_raw_train)

    # fix different encoding in test vs train data
    data = _replace_values(data, dataset_meta.replace_values_raw_train)

    # fix nominal/numerical column format in test vs train data
    data = _force_nominal_columns(data, dataset_meta.forced_nominal_attributes_raw_train)
    data = _force_numerical_columns(data, dataset_meta.forced_numerical_attributes_raw_train)

    return data


def _read_raw_test(dataset_meta, dataset_path):
    number_of_columns = None
    if not dataset_meta.read_raw_column_names:
        number_of_columns = dataset_meta.number_of_columns

    test_data = _read_csv(
        dataset_path + ".test",
        number_of_columns=number_of_columns,
        skip_rows=dataset_meta.skip_rows_raw_test)

    # fix different encoding in test vs train data
    test_data = _replace_values(test_data, dataset_meta.replace_values_raw_test)

    # fix nominal/numerical column format in test vs train data
    test_data = _force_nominal_columns(test_data, dataset_meta.forced_nominal_attributes_raw_test)
    test_data = _force_numerical_columns(test_data, dataset_meta.forced_numerical_attributes_raw_test)

    return test_data


class DatasetReader:
    def __init__(self, data_dir):
        """
        Expects the following directory structure in data_dir:

        * data_dir

            * processed
            * raw
        """

        self.data_dir = data_dir

    def read_dataset(self, dataset_name: DatasetName, dataset_stage: DatasetStage, use_float32=False):
        dataset_meta = meta_dict[dataset_name]

        # read the dataset
        if dataset_stage == DatasetStage.RAW:
            data = self._read_raw(dataset_meta)
        elif dataset_stage == DatasetStage.PROCESSED_RAPIDMINER:
            data = self._read_processed_rapidminer(dataset_meta, use_float32)
        elif dataset_stage == DatasetStage.PROCESSED_PYTHON:
            data = self._read_processed_python(dataset_meta, use_float32)
        elif dataset_stage == DatasetStage.PROCESSED_MODEL_READY:
            data = self._read_processed_model_ready(dataset_meta, use_float32)
        elif dataset_stage == DatasetStage.PROCESSED_MODEL_PREDICTED:
            data = self._read_processed_model_predicted(dataset_meta, use_float32)
        elif dataset_stage == DatasetStage.PROCESSED_PERMUTED_MODEL_PREDICTED:
            data = self._read_processed_permuted_model_predicted(dataset_meta, use_float32)
        else:
            return None, dataset_meta

        return data, dataset_meta

    def _read_raw(self, dataset_meta):
        dataset_path = self.data_dir + "/raw/" + dataset_meta.dataset_dir + dataset_meta.raw_filename
        data = _read_raw_train(dataset_meta, dataset_path)

        if dataset_meta.has_test_split:
            test_data = _read_raw_test(dataset_meta, dataset_path)
            data = pd.concat([data, test_data])

        # declare missing values
        if dataset_meta.missing_value_symbol is not None:
            data = data.replace(dataset_meta.missing_value_symbol, None)

        # convert ground truth to true (positive sample)/false (negative sample)
        if dataset_meta.gt_true_value is not None:
            data[dataset_meta.gt_name] = ((data[dataset_meta.gt_name] == dataset_meta.gt_true_value)
                                          .astype("int64"))
        
        # drop columns
        if dataset_meta.attributes_to_drop is not None:
            attributes_to_keep = [column_name for column_name in data.columns
                                    if column_name not in dataset_meta.attributes_to_drop]
            data = data[attributes_to_keep]

        return data

    def _read_processed(self, dataset_meta, data_filename, csv_delimiter, use_float32=False):
        dataset_path = self.data_dir + "/processed/" + dataset_meta.dataset_dir + data_filename

        data = _read_csv(
            dataset_path,
            delimiter=csv_delimiter,
            skip_rows=dataset_meta.skip_rows_processed)

        # fix different encoding in test vs train data
        data = _replace_values(data, dataset_meta.replace_values_processed)

        # fix nominal/numerical column format in test vs train data
        data = _force_nominal_columns(data, dataset_meta.forced_nominal_attributes_processed)
        data = _force_numerical_columns(data, dataset_meta.forced_numerical_attributes_processed)

        # declare missing values
        if dataset_meta.missing_value_symbol is not None:
            data = data.replace(dataset_meta.missing_value_symbol, None)

        if dataset_meta.score_name in data.columns.values.tolist():
            # set dataset prediction type in metadata
            if data.loc[:, dataset_meta.score_name].dtype == np.float64:
                dataset_meta.prediction_type = PredictionType.CLASSIFICATION_SOFT
            else:
                dataset_meta.prediction_type = PredictionType.CLASSIFICATION_HARD

            # downcast scores to float32 (for better reproduction because the SCaPE authors used Java float)
            if use_float32:
                data.loc[:, dataset_meta.score_name] = (
                    pd.DataFrame(data.loc[:, dataset_meta.score_name], dtype=np.float32))

        return data

    def _read_processed_rapidminer(self, dataset_meta, use_float32=False):
        return self._read_processed(dataset_meta, "rapidminer_all.csv", ";", use_float32)

    def _read_processed_python(self, dataset_meta, use_float32=False):
        return self._read_processed(dataset_meta, "python_all.csv", ",", use_float32)

    def _read_processed_model_ready(self, dataset_meta, use_float32=False):
        return (self._read_processed(dataset_meta, "model_ready_train.csv", ",", use_float32),
                self._read_processed(dataset_meta, "model_ready_test.csv", ",", use_float32))

    def _read_processed_model_predicted(self, dataset_meta, use_float32=False):
        return (self._read_processed(dataset_meta, "model_predicted_train.csv", ",", use_float32),
                self._read_processed(dataset_meta, "model_predicted_test.csv", ",", use_float32))

    def _read_processed_permuted_model_predicted(self, dataset_meta, use_float32=False):
        return (self._read_processed(dataset_meta, "permuted_model_predicted_train.csv", ",", use_float32),
                self._read_processed(dataset_meta, "permuted_model_predicted_test.csv", ",", use_float32))
