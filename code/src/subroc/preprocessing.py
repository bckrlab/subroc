import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from subroc.datasets.metadata import DatasetMetadata
from subroc.model_serialization import ModelName


def remove_missing(data: pd.DataFrame):
    notna_rows = [all(notna_row) for notna_row in pd.notna(data).to_numpy()]
    data = data.loc[notna_rows]
    data = data.reset_index(drop=True)
    return data


def train_test_val_split(array: any, test_size: float, val_size: float, random_state: int) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if val_size == 0:
        train_indices, test_indices = train_test_split(array, test_size=test_size, random_state=random_state)
        return train_indices, test_indices, np.empty_like(train_indices)

    train_indices, test_val_indices = train_test_split(array, test_size=test_size + val_size, random_state=random_state)
    test_indices, val_indices = train_test_split(test_val_indices, test_size=val_size / (test_size + val_size), random_state=random_state)

    return train_indices, test_indices, val_indices


def split_4(array: any, size_1: float, size_2: float, size_3: float, size_4: float, random_state: int) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if size_1 + size_2 + size_3 + size_4 != 1:
        raise ValueError("Split sizes must add up to 1.")

    indices_1, indices_2_3_4 = train_test_split(array, test_size=(size_2 + size_3 + size_4), random_state=random_state)
    indices_2, indices_3_4 = train_test_split(indices_2_3_4, test_size=(size_3 + size_4) / (size_2 + size_3 + size_4), random_state=random_state)
    indices_3, indices_4 = train_test_split(indices_3_4, test_size=(size_4)/(size_3 + size_4), random_state=random_state)

    return indices_1, indices_2, indices_3, indices_4


def _sklearn_gaussian_nb_credit_g_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    gt_name = dataset_meta.gt_name

    # split off gt data
    data_without_gt = data.loc[:, data.columns != gt_name]
    gt_data = data.loc[:, gt_name]

    # replace categorical attributes with dummies
    if dataset_meta.categorical_attributes is not None:
        data_without_gt_dummies = pd.get_dummies(data_without_gt.loc[:, dataset_meta.categorical_attributes])
        non_categorical_attributes = [column_name for column_name in data_without_gt.columns
                                    if column_name not in dataset_meta.categorical_attributes]
        data_without_gt_dummies[non_categorical_attributes] = data_without_gt[non_categorical_attributes]
        data = data_without_gt_dummies

    data[gt_name] = gt_data

    train_indices, test_indices, val_indices = (
        train_test_val_split(range(len(data)), test_size=0.5, val_size=0, random_state=seed))

    return data, train_indices, test_indices, val_indices


def _sklearn_gaussian_nb_credit_approval_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    gt_name = dataset_meta.gt_name

    data = remove_missing(data)

    # split off gt data
    data_without_gt = data.loc[:, data.columns != gt_name]
    data_without_gt_dummies = pd.get_dummies(data_without_gt)
    gt_data = data.loc[:, gt_name]

    # replace labels
    gt_data = gt_data.replace(False, 0).replace(True, 1)

    data = data_without_gt_dummies
    data[gt_name] = gt_data

    train_indices, test_indices, val_indices = (
        train_test_val_split(range(len(data)), test_size=0.2, val_size=0.2, random_state=seed))

    return data, train_indices, test_indices, val_indices


def _sklearn_multinomial_nb_credit_g_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_credit_g_preprocessing(data, dataset_meta, seed)


def _sklearn_multinomial_nb_mushroom_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_credit_approval_preprocessing(data, dataset_meta, seed)


def _sklearn_perceptron_credit_g_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_credit_g_preprocessing(data, dataset_meta, seed)


def _sklearn_gaussian_nb_adult_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_credit_g_preprocessing(data, dataset_meta, seed)


def _sklearn_gaussian_nb_adult_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt_name = dataset_meta.gt_name

    # split off gt data
    data_without_gt = data.loc[:, data.columns != gt_name]
    gt_data = data.loc[:, gt_name]

    # replace categorical attributes with dummies
    if dataset_meta.categorical_attributes is not None:
        data_without_gt_dummies = pd.get_dummies(data_without_gt.loc[:, dataset_meta.categorical_attributes])
        non_categorical_attributes = [column_name for column_name in data_without_gt.columns
                                    if column_name not in dataset_meta.categorical_attributes]
        data_without_gt_dummies[non_categorical_attributes] = data_without_gt[non_categorical_attributes]
        data = data_without_gt_dummies

    data[gt_name] = gt_data

    indices_1, indices_2, indices_3, indices_4 = (
        split_4(range(len(data)), size_1=0.25, size_2=0.25, size_3=0.25, size_4=0.25, random_state=seed))

    return data, indices_1, indices_2, indices_3, indices_4


def _sklearn_gaussian_nb_credit_g_4_splits_preprocessing(data:pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _sklearn_gaussian_nb_uci_mushroom_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _sklearn_gaussian_nb_uci_wisconsin_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = remove_missing(data)
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _sklearn_gaussian_nb_uci_census_kdd_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = remove_missing(data)
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _sklearn_gaussian_nb_uci_credit_a_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = remove_missing(data)
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _sklearn_gaussian_nb_uci_bank_marketing_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _sklearn_gaussian_nb_uci_credit_card_clients_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _sklearn_gaussian_nb_uci_diabetes_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _sklearn_lin_regression_uci_diabetes_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _sklearn_log_regression_uci_diabetes_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    return _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)


def _xgboost_xgbclassifier_census_kdd_4_splits_preprocessing(data: pd.DataFrame, dataset_meta: DatasetMetadata, seed: int) \
        -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    data = remove_missing(data)
    data, indices_1, indices_2, indices_3, indices_4 = _sklearn_gaussian_nb_adult_4_splits_preprocessing(data, dataset_meta, seed)
    data = data.rename(columns=lambda x: x.replace("<", "less than"))

    return data, indices_1, indices_2, indices_3, indices_4


preprocessing_functions = {
    None: _sklearn_gaussian_nb_uci_census_kdd_4_splits_preprocessing,  # default
    ModelName.sklearn_gaussian_nb_credit_g: _sklearn_gaussian_nb_credit_g_preprocessing,
    ModelName.sklearn_gaussian_nb_credit_g_4_splits: _sklearn_gaussian_nb_credit_g_4_splits_preprocessing,
    ModelName.sklearn_gaussian_nb_credit_approval: _sklearn_gaussian_nb_credit_approval_preprocessing,
    ModelName.sklearn_multinomial_nb_credit_g: _sklearn_multinomial_nb_credit_g_preprocessing,
    ModelName.sklearn_multinomial_nb_mushroom: _sklearn_multinomial_nb_mushroom_preprocessing,
    ModelName.sklearn_perceptron_credit_g: _sklearn_perceptron_credit_g_preprocessing,
    ModelName.sklearn_gaussian_nb_adult: _sklearn_gaussian_nb_adult_preprocessing,
    ModelName.sklearn_gaussian_nb_adult_4_splits: _sklearn_gaussian_nb_adult_4_splits_preprocessing,
    ModelName.sklearn_gaussian_nb_uci_mushroom_4_splits: _sklearn_gaussian_nb_uci_mushroom_4_splits_preprocessing,
    ModelName.sklearn_gaussian_nb_uci_wisconsin_4_splits: _sklearn_gaussian_nb_uci_wisconsin_4_splits_preprocessing,
    ModelName.sklearn_gaussian_nb_uci_census_kdd_4_splits: _sklearn_gaussian_nb_uci_census_kdd_4_splits_preprocessing,
    ModelName.sklearn_gaussian_nb_uci_credit_a_4_splits: _sklearn_gaussian_nb_uci_credit_a_4_splits_preprocessing,
    ModelName.sklearn_gaussian_nb_uci_bank_marketing_4_splits: _sklearn_gaussian_nb_uci_bank_marketing_4_splits_preprocessing,
    ModelName.sklearn_gaussian_nb_uci_credit_card_clients_4_splits: _sklearn_gaussian_nb_uci_credit_card_clients_4_splits_preprocessing,
    ModelName.sklearn_gaussian_nb_uci_diabetes_4_splits: _sklearn_gaussian_nb_uci_diabetes_4_splits_preprocessing,
    ModelName.sklearn_lin_regression_adult_4_splits: _sklearn_lin_regression_uci_diabetes_4_splits_preprocessing,
    ModelName.sklearn_log_regression_adult_4_splits: _sklearn_log_regression_uci_diabetes_4_splits_preprocessing,
    ModelName.xgboost_xgbclassifier_census_kdd_4_splits: _xgboost_xgbclassifier_census_kdd_4_splits_preprocessing,
}

