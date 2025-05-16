import os
import pickle
from enum import Enum


class ModelName(str, Enum):
    sklearn_gaussian_nb_credit_g = "sklearn_gaussian_nb_credit-g"
    sklearn_gaussian_nb_credit_g_4_splits = "sklearn_gaussian_nb_credit-g_4_splits"
    sklearn_gaussian_nb_credit_approval = "sklearn_gaussian_nb_credit_approval"
    sklearn_multinomial_nb_credit_g = "sklearn_multinomial_nb_credit-g"
    sklearn_multinomial_nb_mushroom = "sklearn_multinomial_nb_mushroom"
    sklearn_perceptron_credit_g = "sklearn_perceptron_credit-g"
    sklearn_gaussian_nb_adult = "sklearn_gaussian_nb_adult"
    sklearn_gaussian_nb_adult_4_splits = "sklearn_gaussian_nb_adult_4_splits"
    sklearn_gaussian_nb_uci_mushroom_4_splits = "sklearn_gaussian_nb_uci_mushroom_4_splits"
    sklearn_gaussian_nb_uci_wisconsin_4_splits = "sklearn_gaussian_nb_uci_wisconsin_4_splits"
    sklearn_gaussian_nb_uci_census_kdd_4_splits = "sklearn_gaussian_nb_uci_census_kdd_4_splits"
    sklearn_gaussian_nb_uci_credit_a_4_splits = "sklearn_gaussian_nb_uci_credit_a_4_splits"
    sklearn_gaussian_nb_uci_bank_marketing_4_splits = "sklearn_gaussian_nb_uci_bank_marketing_4_splits"
    sklearn_gaussian_nb_uci_credit_card_clients_4_splits = "sklearn_gaussian_nb_uci_credit_card_clients_4_splits"
    sklearn_gaussian_nb_uci_communities_and_crime_4_splits = "sklearn_gaussian_nb_uci_communities_and_crime_4_splits"
    sklearn_gaussian_nb_uci_diabetes_4_splits = "sklearn_gaussian_nb_uci_diabetes_4_splits"
    sklearn_gaussian_nb_uci_student_performance_4_splits = "sklearn_gaussian_nb_uci_student_performance_4_splits"
    sklearn_lin_regression_adult_4_splits = "sklearn_lin_regression_adult_4_splits"
    sklearn_log_regression_adult_4_splits = "sklearn_log_regression_adult_4_splits"
    xgboost_xgbclassifier_census_kdd_4_splits = "xgboost_xgbclassifier_census_kdd_4_splits"



def to_ModelName(model_name):
    for name_enum in ModelName:
        if str.lower(model_name) == str.lower(name_enum):
            return name_enum
    
    return None


def serialize(model, models_dir, name):
    if not os.path.exists(models_dir):
        os.mkdir(models_dir)

    with open(models_dir + "/" + name + ".pickle", "wb") as file:
        pickle.dump(model, file)


def deserialize(models_dir, name):
    with open(models_dir + "/" + name + ".pickle", "rb") as file:
        model = pickle.load(file)

    return model

