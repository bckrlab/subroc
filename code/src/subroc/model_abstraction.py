from enum import Enum
from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

class ModelType(str, Enum):
    sklearn_linear_model_LinearRegression = "sklearn_linear_model_LinearRegression"
    sklearn_linear_model_LogisticRegression = "sklearn_linear_model_LogisticRegression"
    sklearn_naive_bayes_GaussianNB = "sklearn_naive_bayes_GaussianNB"
    sklearn_ensemble_RandomForestClassifier = "sklearn_ensemble_RandomForestClassifier"
    sklearn_neural_network_MLPClassifier = "sklearn_neural_network_MLPClassifier"
    xgboost_XGBClassifier = "xgboost_XGBClassifier"


def to_ModelType(model_type):
    for type_enum in ModelType:
        if str.lower(model_type) == str.lower(type_enum):
            return type_enum
    
    return None


def instantiate(model_type, X, y,
                sklearn_logistic_regression_penalty=None,
                seed=0,
                classes=None,
                sklearn_mlpclassifier_hidden_layer_sizes=(100,)):
    if model_type == ModelType.sklearn_linear_model_LinearRegression:
        model =  SKLinearRegressionSoftClassifier()
        model.fit(X, y)
        return model
    elif model_type == ModelType.sklearn_linear_model_LogisticRegression:
        model = SKLogisticRegressionSoftClassifier(penalty=sklearn_logistic_regression_penalty, seed=seed)
        model.fit(X, y)
        return model
    elif model_type == ModelType.sklearn_naive_bayes_GaussianNB:
        model = SKGaussianNBSoftClassifier()
        model.fit(X, y, classes=classes)
        return model
    elif model_type == ModelType.sklearn_ensemble_RandomForestClassifier:
        model = SKRandomForestClassifierSoftClassifier()
        model.fit(X, y)
        return model
    elif model_type == ModelType.sklearn_neural_network_MLPClassifier:
        model = SKMLPClassifierSoftClassifier(hidden_layer_sizes=sklearn_mlpclassifier_hidden_layer_sizes, random_state=seed)
        model.fit(X, y)
        return model
    elif model_type == ModelType.xgboost_XGBClassifier:
        model = XGBClassifierSoftClassifier()
        model.fit(X, y)
        return model


class SoftClassifier(ABC):
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class SKLinearRegressionSoftClassifier(SoftClassifier):
    def __init__(self):
        self.model = LinearRegression()

    def set_model(self, model):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)


class SKLogisticRegressionSoftClassifier(SoftClassifier):
    def __init__(self, penalty=None, seed=0):
        self.model = LogisticRegression(penalty=penalty, random_state=seed)
    
    def set_model(self, model):
        self.model = model   
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class SKGaussianNBSoftClassifier(SoftClassifier):
    def __init__(self):
        self.model = GaussianNB()
    
    def set_model(self, model):
        self.model = model
    
    def fit(self, X, y, classes=None):
        self.model.partial_fit(X, y, classes=classes)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class SKRandomForestClassifierSoftClassifier(SoftClassifier):
    def __init__(self):
        self.model = RandomForestClassifier()
    
    def set_model(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class SKMLPClassifierSoftClassifier(SoftClassifier):
    def __init__(self, hidden_layer_sizes, random_state):
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, random_state=random_state)
    
    def set_model(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


class XGBClassifierSoftClassifier(SoftClassifier):
    def __init__(self):
        self.model = XGBClassifier()
    
    def set_model(self, model):
        self.model = model
    
    def fit(self, X, y):
        self.model.fit(X, y)
    
    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

