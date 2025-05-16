from enum import Enum

import pandas as pd

from subroc.constraints import ConstraintDisjunction, MinClassesConstraint, ContainsValueConstraint
from subroc.datasets.metadata import DatasetMetadata


class Precondition(str, Enum):
    ContainsAllClassesInTrue = "contains all classes in true labels"
    ContainsAllClassesInPredicted = "contains all classes in predicted labels"
    ContainsTruePositives = "contains true positives"
    ContainsTrueNegatives = "contains true negatives"
    ContainsPredictedPositives = "contains predicted positives"
    ContainsPredictedNegatives = "contains predicted negatives"


def constraint_for_precondition(
        precondition: Precondition,
        data: pd.DataFrame,
        dataset_meta: DatasetMetadata,
        num_classes: int = 2,
        positive_value: int = 1,
        negative_value: int = 0):
    return {
        Precondition.ContainsAllClassesInPredicted: lambda: MinClassesConstraint(num_classes, dataset_meta.score_name, data),
        Precondition.ContainsAllClassesInTrue: lambda: MinClassesConstraint(num_classes, dataset_meta.gt_name, data),
        Precondition.ContainsTruePositives: lambda: ContainsValueConstraint(dataset_meta.gt_name, positive_value, data),
        Precondition.ContainsTrueNegatives: lambda: ContainsValueConstraint(dataset_meta.gt_name, negative_value, data),
        Precondition.ContainsPredictedPositives: lambda: ContainsValueConstraint(dataset_meta.score_name, positive_value, data),
        Precondition.ContainsPredictedNegatives: lambda: ContainsValueConstraint(dataset_meta.score_name, negative_value, data)
    }[precondition]()


def constraint_for_precondition_disjunction(
        precondition_disjunction: list[Precondition],
        data: pd.DataFrame,
        dataset_meta: DatasetMetadata,
        num_classes: int = 2,
        positive_value: int = 1,
        negative_value: int = 0):
    if len(precondition_disjunction) == 1:
        return constraint_for_precondition(
            precondition_disjunction[0],
            data,
            dataset_meta,
            num_classes,
            positive_value,
            negative_value)

    constraints = [
        constraint_for_precondition(precondition, data, dataset_meta, num_classes, positive_value, negative_value)
        for precondition in precondition_disjunction]

    return ConstraintDisjunction(constraints)

