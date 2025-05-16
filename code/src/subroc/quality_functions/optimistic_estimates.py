import math

import numpy as np
from sklearn import metrics

from subroc.metrics import prc_auc_score


def contains_inversion(y_true):
    """
    Returns True if y_true is not just a (possibly empty) sequence of False values followed by
    a (possibly empty) sequence of True values.
    """
    true_found = False

    for label in y_true:
        if (not true_found) and label:
            true_found = True
        if true_found and (not label):
            return True

    return False


def contains_tie(y_true, y_pred):
    """
    Returns True if two different indices with the same y_pred value have different y_true values.
    """
    previous_true = y_true[0]
    previous_pred = y_pred[0]

    for i in range(len(y_true)):
        if y_pred[i] != previous_pred:
            previous_true = y_true[i]
            previous_pred = y_pred[i]
        elif y_true[i] != previous_true:
            return True

    return False


def contains_error(y_true, y_pred):
    """
    Returns True if an index with y_true=1 is followed by an index with y_true=0 and higher y_pred.
    """
    true_found = False
    true_pred = None

    for i in range(len(y_true)):
        if (not true_found) and y_true[i]:
            true_found = True
            true_pred = y_pred[i]
        if true_found and (not y_true[i]) and true_pred < y_pred[i]:
            return True

    return False


class BaseOptimisticEstimate:
    def __init__(self):
        pass

    def upper_bound(self, y_true, y_pred) -> float:
        pass

    def lower_bound(self, y_true, y_pred) -> float:
        pass


class ARLOptimisticEstimate(BaseOptimisticEstimate):
    """
    Optimistic Estimate for Average Ranking Loss (ARL) metric following the definitions in
    https://ieeexplore.ieee.org/document/7023405 (the SCaPE Paper) for binary classification.
    """

    def __init__(self):
        super().__init__()

    def upper_bound(self, y_true, y_pred):
        max_pen = 0
        positive_found_score = None

        for label, score in zip(y_true, y_pred):
            if positive_found_score is None and label:
                # first positive instance found -> start counting negatives from here
                positive_found_score = score

            if positive_found_score is not None and not label:
                # negative instance after positive instance found -> add to penalty
                if positive_found_score == score:
                    max_pen += 0.5
                else:
                    max_pen += 1

        return max_pen

    def lower_bound(self, y_true, y_pred):
        return 0


class ROCAUCOptimisticEstimate(BaseOptimisticEstimate):
    def __init__(self):
        super().__init__()

    def upper_bound(self, y_true, y_pred):
        if not contains_inversion(reversed(y_true)):
            return 0

        return 1

    def lower_bound(self, y_true, y_pred):
        if len(y_pred) == 0:
            return 0

        if contains_error(y_true, y_pred):
            return 0

        if contains_tie(y_true, y_pred):
            return 0.5

        return 1


class PRCAUCOptimisticEstimate(BaseOptimisticEstimate):
    def __init__(self):
        super().__init__()

    def upper_bound(self, y_true, y_pred):
        return np.inf

    def lower_bound(self, y_true, y_pred):
        if len(y_pred) == 0:
            return 0

        if (not contains_error(y_true, y_pred)) and (not contains_tie(y_true, y_pred)):
            return 1

        worst_subset_y_true = []
        worst_subset_y_pred = []
        positive_found = False
        previous_negative_found = 0
        previous_negative_found_score = None

        for label, score in zip(y_true, y_pred):
            if not positive_found and not label:
                if previous_negative_found_score is None or previous_negative_found_score != score:
                    previous_negative_found_score = score
                    previous_negative_found = 1
                else:
                    previous_negative_found += 1

            if not positive_found and label:
                positive_found = True
                worst_subset_y_true.append(label)
                worst_subset_y_pred.append(score)

                # add previous tied negatives
                if score == previous_negative_found_score:
                    for i in range(previous_negative_found):
                        worst_subset_y_true.append(0)
                        worst_subset_y_pred.append(previous_negative_found_score)

            if positive_found and not label:
                worst_subset_y_true.append(label)
                worst_subset_y_pred.append(score)

        return prc_auc_score(worst_subset_y_true, worst_subset_y_pred)


class PrecisionOptimisticEstimate(BaseOptimisticEstimate):
    def __init__(self):
        super().__init__()

    def upper_bound(self, y_true, y_pred):
        return math.ceil(metrics.precision_score(y_true, y_pred))

    def lower_bound(self, y_true, y_pred):
        return math.floor(metrics.precision_score(y_true, y_pred))


class RecallOptimisticEstimate(BaseOptimisticEstimate):
    def __init__(self):
        super().__init__()

    def upper_bound(self, y_true, y_pred):
        return math.ceil(metrics.recall_score(y_true, y_pred))

    def lower_bound(self, y_true, y_pred):
        return math.floor(metrics.recall_score(y_true, y_pred))


class FBetaOptimisticEstimate(BaseOptimisticEstimate):
    def __init__(self, beta: float = 1):
        super().__init__()

        self.beta = beta

        self.precision_optimistic_estimate = PrecisionOptimisticEstimate()
        self.recall_optimistic_estimate = RecallOptimisticEstimate()

    def upper_bound(self, y_true, y_pred):
        precision_upper_bound = self.precision_optimistic_estimate.upper_bound(y_true, y_pred)
        recall_upper_bound = self.recall_optimistic_estimate.upper_bound(y_true, y_pred)

        # prevent division by zero
        if self.beta == 0 or precision_upper_bound == 0 or precision_upper_bound == 0:
            return 1 + (self.beta ** 2)  # theoretical maximum of the fbeta score

        return ((1 + (self.beta ** 2)) *
                ((precision_upper_bound * recall_upper_bound) /
                 (((self.beta ** 2) * precision_upper_bound) + recall_upper_bound)))

    def lower_bound(self, y_true, y_pred):
        precision_lower_bound = self.precision_optimistic_estimate.lower_bound(y_true, y_pred)
        recall_lower_bound = self.precision_optimistic_estimate.lower_bound(y_true, y_pred)

        # prevent divison by zero
        if self.beta == 0 or precision_lower_bound == 0 or recall_lower_bound == 0:
            return 0  # theoretical minimum of the fbeta score

        return ((1 + (self.beta ** 2)) *
                ((precision_lower_bound * recall_lower_bound) /
                 (((self.beta ** 2) * precision_lower_bound) + recall_lower_bound)))

