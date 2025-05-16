import numpy as np

from sklearn import metrics


def average_ranking_loss(y_true, y_pred):
    """
    Average Ranking Loss (ARL) metric following the definitions in https://ieeexplore.ieee.org/document/7023405
    (the SCaPE Paper) for binary classification.

    Args:
        y_true: Binary Labels, must be ordered to match y_pred
        y_pred: Predicted Scores, must be in ascending order
    """
    negatives_loop_count = 0
    penalty_sum = 0

    last_score = np.inf
    current_positives_count = 0
    current_negatives_count = 0
    for current_gt, current_score in zip(reversed(y_true), reversed(y_pred)):
        if current_score < last_score:
            penalty_sum += (2 * negatives_loop_count + current_negatives_count) * current_positives_count
            last_score = current_score
            negatives_loop_count += current_negatives_count
            current_negatives_count = 0
            current_positives_count = 0

        if current_gt:
            current_positives_count += 1
        else:
            current_negatives_count += 1

    penalty_sum += 2 * negatives_loop_count * current_positives_count
    penalty_sum += current_negatives_count * current_positives_count  # fix to the original implementation
    negatives_loop_count += current_negatives_count
    positives_count = len(y_true) - negatives_loop_count

    ARL = penalty_sum / (2 * positives_count)

    return ARL


def prc_auc_score(y_true, y_pred):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_pred, drop_intermediate=True)
    return metrics.auc(recall, precision)
