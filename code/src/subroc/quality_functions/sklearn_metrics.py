from sklearn import metrics

from subroc.quality_functions.base_qf import OptimizationMode, PredictionType
from subroc.quality_functions.metric_qf_wrapper import ClassifierMetricQF, MetricType, Precondition
from subroc.quality_functions.optimistic_estimates import (ROCAUCOptimisticEstimate, PrecisionOptimisticEstimate,
                                                           RecallOptimisticEstimate, FBetaOptimisticEstimate)


def extend_probabilities(data):
    return [[1 - x, x] for x in data]


class SklearnMetricWrapper:
    def __init__(
            self,
            metric,
            extend_probabilities=False,
            **kwargs):
        self.metric = metric
        self.__name__ = metric.__name__
        self.extend_probabilities = extend_probabilities
        self.kwargs = kwargs

    def __call__(self, y_true, y_pred, *args, **kwargs):
        # take self.kwargs as baseline, that can be overwritten by kwargs
        merged_kwargs = {**self.kwargs, **kwargs}

        if self.extend_probabilities:
            y_true = extend_probabilities(y_true)
            y_pred = extend_probabilities(y_pred)

        return self.metric(y_true, y_pred, *args, **merged_kwargs)


soft_classification_metrics = [
    metrics.average_precision_score,
    metrics.brier_score_loss,
    SklearnMetricWrapper(metrics.dcg_score, extend_probabilities=True),
    metrics.hinge_loss,
    SklearnMetricWrapper(metrics.log_loss, labels=[0, 1]),
    SklearnMetricWrapper(metrics.ndcg_score, extend_probabilities=True),
    metrics.roc_auc_score,
    SklearnMetricWrapper(metrics.top_k_accuracy_score, k=1, labels=[0, 1])
]


def get_soft_classification_metric_qfs(optimization_mode: OptimizationMode,
                                       random_sampling_normalization: bool = False,
                                       num_random_samples: int = 10000,
                                       seed: int = 0,
                                       enable_optimistic_estimates: bool = False):
    return list(
        get_soft_classification_metric_qf_dict(
            optimization_mode,
            random_sampling_normalization,
            num_random_samples,
            seed,
            enable_optimistic_estimates
        ).values()
    )


def get_soft_classification_metric_qf_dict(optimization_mode: OptimizationMode,
                                           random_sampling_normalization: bool = False,
                                           num_random_samples: int = 10000,
                                           seed: int = 0,
                                           enable_optimistic_estimates: bool = False):
    roc_auc_optimistic_estimate = None
    if enable_optimistic_estimates:
        roc_auc_optimistic_estimate = ROCAUCOptimisticEstimate()

    return {
        "sklearn.metrics.average_precision_score": ClassifierMetricQF(metrics.average_precision_score,
                                                                      MetricType.Score,
                                                                      PredictionType.CLASSIFICATION_SOFT,
                                                                      optimization_mode,
                                                                      Precondition.ContainsTruePositives,
                                                                      random_sampling_normalization=random_sampling_normalization,
                                                                      num_random_samples=num_random_samples,
                                                                      seed=seed),
        "sklearn.metrics.brier_score_loss": ClassifierMetricQF(metrics.brier_score_loss,
                                                               MetricType.Loss,
                                                               PredictionType.CLASSIFICATION_SOFT,
                                                               optimization_mode,
                                                               random_sampling_normalization=random_sampling_normalization,
                                                               num_random_samples=num_random_samples,
                                                               seed=seed),
        "sklearn.metrics.dcg_score": ClassifierMetricQF(SklearnMetricWrapper(metrics.dcg_score, extend_probabilities=True),
                                                        MetricType.Score,
                                                        PredictionType.CLASSIFICATION_SOFT,
                                                        optimization_mode,
                                                        random_sampling_normalization=random_sampling_normalization,
                                                        num_random_samples=num_random_samples,
                                                        seed=seed),
        "sklearn.metrics.log_loss": ClassifierMetricQF(SklearnMetricWrapper(metrics.log_loss, labels=[0, 1]),
                                                       MetricType.Loss,
                                                       PredictionType.CLASSIFICATION_SOFT,
                                                       optimization_mode,
                                                       random_sampling_normalization=random_sampling_normalization,
                                                       num_random_samples=num_random_samples,
                                                       seed=seed),
        "sklearn.metrics.ndcg_score": ClassifierMetricQF(SklearnMetricWrapper(metrics.ndcg_score, extend_probabilities=True),
                                                         MetricType.Score,
                                                         PredictionType.CLASSIFICATION_SOFT,
                                                         optimization_mode,
                                                         random_sampling_normalization=random_sampling_normalization,
                                                         num_random_samples=num_random_samples,
                                                         seed=seed),
        "sklearn.metrics.roc_auc_score": ClassifierMetricQF(metrics.roc_auc_score,
                                                            MetricType.Score,
                                                            PredictionType.CLASSIFICATION_SOFT,
                                                            optimization_mode,
                                                            Precondition.ContainsAllClassesInTrue,
                                                            random_sampling_normalization=random_sampling_normalization,
                                                            num_random_samples=num_random_samples,
                                                            seed=seed,
                                                            metric_optimistic_estimate=roc_auc_optimistic_estimate),
        "sklearn.metrics.top_k_accuracy_score": ClassifierMetricQF(SklearnMetricWrapper(metrics.top_k_accuracy_score, k=1, labels=[0, 1]),
                                                                   MetricType.Score,
                                                                   PredictionType.CLASSIFICATION_SOFT,
                                                                   optimization_mode,
                                                                   random_sampling_normalization=random_sampling_normalization,
                                                                   num_random_samples=num_random_samples,
                                                                   seed=seed)
    }


hard_classification_metrics = [
    metrics.accuracy_score,
    metrics.balanced_accuracy_score,
    # metrics.class_likelihood_ratios,
    # metrics.classification_report,
    metrics.cohen_kappa_score,
    # metrics.confusion_matrix,
    metrics.f1_score,
    SklearnMetricWrapper(metrics.fbeta_score, beta=1),
    metrics.hamming_loss,
    metrics.jaccard_score,
    metrics.matthews_corrcoef,
    # metrics.multilabel_confusion_matrix,
    # metrics.precision_recall_fscore_support,
    metrics.precision_score,
    metrics.recall_score,
    metrics.zero_one_loss
]


def get_hard_classification_metric_qfs(optimization_mode: OptimizationMode,
                                       random_sampling_normalization: bool = False,
                                       num_random_samples: int = 10000,
                                       seed: int = 0,
                                       enable_optimistic_estimates: bool = False):
    return list(
        get_hard_classification_metric_qf_dict(
            optimization_mode,
            random_sampling_normalization,
            num_random_samples,
            seed,
            enable_optimistic_estimates
        ).values()
    )


def get_hard_classification_metric_qf_dict(optimization_mode: OptimizationMode,
                                           random_sampling_normalization: bool = False,
                                           num_random_samples: int = 10000,
                                           seed: int = 0,
                                           enable_optimistic_estimates: bool = False,
                                           beta: float = 1):
    precision_optimistic_estimate = None
    recall_optimistic_estimate = None
    f1_optimistic_estimate = None
    fbeta_optimistic_estimate = None
    if enable_optimistic_estimates:
        precision_optimistic_estimate = PrecisionOptimisticEstimate()
        recall_optimistic_estimate = RecallOptimisticEstimate()
        f1_optimistic_estimate = FBetaOptimisticEstimate(beta=1)
        fbeta_optimistic_estimate = FBetaOptimisticEstimate(beta=beta)

    return {
        "sklearn.metrics.accuracy_score": ClassifierMetricQF(metrics.accuracy_score,
                                                             MetricType.Score,
                                                             PredictionType.CLASSIFICATION_HARD,
                                                             optimization_mode,
                                                             random_sampling_normalization=random_sampling_normalization,
                                                             num_random_samples=num_random_samples,
                                                             seed=seed),
        "sklearn.metrics.balanced_accuracy_score": ClassifierMetricQF(metrics.balanced_accuracy_score,
                                                                      MetricType.Score,
                                                                      PredictionType.CLASSIFICATION_HARD,
                                                                      optimization_mode,
                                                                      Precondition.ContainsAllClassesInTrue,
                                                                      random_sampling_normalization=random_sampling_normalization,
                                                                      num_random_samples=num_random_samples,
                                                                      seed=seed),
        "sklearn.metrics.cohen_kappa_score": ClassifierMetricQF(metrics.cohen_kappa_score,
                                                                MetricType.Score,
                                                                PredictionType.CLASSIFICATION_HARD,
                                                                optimization_mode,
                                                                Precondition.ContainsAllClassesInTrue,
                                                                random_sampling_normalization=random_sampling_normalization,
                                                                num_random_samples=num_random_samples,
                                                                seed=seed),
        "sklearn.metrics.f1_score": ClassifierMetricQF(metrics.f1_score,
                                                       MetricType.Score,
                                                       PredictionType.CLASSIFICATION_HARD, optimization_mode,
                                                       [Precondition.ContainsTruePositives,
                                                        Precondition.ContainsPredictedPositives],
                                                       random_sampling_normalization=random_sampling_normalization,
                                                       num_random_samples=num_random_samples,
                                                       seed=seed,
                                                       metric_optimistic_estimate=f1_optimistic_estimate),
        "sklearn.metrics.fbeta_score": ClassifierMetricQF(SklearnMetricWrapper(metrics.fbeta_score, beta=beta),
                                                          MetricType.Score,
                                                          PredictionType.CLASSIFICATION_HARD, optimization_mode,
                                                          [Precondition.ContainsTruePositives,
                                                           Precondition.ContainsPredictedPositives],
                                                          random_sampling_normalization=random_sampling_normalization,
                                                          num_random_samples=num_random_samples,
                                                          seed=seed,
                                                          metric_optimistic_estimate=fbeta_optimistic_estimate),
        "sklearn.metrics.hamming_loss": ClassifierMetricQF(metrics.hamming_loss,
                                                           MetricType.Loss,
                                                           PredictionType.CLASSIFICATION_HARD,
                                                           optimization_mode,
                                                           random_sampling_normalization=random_sampling_normalization,
                                                           num_random_samples=num_random_samples,
                                                           seed=seed),
        "sklearn.metrics.jaccard_score": ClassifierMetricQF(metrics.jaccard_score,
                                                            MetricType.Score,
                                                            PredictionType.CLASSIFICATION_HARD,
                                                            optimization_mode,
                                                            [[Precondition.ContainsTruePositives,
                                                              Precondition.ContainsPredictedPositives]],
                                                            random_sampling_normalization=random_sampling_normalization,
                                                            num_random_samples=num_random_samples,
                                                            seed=seed),
        "sklearn.metrics.matthews_corrcoef": ClassifierMetricQF(metrics.matthews_corrcoef,
                                                                MetricType.Score,
                                                                PredictionType.CLASSIFICATION_HARD,
                                                                optimization_mode,
                                                                random_sampling_normalization=random_sampling_normalization,
                                                                num_random_samples=num_random_samples,
                                                                seed=seed),
        "sklearn.metrics.precision_score": ClassifierMetricQF(metrics.precision_score,
                                                              MetricType.Score,
                                                              PredictionType.CLASSIFICATION_HARD,
                                                              optimization_mode,
                                                              Precondition.ContainsPredictedPositives,
                                                              random_sampling_normalization=random_sampling_normalization,
                                                              num_random_samples=num_random_samples,
                                                              seed=seed,
                                                              metric_optimistic_estimate=precision_optimistic_estimate),
        "sklearn.metrics.recall_score": ClassifierMetricQF(metrics.recall_score,
                                                           MetricType.Score,
                                                           PredictionType.CLASSIFICATION_HARD,
                                                           optimization_mode,
                                                           Precondition.ContainsTruePositives,
                                                           random_sampling_normalization=random_sampling_normalization,
                                                           num_random_samples=num_random_samples,
                                                           seed=seed,
                                                           metric_optimistic_estimate=recall_optimistic_estimate),
        "sklearn.metrics.zero_one_loss": ClassifierMetricQF(metrics.zero_one_loss,
                                                            MetricType.Loss,
                                                            PredictionType.CLASSIFICATION_HARD,
                                                            optimization_mode,
                                                            random_sampling_normalization=random_sampling_normalization,
                                                            num_random_samples=num_random_samples,
                                                            seed=seed)
    }


multilabel_ranking_metrics = [
    SklearnMetricWrapper(metrics.coverage_error, extend_probabilities=True),
    SklearnMetricWrapper(metrics.label_ranking_average_precision_score, extend_probabilities=True),
    SklearnMetricWrapper(metrics.label_ranking_loss, extend_probabilities=True)
]


def get_multilabel_ranking_metric_qfs(optimization_mode: OptimizationMode,
                                      random_sampling_normalization: bool = False,
                                      num_random_samples: int = 10000,
                                      seed: int = 0,
                                      enable_optimistic_estimates: bool = False):
    return list(
        get_multilabel_ranking_metric_qf_dict(
            optimization_mode,
            random_sampling_normalization,
            num_random_samples,
            seed,
            enable_optimistic_estimates
        ).values()
    )


def get_multilabel_ranking_metric_qf_dict(optimization_mode: OptimizationMode,
                                          random_sampling_normalization: bool = False,
                                          num_random_samples: int = 10000,
                                          seed: int = 0,
                                          enable_optimistic_estimates: bool = False):
    return {
        "sklearn.metrics.coverage_error": ClassifierMetricQF(SklearnMetricWrapper(metrics.coverage_error, extend_probabilities=True),
                                                             MetricType.Loss,
                                                             PredictionType.CLASSIFICATION_SOFT,
                                                             optimization_mode,
                                                             random_sampling_normalization=random_sampling_normalization,
                                                             num_random_samples=num_random_samples,
                                                             seed=seed),
        "sklearn.metrics.label_ranking_average_precision_score": ClassifierMetricQF(SklearnMetricWrapper(metrics.label_ranking_average_precision_score, extend_probabilities=True),
                                                                                    MetricType.Score,
                                                                                    PredictionType.CLASSIFICATION_SOFT,
                                                                                    optimization_mode,
                                                                                    random_sampling_normalization=random_sampling_normalization,
                                                                                    num_random_samples=num_random_samples,
                                                                                    seed=seed),
        "sklearn.metrics.label_ranking_loss": ClassifierMetricQF(SklearnMetricWrapper(metrics.label_ranking_loss, extend_probabilities=True),
                                                                 MetricType.Loss,
                                                                 PredictionType.CLASSIFICATION_SOFT,
                                                                 optimization_mode,
                                                                 random_sampling_normalization=random_sampling_normalization,
                                                                 num_random_samples=num_random_samples,
                                                                 seed=seed)
    }
