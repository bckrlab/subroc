from subroc.preconditions import Precondition
from subroc.quality_functions.base_qf import OptimizationMode, PredictionType
from subroc.quality_functions.metric_qf_wrapper import ClassifierMetricQF, MetricType
from subroc.quality_functions.optimistic_estimates import ARLOptimisticEstimate, PRCAUCOptimisticEstimate
from subroc.metrics import average_ranking_loss, prc_auc_score


def get_custom_metric_qfs(optimization_mode: OptimizationMode,
                          random_sampling_normalization: bool = False,
                          num_random_samples: int = 10000,
                          seed: int = 0,
                          enable_optimistic_estimates: bool = False):
    return list(
        get_custom_metric_qf_dict(
            optimization_mode,
            random_sampling_normalization,
            num_random_samples,
            seed,
            enable_optimistic_estimates
        ).values()
    )


def get_custom_metric_qf_dict(optimization_mode: OptimizationMode,
                              random_sampling_normalization: bool = False,
                              num_random_samples: int = 10000,
                              seed: int = 0,
                              enable_optimistic_estimates: bool = False):
    arl_optimistic_estimate = None
    prc_auc_optimistic_estimate = None
    if enable_optimistic_estimates:
        arl_optimistic_estimate = ARLOptimisticEstimate()
        prc_auc_optimistic_estimate = PRCAUCOptimisticEstimate()

    return {
        "average_ranking_loss": ClassifierMetricQF(
            average_ranking_loss,
            MetricType.Loss,
            PredictionType.CLASSIFICATION_SOFT,
            optimization_mode,
            Precondition.ContainsTruePositives,
            random_sampling_normalization=random_sampling_normalization,
            num_random_samples=num_random_samples,
            seed=seed,
            metric_optimistic_estimate=arl_optimistic_estimate
        ),
        "prc_auc_score": ClassifierMetricQF(
            prc_auc_score,
            MetricType.Score,
            PredictionType.CLASSIFICATION_SOFT,
            optimization_mode,
            Precondition.ContainsTruePositives,
            random_sampling_normalization=random_sampling_normalization,
            num_random_samples=num_random_samples,
            seed=seed,
            metric_optimistic_estimate=prc_auc_optimistic_estimate
        )
    }
