from enum import Enum

import numpy as np
import pandas as pd

from subroc.quality_functions.base_qf import BaseQF, Precondition, OptimizationMode, PredictionType
from subroc.quality_functions.optimistic_estimates import BaseOptimisticEstimate
from subroc.quality_functions.soft_classifier_target import SoftClassifierTarget
from subroc.util import create_subgroup, from_str_Conjunction

import pysubgroup as ps


class MetricType(Enum):
    Score = 0,
    """Higher is better"""

    Loss = 1
    """Lower is better"""


class ClassifierMetricQF(BaseQF):
    def __init__(
            self,
            metric,
            metric_type: MetricType,
            prediction_type: PredictionType,
            optimization_mode: OptimizationMode = OptimizationMode.Maximal,
            preconditions: Precondition | list[Precondition] | list[list[Precondition]] = None,
            subgroup_class_balance_weight: float = 0,
            subgroup_size_weight: float = 0,
            relative_quality: bool = True,
            random_sampling_p_value_factor: bool = False,
            random_sampling_normalization: bool = False,
            num_random_samples: int = 10000,
            seed: int = 0,
            constraints: list[any] = None,
            max_random_sampling_tries: int = 10,
            metric_optimistic_estimate: BaseOptimisticEstimate = None,
            check_own_constraints: bool = False):
        """
        :param metric_type: determines whether higher is better (MetricType.Score) or lower is better (MetricType.Loss)
        """
        super().__init__(
            metric.__name__,
            optimization_mode,
            preconditions,
            subgroup_class_balance_weight,
            subgroup_size_weight,
            relative_quality,
            random_sampling_p_value_factor,
            random_sampling_normalization,
            num_random_samples,
            seed,
            constraints,
            max_random_sampling_tries,
            check_own_constraints)

        self.prediction_type = prediction_type
        self.metric = metric
        self.metric_type = metric_type

        self.dataset_quality = None

        self.metric_optimistic_estimate = metric_optimistic_estimate

    def calculate_constant_statistics(self, data: pd.DataFrame, target: SoftClassifierTarget):
        """ calculate_constant_statistics
            This function is called once for every search execution,
            it should do any preparation that is necessary prior to an execution.
        """
        super().calculate_constant_statistics(data, target)

        y_true = self.gt_sorted_by_score.to_numpy()
        y_pred = self.scores_sorted.to_numpy()
        self.dataset_quality = self.metric(y_true, y_pred)

    def calculate_statistics(self, subgroup, target, data, statistics=None):
        """ calculates necessary statistics
            this statistics object is passed on to the evaluate
            and optimistic_estimate functions
        """
        if not hasattr(subgroup, "representation"):
            subgroup = create_subgroup(data, subgroup._selectors)

        statistics = super().calculate_statistics(subgroup, target, data, statistics)

        return statistics

    def evaluate(self, subgroup, target: SoftClassifierTarget, data: pd.DataFrame, statistics=None):
        """ return the quality calculated from the statistics """
        if subgroup == slice(None):
            sel_conjunction = from_str_Conjunction("Dataset")
            subgroup = create_subgroup(data, sel_conjunction.selectors)

        if not hasattr(subgroup, "representation"):
            subgroup = create_subgroup(data, subgroup._selectors)

        if statistics is None:
            statistics = self.calculate_statistics(subgroup, target, data)
            # raise ValueError("Statistics need to be provided. Please run calculate_statistics first.")

        if self.check_own_constraints and self.constraints is not None:
            if not ps.constraints_satisfied(
                    self.constraints,
                    subgroup,
                    statistics,
                    data,
            ):
                return -np.inf  # subgroup does not fulfill the constraints -> maximally uninteresting

        sorted_subgroup_representation = \
            [subgroup.representation[original_index] for original_index in self.sorted_to_original_index]
        sorted_subgroup_y_true = self.gt_sorted_by_score[sorted_subgroup_representation].to_numpy()
        sorted_subgroup_y_pred = self.scores_sorted[sorted_subgroup_representation].to_numpy()
        statistics["metric"] = self.metric(sorted_subgroup_y_true, sorted_subgroup_y_pred)

        quality = statistics["metric"]

        if self.relative_quality:
            quality = quality - self.dataset_quality

        if self.metric_type == MetricType.Score:
            quality = -quality

        if self.optimization_mode == OptimizationMode.Minimal:
            quality = -quality
        elif self.optimization_mode == OptimizationMode.Exceptional:
            quality = abs(quality)

        return self.apply_significance_weighting(subgroup, target, data, quality, statistics=statistics)

    def optimistic_estimate(self, subgroup, target: SoftClassifierTarget, data: pd.DataFrame, statistics=None):
        """ returns optimistic estimate
            if one is available return it otherwise infinity"""
        if self.metric_optimistic_estimate is None:
            return np.inf

        if not hasattr(subgroup, "representation"):
            subgroup = create_subgroup(data, subgroup._selectors)

        # step 1: prepare estimate input
        sorted_subgroup_representation = \
            [subgroup.representation[original_index] for original_index in self.sorted_to_original_index]
        sorted_subgroup_y_true = self.gt_sorted_by_score[sorted_subgroup_representation].to_numpy()
        sorted_subgroup_y_pred = self.scores_sorted[sorted_subgroup_representation].to_numpy()

        # step 2: compute estimate for most extreme metric result
        if self.optimization_mode == OptimizationMode.Exceptional:
            metric_lower_bound = self.metric_optimistic_estimate.lower_bound(sorted_subgroup_y_true, sorted_subgroup_y_pred)
            metric_upper_bound = self.metric_optimistic_estimate.upper_bound(sorted_subgroup_y_true, sorted_subgroup_y_pred)

            quality_lower_bound = metric_lower_bound
            quality_upper_bound = metric_upper_bound
            if self.relative_quality:
                quality_lower_bound = abs(quality_lower_bound - self.dataset_quality)
                quality_upper_bound = abs(quality_upper_bound - self.dataset_quality)

            metric_estimate = metric_lower_bound if quality_lower_bound > quality_upper_bound else metric_upper_bound
        else:
            if self.metric_type == MetricType.Score:
                if self.optimization_mode == OptimizationMode.Maximal:
                    metric_estimate = self.metric_optimistic_estimate.lower_bound(sorted_subgroup_y_true, sorted_subgroup_y_pred)
                else:  # self.optimization_mode == OptimizationMode.Minimal
                    metric_estimate = self.metric_optimistic_estimate.upper_bound(sorted_subgroup_y_true, sorted_subgroup_y_pred)
            else:  # self.metric_type == MetricType.Loss
                if self.optimization_mode == OptimizationMode.Maximal:
                    metric_estimate = self.metric_optimistic_estimate.upper_bound(sorted_subgroup_y_true, sorted_subgroup_y_pred)
                else:  # self.optimization_mode == OptimizationMode.Minimal
                    metric_estimate = self.metric_optimistic_estimate.lower_bound(sorted_subgroup_y_true, sorted_subgroup_y_pred)

        # step 3: postprocess the result as in evaluate()
        quality_estimate = metric_estimate
        if self.relative_quality:
            quality_estimate = quality_estimate - self.dataset_quality

        if self.metric_type == MetricType.Score:
            quality_estimate = -quality_estimate

        if self.optimization_mode == OptimizationMode.Minimal:
            quality_estimate = -quality_estimate
        elif self.optimization_mode == OptimizationMode.Exceptional:
            quality_estimate = abs(quality_estimate)

        if statistics is None:
            statistics = self.calculate_statistics(subgroup, target, data, statistics)

        # special case of significance weighting where cover size and class balance parameter are both 1
        if 0 < self.subgroup_size_weight <= self.subgroup_class_balance_weight:
            subgroup_labels = data.loc[subgroup.representation, target.gt_name]

            if subgroup_labels.nunique() != 2:
                return 0  # class balance factor is 0 and that cannot change for any refinement

            subgroup_labels = subgroup_labels.groupby(by=lambda x: subgroup_labels[x]).count()
            min_label_count = min(subgroup_labels.iloc[0], subgroup_labels.iloc[1])
            quality_estimate *= (2 * min_label_count) ** self.subgroup_size_weight

        return quality_estimate

