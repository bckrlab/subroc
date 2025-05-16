import multiprocessing
from enum import Enum

import numpy as np
import pandas as pd
import pysubgroup as ps

from subroc.preconditions import Precondition
from subroc.selectors import InformedRandomSelector, NegativeClassCountRandomSelector
from subroc.util import create_subgroup


class OptimizationMode(Enum):
    Maximal = 0,
    """
    Configures classification quality functions for finding subgroups for which the classifier does not work.
    """

    Minimal = 1,
    """
    Configures classification quality functions for finding subgroups for which the classifier does work.
    """

    Exceptional = 2
    """
    Configures classification quality functions for finding subgroups for which the classifier works best 
    or worst. (most exceptional performance)
    """


class PredictionType(str, Enum):
    CLASSIFICATION_HARD = "Hard Classification"
    CLASSIFICATION_SOFT = "Soft Classification"


def label_balance_fraction(labels: pd.Series):
    if labels.nunique() != 2:
        return 0

    labels = labels.groupby(by=lambda x: labels[x]).count()

    if labels.iloc[1] == 0:
        return np.inf

    result = labels.iloc[0] / labels.iloc[1]

    if result > 1:
        result = 1 / result

    return result


class RandomSamplingTask:
    def __init__(self, qf, data, target):
        self.qf = qf
        self.data = data
        self.target = target

    def compute_random_sample_value(self, selector):
        subgroup = create_subgroup(self.data, selector)
        statistics = self.qf.calculate_statistics(subgroup, self.target, self.data)
        return self.qf.evaluate(subgroup, self.target, self.data, statistics)


class BaseQF(ps.BoundedInterestingnessMeasure):
    def __init__(
            self,
            name: str,
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
            check_own_constraints: bool = False):
        self.__name__ = name

        self.optimization_mode = optimization_mode

        # put preconditions into CNF form
        if preconditions is None:
            self.preconditions = []
        elif isinstance(preconditions, Precondition):
            self.preconditions = [[preconditions]]
        elif isinstance(preconditions, list) and isinstance(preconditions[0], Precondition):
            self.preconditions = [[precondition] for precondition in preconditions]
        else:
            self.preconditions = preconditions

        self.check_own_constraints = check_own_constraints

        self.subgroup_class_balance_weight = subgroup_class_balance_weight
        self.subgroup_size_weight = subgroup_size_weight

        self.relative_quality = relative_quality

        # random sampling for significance
        self.random_sampling_p_value_factor = random_sampling_p_value_factor
        self.random_sampling_normalization = random_sampling_normalization
        self.random_sampling_statistics = None  # List of [mean, std] lists if used
        self.num_random_samples = num_random_samples

        self.random_sampling_distributions = {}

        self.seed = seed
        self.np_rng = np.random.default_rng(seed)

        self.max_random_sampling_tries = max_random_sampling_tries
        if constraints is None:
            constraints = []
        self.constraints = constraints

        self.scores_sorted = None
        self.gt_sorted_by_score = None
        self.sorted_to_original_index = None
        self.has_constant_statistics = None

        self.calculate_statistics_invocation_count = 0

    def _compute_random_sampling_distribution(self, subgroup_size, subgroup_labels, target, data):
        negatives_count = subgroup_size - np.sum(subgroup_labels)

        invocatoin_key = f"{subgroup_size}:{negatives_count}"
        if invocatoin_key in self.random_sampling_distributions:
            return self.random_sampling_distributions[invocatoin_key]

        positive_class_indices = data[data[target.gt_name] == 1].index
        negative_class_indices = data[data[target.gt_name] == 0].index

        # save current random sampling normalization switch value to restore it later and disable it
        random_sampling_p_value_factor_value = self.random_sampling_p_value_factor
        self.random_sampling_p_value_factor = False

        selectors = [[NegativeClassCountRandomSelector(self.np_rng, self.seed, positive_class_indices, 
                                                        negative_class_indices, subgroup_size, negatives_count,
                                                        False)] for _ in range(self.num_random_samples)]

        random_sampling_task = RandomSamplingTask(self, data, target)
        
        sample_values = [random_sampling_task.compute_random_sample_value(selector) for selector in selectors]

        # reset random sampling normalization switch
        self.random_sampling_p_value_factor = random_sampling_p_value_factor_value

        self.random_sampling_distributions[invocatoin_key] = sample_values

        return sample_values
    
    def _compute_random_sampling_p_value(self, subgroup_size, subgroup_labels, target, data, qf_value):
            sample_values = self._compute_random_sampling_distribution(subgroup_size, subgroup_labels, target, data)
            
            # This automatically implements the correctly sided test in case of OptimizationMode Maximal or Minimal and
            # the two-sided test in case of OptimizationMode Exceptional as long as self.evaluate implements the
            # sign adaptation to these OptimizationModes. The evaluate method of ClassifierMetricQF is an example for that.
            num_at_least_as_extreme_values = sum([sample_value >= qf_value for sample_value in sample_values])

            return num_at_least_as_extreme_values / self.num_random_samples

    def _compute_random_sampling_statistics(self, sample_size, target, data):
        if self.random_sampling_statistics[sample_size, 0] is not None and \
                self.random_sampling_statistics[sample_size, 1] is not None:
            # Do not run this function twice for the same sample size.
            return

        # save current random sampling normalization switch value to restore it later and disable it
        random_sampling_normalization_value = self.random_sampling_normalization
        self.random_sampling_normalization = False

        selectors = [[InformedRandomSelector(self, target, self.np_rng, self.seed, data, sample_size, self.constraints,
                                            replace=False)] for _ in range(self.num_random_samples)]
        random_sampling_task = RandomSamplingTask(self, data, target)

        with multiprocessing.Pool(4) as pool:
            sample_values = pool.map(random_sampling_task.compute_random_sample_value, selectors)

        # reset random sampling normalization switch
        self.random_sampling_normalization = random_sampling_normalization_value

        self.random_sampling_statistics[sample_size, 0] = np.mean(sample_values)
        self.random_sampling_statistics[sample_size, 1] = np.std(sample_values)

    def apply_significance_weighting(
            self,
            subgroup,
            target,
            data: pd.DataFrame,
            qf_value: float,
            subgroup_class_balance_weight: float = None,
            subgroup_size_weight: float = None,
            statistics: dict = None):
        if subgroup_class_balance_weight is None:
            subgroup_class_balance_weight = self.subgroup_class_balance_weight
        if subgroup_size_weight is None:
            subgroup_size_weight = self.subgroup_size_weight

        subgroup_labels = data.loc[subgroup.representation, target.gt_name]
        subgroup_label_balance_fraction = label_balance_fraction(subgroup_labels)

        subgroup_size = len(subgroup_labels)

        significance_factor = ((subgroup_label_balance_fraction ** subgroup_class_balance_weight)
                               * (subgroup_size ** subgroup_size_weight))

        if self.random_sampling_p_value_factor:
            qf_value *= significance_factor
            subgroup_p_value = self._compute_random_sampling_p_value(subgroup_size, subgroup_labels, target, data, qf_value)
            qf_value = qf_value / max(subgroup_p_value, 0.0000001)
        elif self.random_sampling_normalization:
            if self.random_sampling_statistics is None:
                self.random_sampling_statistics = np.full((len(data) + 1, 2), None)

            self._compute_random_sampling_statistics(subgroup_size, target, data)

            sampling_mean = self.random_sampling_statistics[subgroup_size, 0]
            sampling_std = self.random_sampling_statistics[subgroup_size, 1]

            epsilon = 0.000001
            qf_value = (qf_value - sampling_mean) / (sampling_std + epsilon)
        else:
            qf_value *= significance_factor

        return qf_value

    def calculate_constant_statistics(self, data: pd.DataFrame, target):
        """ calculate_constant_statistics
            This function is called once for every search execution,
            it should do any preparation that is necessary prior to an execution.
        """
        dataset_sorted_by_score = data.sort_values(target.score_name)
        self.scores_sorted = dataset_sorted_by_score.loc[:, target.score_name]
        self.gt_sorted_by_score = dataset_sorted_by_score.loc[:, target.gt_name]
        self.sorted_to_original_index = [index for index, _ in dataset_sorted_by_score.iterrows()]

        self.has_constant_statistics = True

    def calculate_statistics(self, subgroup, target, data: pd.DataFrame, statistics=None):
        """ calculates necessary statistics
            this statistics object is passed on to the evaluate
            and optimistic_estimate functions
        """
        self.calculate_statistics_invocation_count += 1

        return {"size_sg": sum(subgroup.representation)}

    def evaluate(self, subgroup, target, data: pd.DataFrame, statistics=None):
        """ return the quality calculated from the statistics """
        return np.inf

    def optimistic_estimate(self, subgroup, target, data: pd.DataFrame, statistics=None):
        """ returns optimistic estimate
            if one is available return it otherwise infinity"""
        return np.inf
