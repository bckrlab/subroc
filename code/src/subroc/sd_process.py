import math
from typing import Union

import numpy as np
import pandas as pd
import pysubgroup as ps

from subroc.quality_functions.base_qf import PredictionType, label_balance_fraction
from subroc.quality_functions.bounded_generalization_aware_qf import BoundedGeneralizationAwareQF
from subroc.preconditions import constraint_for_precondition_disjunction
from subroc.selectors import create_selectors
from subroc.datasets.metadata import DatasetMetadata
from subroc.util import create_subgroup

UnpackedGroupedResult = list[list[tuple[float, ps.BitSet_Conjunction, dict[str, any]]]]


def ceil(x: float, decimals: int = 0):
    """
    Return the ceiling of x at the given decimal.

    The ceil of the scalar x at decimal d is (uncommonly) defined as
    the fraction i/(10^d) for the smallest integer i, such that i >= x * 10^d.
    The usual definition of the ceiling is that of ceil(x, 0).
    """
    return np.ceil(x * (10 ** decimals)) / 10**decimals


def add_subgroup_representation_if_necessary(result: ps.SubgroupDiscoveryResult, data) -> ps.SubgroupDiscoveryResult:
    results_with_representation = []

    for subgroup_result in result.results:
        if not hasattr(subgroup_result[1], "representation"):
            subgroup_result = (subgroup_result[0], create_subgroup(data, subgroup_result[1]._selectors), subgroup_result[2])

        results_with_representation.append(subgroup_result)

    return ps.SubgroupDiscoveryResult(results_with_representation, result.task)


def filter_highest_quality(result: ps.SubgroupDiscoveryResult) -> ps.SubgroupDiscoveryResult:
    if len(result.results) == 0:
        return result

    filtered_result = []
    highest_quality = ceil(result.results[0][0], 3)

    for subgroup_result in result.results:
        if ceil(subgroup_result[0], 3) < highest_quality:
            continue

        filtered_result.append(subgroup_result)

    return ps.SubgroupDiscoveryResult(filtered_result, result.task)


def filter_highest_quality_per_attribute(result: ps.SubgroupDiscoveryResult) -> list[ps.SubgroupDiscoveryResult]:
    filtered_result = {}

    for subgroup_result in result.results:
        if subgroup_result[1].depth != 1:
            continue

        filter_attribute = subgroup_result[1].selectors[0].attribute_name
        subgroup_quality = ceil(subgroup_result[0], 3)

        if (filter_attribute not in filtered_result.keys() or
                subgroup_quality > ceil(filtered_result[filter_attribute][0][0], 3)):
            filtered_result[filter_attribute] = [subgroup_result]
        elif (filter_attribute in filtered_result.keys() and
                subgroup_quality == ceil(filtered_result[filter_attribute][0][0], 3)):
            filtered_result[filter_attribute].append(subgroup_result)

    return [ps.SubgroupDiscoveryResult(list(filtered_result_value), result.task)
            for filtered_result_value in filtered_result.values()]


def average_label_balance_fraction(
        data: pd.DataFrame,
        dataset_meta: DatasetMetadata,
        results: UnpackedGroupedResult):
    label_balance_fraction_sum = 0
    subgroup_count = 0

    if data[dataset_meta.gt_name].nunique() == 2:
        for equal_subgroups in results:
            for subgroup in equal_subgroups:
                subgroup_count += 1

                subgroup_labels = data.loc[subgroup[1].representation, dataset_meta.gt_name]
                label_balance_fraction_sum += label_balance_fraction(subgroup_labels)

    if subgroup_count == 0:
        return None

    return label_balance_fraction_sum / subgroup_count


def print_subgroups(
        data: pd.DataFrame,
        dataset_meta: DatasetMetadata,
        results: UnpackedGroupedResult,
        result_set_size: int,
        qf: any,
        qf_name: str):
    label_balance_fraction_avg = average_label_balance_fraction(data, dataset_meta, results)

    print()
    print(qf_name)
    print(f"overall metric value: {qf.metric(qf.gt_sorted_by_score.to_numpy(), qf.scores_sorted.to_numpy())}")
    print(f"average label balance fraction: {label_balance_fraction_avg}")
    print(
        f"top {result_set_size} {qf.optimization_mode.name} {qf_name} subgroups after filtering:")

    for equal_subgroups in results:
        subgroup_labels_list = []
        for subgroup in equal_subgroups:
            subgroup_labels = data.loc[subgroup[1].representation, dataset_meta.gt_name]
            subgroup_labels = subgroup_labels.groupby(by=lambda x: subgroup_labels[x]).count()
            subgroup_labels_list.append({label: subgroup_labels[label] for label in subgroup_labels.index})

        print(
            f"    {qf_name}: {round(equal_subgroups[0][0], 3):08.3f}, rule(s): "
            f"({[str(subgroup[1]) + ', true labels: ' + str(labels) for subgroup, labels in zip(equal_subgroups, subgroup_labels_list)]})")


def correct_prediction_type(data: pd.Series, data_pred_type: PredictionType, qf_pred_type: PredictionType):
    if (data_pred_type == PredictionType.CLASSIFICATION_SOFT
            and qf_pred_type == PredictionType.CLASSIFICATION_HARD):
        return np.round(data)

    return data


def run_subgroup_discovery(
        quality_functions: list,
        qf_names: list[str],
        data: pd.DataFrame,
        dataset_meta: DatasetMetadata,
        result_set_size: int = 1000,
        ignore_null: bool = False,
        summarize: bool = True,
        tqdm_function: any = None,
        target: any = None,
        constraints: list[any] = None,
        search_space: list[any] = None,
        depth: int = 1,
        min_quality: float = -1,
        enable_filter_highest_quality: bool = False,
        enable_filter_highest_quality_per_attribute: bool = False,
        sd_algorithm: any = ps.Apriori(),
        enable_generalization_awareness: bool = False) -> Union[None, dict[str, UnpackedGroupedResult]]:
    if isinstance(sd_algorithm, ps.Apriori):
        sd_algorithm.use_vectorization = False

    if summarize:
        print(f"--- Dataset: {dataset_meta.name} ---")

    if target is None:
        raise ValueError("Argument 'target' may not be None.")

    if constraints is None:
        constraints = [ps.MinSupportConstraint(math.ceil(len(data) / 100))]

    if search_space is None:
        search_space = create_selectors(
                data,
                nbins=10,
                ignore=[dataset_meta.gt_name, dataset_meta.score_name],
                ignore_null=ignore_null
            )

    if summarize:
        print("--- Search Space ---")
        print(search_space)

    qf_index_list = range(len(quality_functions))
    if tqdm_function is not None:
        qf_index_list = tqdm_function(qf_index_list, desc="quality functions")

    results_per_qf = {}
    for qf_index in qf_index_list:
        qf = quality_functions[qf_index]
        qf_name = qf_names[qf_index]

        iteration_constraints = constraints.copy()
        # turn preconditions into data-specific constraints
        if hasattr(qf, "preconditions"):
            for precondition_disjunction in qf.preconditions:
                iteration_constraints.append(
                    constraint_for_precondition_disjunction(precondition_disjunction, data, dataset_meta))
        qf.constraints = iteration_constraints

        # handle special case of the non-DFS implementations
        if not isinstance(sd_algorithm, ps.DFS):
            qf.check_own_constraints = True

        # change dataset prediction type to hard if needed
        if dataset_meta.prediction_type is not None:
            data.loc[:, dataset_meta.score_name] = correct_prediction_type(
                data[dataset_meta.score_name],
                dataset_meta.prediction_type,
                qf.prediction_type)
        
        # make qf generalization aware
        task_qf = qf
        if enable_generalization_awareness:
            task_qf = BoundedGeneralizationAwareQF(task_qf)

        # define task
        task = ps.SubgroupDiscoveryTask(
            data,
            target,
            search_space,
            result_set_size=result_set_size,
            depth=depth,
            min_quality=min_quality,
            qf=task_qf,
            constraints=iteration_constraints)

        # get result
        result = sd_algorithm.execute(task)

        if not isinstance(sd_algorithm, ps.DFS):
            result = add_subgroup_representation_if_necessary(result, data)

        # filter result
        if enable_filter_highest_quality:
            result = filter_highest_quality(result)

        if enable_filter_highest_quality_per_attribute:
            filtered_result = filter_highest_quality_per_attribute(result)
        else:
            filtered_result = [
                ps.SubgroupDiscoveryResult([subgroup_result], result.task) for subgroup_result in result.results
            ]

        unpacked_filtered_results = [result_for_attribute.results for result_for_attribute in filtered_result]
        results_per_qf[qf_name] = unpacked_filtered_results

        if summarize:
            if len(unpacked_filtered_results) == 0:
                print(f"no {qf.optimization_mode.name} {qf_name} subgroups found")
            else:
                print_subgroups(data, dataset_meta, unpacked_filtered_results, result_set_size, qf, qf_name)

    return results_per_qf

