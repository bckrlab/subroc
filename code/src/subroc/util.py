import math

import numpy as np
import pandas as pd
from pysubgroup import MinSupportConstraint, BitSetRepresentation, BitSet_Conjunction, constraints_satisfied, \
    EqualitySelector, Conjunction, IntervalSelector, NegatedSelector
from termcolor import cprint
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
import os
import pysubgroup as ps
from itertools import combinations

from subroc.preconditions import constraint_for_precondition_disjunction


def create_subgroup(data, selectors):
    data_representation = BitSetRepresentation(data, selectors)
    data_representation.patch_all_selectors()
    BitSet_Conjunction.n_instances = len(data)
    return BitSet_Conjunction(selectors)


def selected_indices_to_bits(_all: list, subset: list) -> list[bool]:
    bits = [False] * len(_all)

    all_i = 0
    subset_i = 0

    while all_i < len(_all) and subset_i < len(subset):
        if subset[subset_i] == _all[all_i]:
            bits[all_i] = True
            subset_i += 1

        all_i += 1

    return bits


def print_metric_colored(metric_name, metric_value):
    if metric_name.endswith("loss"):
        print(f"{metric_name}: {metric_value}")
        return

    if metric_name.endswith("error"):
        metric_score = 1 - metric_value
    else:
        metric_score = metric_value

    if 0 <= metric_score <= 1:
        on_color = None
        if metric_score <= 0.1:
            color = "red"
        elif metric_score <= 0.5:
            color = "light_red"
        elif metric_score <= 0.9:
            color = "light_green"
        else:
            color = "green"
    else:
        color = "black"
        on_color = "on_white"

    print(f"{metric_name}: ", end="")
    cprint(f"{metric_value}", color=color, on_color=on_color)


def round_exponent_up(x):
    if x >= 0:
        return 2 ** math.ceil(math.log2(x))
    else:
        return -1 * 2 ** math.floor(math.log2(-x))


def round_exponent_down(x):
    if x < 0:
        return -1 * 2 ** math.ceil(math.log2(-x))
    else:
        return 1 * 2 ** math.floor(math.log2(x))


def mask_value_continuous(array, value):
    mask = np.zeros_like(array, dtype=bool)

    for i in range(len(array)):
        for j in range(len(array[0])):
            if (0 <= i < len(array) - 1 and
                    (array[i][j] <= value < array[i + 1][j] or array[i][j] > value >= array[i + 1][j])):
                mask[i][j] = True

            if (0 <= j < len(array[0]) - 1 and
                    (array[i][j] <= value < array[i][j + 1] or array[i][j] > value >= array[i][j + 1])):
                mask[i][j] = True

    return np.ma.masked_array(array, mask=mask)


def plot_2d_function(
        func,
        x_min,
        x_max,
        x_num,
        y_min,
        y_max,
        y_num,
        x_name,
        y_name,
        title,
        fig_ax_tuple=None,
        show_value=None,
        interpolation="nearest",
        cmap="RdBu",
        vmin=None,
        vmax=None,
        show_contour_lines=True):
    aspect_ratio = (x_max - x_min) / (y_max - y_min)
    if aspect_ratio > 1:
        x_num_aspect = x_num
        y_num_aspect = max(math.floor(y_num / aspect_ratio), 1)
    else:
        x_num_aspect = max(math.floor(x_num * aspect_ratio), 1)
        y_num_aspect = y_num

    # compute function values
    results = np.ndarray((y_num_aspect, x_num_aspect))
    for row_i, y in enumerate(np.linspace(y_max, y_min, y_num_aspect)):
        for col_i, x in enumerate(np.linspace(x_min, x_max, x_num_aspect)):
            results[row_i, col_i] = func(x, y)

    if fig_ax_tuple is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax_tuple

    # change colormap to highlight values as indicated by show_value, if given
    colormap = plt.get_cmap(cmap)
    if show_value is not None:
        results = mask_value_continuous(results, value=show_value)
        colormap.set_bad(color="black")

    # overview plot with samples
    # plot function values as background
    extreme_value = max(abs(np.min(results)), abs(np.max(results)))
    if vmin is None: vmin = -extreme_value
    if vmax is None: vmax = extreme_value
    pos = ax.imshow(results, extent=(x_min, x_max, y_min, y_max),
                     vmin=vmin, vmax=vmax,
                     interpolation=interpolation, cmap=colormap)
    # fig.colorbar(pos, ax=ax)

    if show_contour_lines:
        ax.contour(
            results[::-1],  # np.sqrt(results[::-1]),
            extent=(x_min, x_max, y_min, y_max),
            colors='k', linewidths=0.1)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)

    return fig, ax, pos


def plot_2d_function_with_samples(
        samples,
        func,
        x_num,
        y_num,
        x_name,
        y_name,
        margin=0.1,
        show_value=None,
        interpolation="nearest",
        cmap="RdBu",
        samples_cmap="summer",
        vmin=None,
        vmax=None,
        show_contour_lines=True):
    x_min = min(samples[:, 0])
    x_max = max(samples[:, 0])

    x_range = x_max - x_min
    x_min -= margin * x_range
    x_max += margin * x_range

    y_min = min(samples[:, 1])
    y_max = max(samples[:, 1])

    y_range = y_max - y_min
    y_min -= margin * y_range
    y_max += margin * y_range

    x_min_extended = round_exponent_down(x_min)
    x_max_extended = round_exponent_up(x_max)
    y_min_extended = round_exponent_down(y_min)
    y_max_extended = round_exponent_up(y_max)

    max_abs_bound = max(abs(x_min_extended), abs(x_max_extended), abs(y_min_extended), abs(y_max_extended))
    x_min_extended = math.copysign(max_abs_bound, x_min_extended)
    x_max_extended = math.copysign(max_abs_bound, x_max_extended)
    y_min_extended = math.copysign(max_abs_bound, y_min_extended)
    y_max_extended = math.copysign(max_abs_bound, y_max_extended)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 4))

    plot_2d_function(
        func,
        x_min_extended,
        x_max_extended,
        x_num,
        y_min_extended,
        y_max_extended,
        y_num,
        x_name,
        y_name,
        title="subgroups on search data vs. validation data",
        fig_ax_tuple=(fig, ax1),
        show_value=show_value,
        interpolation=interpolation,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_contour_lines=show_contour_lines)

    # plot samples
    if len(samples[0]) > 2:
        sample_colors = samples[:, 2]
    else:
        sample_colors = 'b'

    ax1.scatter(samples[:, 0], samples[:, 1], c=sample_colors, cmap=samples_cmap, s=10, marker='x')

    _, _, pos = plot_2d_function(
        func,
        x_min,
        x_max,
        x_num,
        y_min,
        y_max,
        y_num,
        x_name,
        y_name,
        title="close-up of samples",
        fig_ax_tuple=(fig, ax2),
        show_value=show_value,
        interpolation=interpolation,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        show_contour_lines=show_contour_lines)
    fig.colorbar(pos, ax=ax2, pad=0.2)

    # plot samples
    scatter2 = ax2.scatter(samples[:, 0], samples[:, 1], c=sample_colors, cmap=samples_cmap, s=10, marker='x')

    fig.colorbar(scatter2)

    plt.show()


def copy_selector(selector):
    # Let the selector's class reconstruct the selector from string except
    # if it is an EqualitySelector with a boolean value.
    # (EqualitySelector does not support boolean values in reconstruction from string.)

    selector_str = selector.__str__().strip()
    if isinstance(selector, EqualitySelector):
        attribute_name, attribute_value = selector_str.split("==")
        if attribute_value == "True" or attribute_value == "False":
            return EqualitySelector(attribute_name, bool(attribute_value))

    return type(selector).from_str(selector_str)


def validation_and_original_scores(qf, qf_results, validation_data, dataset_meta, target, enable_progress_bar=True,
                                   return_overall_score=False, return_size_stats=False):
    # turn preconditions into data-specific constraints
    constraints = [MinSupportConstraint(1)]
    if hasattr(qf, "preconditions"):
        for precondition_disjunction in qf.preconditions:
            constraints.append(
                constraint_for_precondition_disjunction(precondition_disjunction, validation_data, dataset_meta))
    qf.constraints = constraints

    val_scores = []
    original_scores = []

    overall_val_score = None
    overall_original_score = qf.dataset_quality

    val_sizes = []
    original_sizes = []

    results_list = qf_results
    if enable_progress_bar:
        results_list = tqdm(results_list)

    for subgroup_result in results_list:
        subgroup = subgroup_result[1]

        selectors = [copy_selector(selector) for selector in subgroup.selectors]
        val_data_representation = BitSetRepresentation(validation_data, selectors)
        val_data_representation.patch_all_selectors()
        BitSet_Conjunction.n_instances = len(validation_data)
        val_subgroup = BitSet_Conjunction(selectors)

        qf.calculate_constant_statistics(validation_data, target)
        statistics = qf.calculate_statistics(val_subgroup, target, validation_data)
        overall_val_score = qf.dataset_quality

        if not constraints_satisfied(constraints, val_subgroup, statistics, validation_data):
            continue

        # val_scores.append(qf.evaluate(val_subgroup, target, validation_data, statistics))  # appends validation qf
        qf.evaluate(val_subgroup, target, validation_data, statistics)  # adds the validation metric value to the statistics
        val_scores.append(statistics["metric"])  # appends validation metric value
        # original_scores.append(subgroup_result[2]["metric"])  # appends original qf
        original_scores.append(subgroup_result[2]["metric"])  # appends original metric value

        val_sizes.append(statistics["size_sg"])
        original_sizes.append(subgroup_result[2]["size_sg"])

    if return_overall_score:
        return val_scores, original_scores, overall_val_score, overall_original_score

    if return_size_stats:
        return (val_scores,
                original_scores,
                np.average(val_sizes),
                np.average(original_sizes),
                np.var(val_sizes),
                np.var(original_sizes))

    return val_scores, original_scores


def generalization_errors(qf, qf_results, generalization_error, validation_data, dataset_meta, target):
    val_scores, original_scores, overall_val_score, overall_original_score = (
        validation_and_original_scores(qf, qf_results, validation_data, dataset_meta, target, return_overall_score=True))

    generalization_errors_with_subgroups = []
    for subgroup_result, val_score, original_score in zip(qf_results, val_scores, original_scores):
        subgroup_score = subgroup_result[0]
        generalization_errors_with_subgroups.append({"generalization_error": generalization_error(subgroup_score, val_score, original_scores, val_scores),
                                                     "val_score": val_score,
                                                     "original_score": original_score,
                                                     "overall_val_score": overall_val_score,
                                                     "overall_original_score": overall_original_score,
                                                     "subgroup_result": subgroup_result})

    return generalization_errors_with_subgroups


def generalization_errors_per_qf(
        results,
        qfs,
        generalization_error,
        validation_data,
        dataset_meta,
        target):
    errors_per_qf = {}

    print(f"Got {len(qfs)} qfs")
    for qf in qfs:
        qf_string = f"{qf.optimization_mode} {qf.__name__} {qf.subgroup_size_weight} {qf.subgroup_class_balance_weight}"
        generalization_errors_with_subgroups = generalization_errors(
            qf,
            [result[0] for result in results[qf_string]],
            generalization_error,
            validation_data,
            dataset_meta,
            target)

        errors_per_qf[qf_string] = generalization_errors_with_subgroups

    return errors_per_qf


def print_top_k_generalization_errors(generalization_errors_with_subgroups, k):
    generalization_errors_with_subgroups.sort()
    generalization_errors_with_subgroups.reverse()
    print(f"Top {k} subgroups with highest generalization error")
    for index, (generalization_error_value, val_score, subgroup_result) in enumerate(
            generalization_errors_with_subgroups[:k]):
        print(f"#{index + 1}: {generalization_error_value}, {val_score}, {subgroup_result}")
    print()

    generalization_errors_with_subgroups.reverse()
    print(f"Top {k} subgroups with lowest generalization error")
    for index, (generalization_error_value, val_score, subgroup_result) in enumerate(
            generalization_errors_with_subgroups[:k]):
        print(f"#{index + 1}: {generalization_error_value}, {val_score}, {subgroup_result}")
    print()


def plot_subgroups_on_generalization_error(
        generalization_errors_with_subgroups,
        generalization_error,
        cmap="RdBu",
        samples_cmap="summer",
        show_value=None,
        vmin=None,
        vmax=None,
        show_contour_lines=True):
    original_scores = [e[2][0] for e in generalization_errors_with_subgroups]
    val_scores = [e[1] for e in generalization_errors_with_subgroups]

    plot_2d_function_with_samples(
        np.array([[e[2][0], e[1], e[2][2]["size_sg"]] for e in generalization_errors_with_subgroups]),
        lambda original_score, val_score: generalization_error(original_score, val_score, original_scores, val_scores),
        100, 100,
        "search score", "validation score",
        show_value=show_value, cmap=cmap, samples_cmap=samples_cmap,
        vmin=vmin, vmax=vmax,
        show_contour_lines=show_contour_lines)

    average_generalization_error = np.average([generalization_error(e[2][0], e[1], original_scores, val_scores) for e in generalization_errors_with_subgroups])
    print(f"Average generalization error: {average_generalization_error}")


def plot_qf_errors_per_attribute(qf_errors, qf_name):
    errors_per_attribute = {}
    for error in qf_errors:
        if len(error[2][1].selectors) > 0:
            attribute_name = error[2][1].selectors[0].attribute_name
        else:
            attribute_name = "True"

        if attribute_name not in errors_per_attribute:
            errors_per_attribute[attribute_name] = [error]
        else:
            errors_per_attribute[attribute_name].append(error)

    for attribute_name, attribute_errors in errors_per_attribute.items():
        # plot subgroup scores magnitude dependent on subgroup size
        fig, ax = plt.subplots(figsize=(4, 3))
        scatter = ax.scatter(
            [e[2][2]['size_sg'] for e in attribute_errors],
            np.sqrt((np.array([e[2][0] for e in attribute_errors]) ** 2) + (np.array([e[1] for e in attribute_errors]) ** 2)),
            c=[np.sqrt(e[0]) for e in attribute_errors],
            cmap='hot',
            s=10)
        fig.colorbar(scatter, label="sqrt(generalization error)")

        ax.set(facecolor='0.8')
        ax.set_xlabel('subgroup size')
        ax.set_ylabel('vector length')
        ax.set_title(
            f"{qf_name} - {attribute_name} vector length in (orig score, val score)-space vs. subgroup size")

        plt.show()


def show_generalization_results_per_qf(
        errors_per_qf,
        k,
        generalization_error,
        generalization_error_2=None,
        cmap="RdBu",
        cmap_2=None,
        samples_cmap="summer",
        show_value=None,
        vmin=None,
        vmax=None,
        enable_plots=True,
        show_contour_lines=True):
    for qf_name, errors in errors_per_qf.items():
        print(f"{qf_name}")

        original_scores = [e[2][0] for e in errors]
        val_scores = [e[1] for e in errors]
        subgroups_pearsonr = scipy.stats.pearsonr(original_scores, val_scores)
        print(f"Correlation: {subgroups_pearsonr.statistic}")
        print(f"p-value: {subgroups_pearsonr.pvalue}")
        print()

        print_top_k_generalization_errors(errors, k)

        if enable_plots:
            plot_subgroups_on_generalization_error(
                errors,
                generalization_error,
                cmap,
                samples_cmap,
                show_value,
                vmin=vmin,
                vmax=vmax,
                show_contour_lines=show_contour_lines
            )

            if generalization_error_2 is not None and cmap_2 is not None:
                plot_subgroups_on_generalization_error(errors, generalization_error_2, cmap_2, samples_cmap, show_value, vmin=vmin, vmax=vmax)

        # plot_qf_errors_per_attribute(errors, qf_name)


def str_to_bool(s: str) -> bool:
    if s.lower() in ['y', 'yes', 't', 'true', 'on', '1']:
        return True
    elif s.lower() in ['n', 'no', 'f', 'false', 'off', '0']:
        return False

    raise ValueError(f"'{s}' is not a valid string representation of a boolean value")


def from_str_EqualitySelector(s: str) -> ps.EqualitySelector:
    s = s.strip()
    attribute_name, attribute_value = s.split("==")
    if attribute_value[0] == "'" and attribute_value[-1] == "'":
        if attribute_value.startswith("'b'") and attribute_value.endswith("''"):
            attribute_value = str.encode(attribute_value[3:-2])
        else:
            attribute_value = attribute_value[1:-1]
    try:
        attribute_value = int(attribute_value)
    except ValueError:
        try:
            attribute_value = float(attribute_value)
        except ValueError:
            try:
                attribute_value = str_to_bool(attribute_value)
            except ValueError:
                pass
    return EqualitySelector(attribute_name, attribute_value)


def from_str_Conjunction(s: str) -> ps.Conjunction:
    if s.strip() == "Dataset":
        return Conjunction([])
    selector_strings = s.split(" AND ")
    selectors = []
    for selector_string in selector_strings:
        selector_string = selector_string.strip()
        if "==" in selector_string:
            selectors.append(from_str_EqualitySelector(selector_string))
        else:
            selectors.append(IntervalSelector.from_str(selector_string))
    return Conjunction(selectors)


def restore_categorical(pattern: ps.Conjunction, data: pd.DataFrame) -> ps.Conjunction:
    new_selectors = []

    for selector in pattern.selectors:
        new_selector = selector

        if isinstance(selector, EqualitySelector) and isinstance(selector.attribute_value, bool):
            # Suspect that this selector refers to a dummy attribute, which is based on a nominal attribute and
            # represents equality to a value of that original nominal attribute.
            for column in data.columns.values.tolist():
                if selector.attribute_name.startswith(column):
                    new_selector = EqualitySelector(column, selector.attribute_name[len(column) + 1:])
                    if not selector.attribute_value:
                        new_selector = NegatedSelector(new_selector)

        new_selectors.append(new_selector)

    return ps.Conjunction(new_selectors)


def prepend_experiment_output_path(path):
    return path if os.environ.get("EXPERIMENT_OUTPUT_PATH") is None else (os.environ.get("EXPERIMENT_OUTPUT_PATH") + "/" + path)


def prepend_experiment_definition_path(path):
    definitions_path = os.environ.get("EXPERIMENT_DEFINITIONS_PATH")
    experiment_name = os.environ.get("EXPERIMENT_NAME")
    if definitions_path is None or experiment_name is None:
        return path

    return definitions_path + "/" + experiment_name + "/" + path


def iou(a: set, b: set):
    return len(a.intersection(b)) / len(a.union(b))


def mean_pairwise_iou(result_df, data):
    pairwise_ious = []

    for result_a, result_b in combinations(result_df.itertuples(), 2):
        # recreate the pysubgroup objects for the subgroups with a representation for the dataset
        sel_conjunction_a = from_str_Conjunction(result_a.pattern)
        subgroup_a = create_subgroup(data, sel_conjunction_a.selectors)

        sel_conjunction_b = from_str_Conjunction(result_b.pattern)
        subgroup_b = create_subgroup(data, sel_conjunction_b.selectors)
        
        # get indices of the covered instances of each subgroup
        idx_a = np.nonzero(subgroup_a.representation)[0]
        idx_b = np.nonzero(subgroup_b.representation)[0]

        pairwise_ious.append(iou(set(idx_a), set(idx_b)))
    
    return np.mean(pairwise_ious)


def print_proc_status():
    pid = os.getpid()
    print("os.getpid: ", pid)
    with open(f"/proc/{pid}/status", "r") as file:
        print(file.read())

