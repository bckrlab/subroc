# This is a derivation of the selectors implementation in https://github.com/flemmerich/pysubgroup

import numpy as np
import pysubgroup as ps
from pysubgroup import MinSupportConstraint

from subroc.constraints import ContainsValueConstraint
from subroc.util import create_subgroup


class ListSelector(ps.SelectorBase):
    def __init__(self, indices):
        self._included_indices = indices

        self._hash = None
        self._query = None
        self._string = None
        self.set_descriptions(indices)

        super().__init__()

    def covers(self, data_instance):
        return [index in self._included_indices for index in data_instance.index]

    def set_descriptions(self, indices, *args, **kwargs):
        query = str(indices)
        self._hash, self._query, self._string = (hash(query), query, query)

    @property
    def selectors(self):
        return (self,)


def _get_unsatisfied_constraints(included_indices, constraints, qf, target, data):
    selectors = [ListSelector(included_indices)]
    subgroup = create_subgroup(data, selectors)
    statistics = qf.calculate_statistics(subgroup, target, data)
    return constraints[
        [not constraint.is_satisfied(subgroup, statistics, data) for constraint in constraints]
    ]


class InformedRandomSelector(ps.SelectorBase):
    def __init__(self, qf, target, np_rng, seed, data, size, constraints, replace=True):
        self._included_indices = self._select_included_indices(qf, target, np_rng, data, size, constraints, replace)

        self._hash = None
        self._query = None
        self._string = None
        self.set_descriptions(qf, target, np_rng, seed, data, size, constraints, replace)

        super().__init__()

    def _select_included_indices(self, qf, target, np_rng, data, size, constraints, replace):
        """Randomly select (size) many indices from data, such that they satisfy the constraints.
        Only ensure that constraints are satisfied at the latest possible step, right before the number of indices
        to pick would be exhausted."""
        data_indices = data.index
        included_indices = []

        constraints = np.array(constraints)
        unsatisfied_constraints = constraints
        number_of_rows_left = size
        while number_of_rows_left > len(unsatisfied_constraints):
            number_of_new_indices = number_of_rows_left - len(unsatisfied_constraints)
            new_selected_indices = np_rng.choice(
                data_indices,
                number_of_new_indices,
                replace=replace)

            included_indices.extend(new_selected_indices)
            data_indices = data_indices.difference(new_selected_indices)

            number_of_rows_left = size - len(included_indices)
            unsatisfied_constraints = _get_unsatisfied_constraints(included_indices, constraints, qf, target, data)

            if number_of_rows_left == len(unsatisfied_constraints):
                while len(unsatisfied_constraints) != 0:
                    constraint_to_satisfy = unsatisfied_constraints[0]

                    # Pick a row such that the selected indices together with that row satisfy the constraint.
                    if isinstance(constraint_to_satisfy, ContainsValueConstraint):
                        possible_rows = data.loc[data[constraint_to_satisfy.attribute_name] == constraint_to_satisfy.value]

                        if possible_rows.empty:
                            raise ValueError(f"data does not contain necessary rows for {constraint_to_satisfy}")

                        included_indices.extend(np_rng.choice(possible_rows.index, 1))
                        number_of_rows_left = size - len(included_indices)
                    elif isinstance(constraint_to_satisfy, MinSupportConstraint):
                        if size < constraint_to_satisfy.min_support:
                            raise ValueError(f"{constraint_to_satisfy} needs more support than {size} (size) allows for")

                        included_indices.extend(np_rng.choice(
                            data.index,
                            constraint_to_satisfy.min_support - len(included_indices)))
                        number_of_rows_left = size - len(included_indices)
                    else:
                        raise ValueError(f"InformedRandomSelector is not implemented for constraint of type {type(constraint_to_satisfy)}.")

                    unsatisfied_constraints = _get_unsatisfied_constraints(included_indices, constraints, qf, target,
                                                                           data)

                    if len(unsatisfied_constraints) < number_of_rows_left:
                        break

        return included_indices

    def covers(self, data_instance):
        return [index in self._included_indices for index in data_instance.index]

    def set_descriptions(self, qf, target, np_rng, seed, data, size, constraints, replace, *args, **kwargs):
        replace_str = "with replacement" if replace else "without replacement"
        query = (f"{size} rows picked by {str(np_rng)} with seed {seed}, {replace_str}, satisfying {str(constraints)} "
                 f"for qf {str(qf)} with target {str(target)}")
        self._hash, self._query, self._string = (hash(query), query, query)

    @property
    def selectors(self):
        return (self,)


class NegativeClassCountRandomSelector(ps.SelectorBase):
    def __init__(self, np_rng, seed, positive_class_indices, negative_class_indices, size, negative_class_count, replace):
        included_positive_class_indices = np_rng.choice(positive_class_indices, size - negative_class_count, replace)
        included_negative_class_indices = np_rng.choice(negative_class_indices, negative_class_count, replace)
        self._included_indices = [*included_positive_class_indices, *included_negative_class_indices]

        self._hash = None
        self._query = None
        self._string = None
        self.set_descriptions(np_rng, seed, size, negative_class_count, replace)

        super().__init__()
    
    def covers(self, data_instance):
        return data_instance.index.isin(self._included_indices)

    def set_descriptions(self, np_rng, seed, size, negative_class_count, replace, *args, **kwargs):
        replace_str = "with replacement" if replace else "without replacement"
        query = f"{size} rows with negative class count {negative_class_count} picked by {np_rng.__str__()} with seed {seed}, {replace_str}"
        self._hash, self._query, self._string = (hash(query), query, query)

    @property
    def selectors(self):
        return (self,)


class RandomSelector(ps.SelectorBase):
    def __init__(self, np_rng, seed, indices, size, replace):
        self._included_indices = np_rng.choice(indices, size, replace)

        self._hash = None
        self._query = None
        self._string = None
        self.set_descriptions(np_rng, seed, size, replace)

        super().__init__()

    def covers(self, data_instance):
        return [index in self._included_indices for index in data_instance.index]

    def set_descriptions(self, np_rng, seed, size, replace, *args, **kwargs):
        replace_str = "with replacement" if replace else "without replacement"
        query = f"{size} rows picked by {np_rng.__str__()} with seed {seed}, {replace_str}"
        self._hash, self._query, self._string = (hash(query), query, query)

    @property
    def selectors(self):
        return (self,)


# Including the lower bound, excluding the upper_bound
class IntervalSelector(ps.SelectorBase):
    def __init__(
            self,
            attribute_name,
            lower_bound,
            upper_bound,
            selector_name=None,
            include_lower=True,
            include_upper=False
    ):
        assert lower_bound < upper_bound
        # this is kind of redundant due to `__new__` and `set_descriptions`
        self._attribute_name = attribute_name
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._include_lower = include_lower
        self._include_upper = include_upper
        self.selector_name = selector_name
        self.set_descriptions(
            attribute_name,
            lower_bound,
            upper_bound,
            selector_name,
            include_lower,
            include_upper
        )

        super().__init__()

    @property
    def attribute_name(self):
        return self._attribute_name

    @property
    def lower_bound(self):
        return self._lower_bound

    @property
    def upper_bound(self):
        return self._upper_bound

    @property
    def include_lower(self):
        return self._include_lower

    @property
    def include_upper(self):
        return self._include_upper

    def _check_lower(self, val):
        if self._include_lower:
            return val >= self.lower_bound
        return val > self.lower_bound

    def _check_upper(self, val):
        if self._include_upper:
            return val <= self.upper_bound
        return val < self.upper_bound

    def covers(self, data_instance):
        val = data_instance[self.attribute_name].to_numpy()
        return np.logical_and(self._check_lower(val), self._check_upper(val))

    def __repr__(self):
        return self._query

    def __hash__(self):
        return self._hash

    def __str__(self):
        return self._string

    @classmethod
    def compute_descriptions(
        cls, attribute_name, lower_bound, upper_bound, selector_name=None, include_lower=True, include_upper=False
    ):
        if selector_name is None:
            _string = cls.compute_string(
                attribute_name,
                lower_bound,
                upper_bound,
                rounding_digits=2,
                include_lower=include_lower,
                include_upper=include_upper
            )
        else:
            _string = selector_name
        _query = cls.compute_string(
            attribute_name,
            lower_bound,
            upper_bound,
            rounding_digits=None,
            include_lower=include_lower,
            include_upper=include_upper
        )
        _hash = hash(_query)
        return (_hash, _query, _string)

    def set_descriptions(
        self, attribute_name, lower_bound, upper_bound, selector_name=None, include_lower=True, include_upper=False
    ):  # pylint: disable=arguments-differ
        self._hash, self._query, self._string = IntervalSelector.compute_descriptions(
            attribute_name,
            lower_bound,
            upper_bound,
            selector_name=selector_name,
            include_lower=include_lower,
            include_upper=include_upper
        )

    @classmethod
    def compute_string(
            cls,
            attribute_name,
            lower_bound,
            upper_bound,
            rounding_digits,
            include_lower=True,
            include_upper=False
    ):
        if rounding_digits is None:
            formatter = "{}"
        else:
            formatter = "{0:." + str(rounding_digits) + "f}"
        ub = upper_bound
        lb = lower_bound
        if ub % 1:
            ub = formatter.format(ub)
        if lb % 1:
            lb = formatter.format(lb)

        if lower_bound == float("-inf") and upper_bound == float("inf"):
            repre = attribute_name + " = anything"
        elif lower_bound == float("-inf"):
            repre = attribute_name + ("<=" if include_upper else "<") + str(ub)
        elif upper_bound == float("inf"):
            repre = attribute_name + (">=" if include_lower else ">") + str(lb)
        else:
            repre = (attribute_name + ": " +
                     ("]" if include_lower else "[") + str(lb) + ":" + str(ub) + ("[" if include_upper else "]"))
        return repre

    @staticmethod
    def from_str(s):
        s = s.strip()
        if s.endswith(" = anything"):
            return IntervalSelector(
                s[: -len(" = anything")], float("-inf"), float("+inf")
            )
        if "<=" in s:
            attribute_name, ub = s.split("<=")
            try:
                return IntervalSelector(attribute_name.strip(), float("-inf"), int(ub), include_upper=True)
            except ValueError:
                return IntervalSelector(attribute_name.strip(), float("-inf"), float(ub), include_upper=True)
        elif "<" in s:
            attribute_name, ub = s.split("<")
            try:
                return IntervalSelector(attribute_name.strip(), float("-inf"), int(ub), include_upper=False)
            except ValueError:
                return IntervalSelector(
                    attribute_name.strip(), float("-inf"), float(ub), include_upper=False
                )
        if ">=" in s:
            attribute_name, lb = s.split(">=")
            try:
                return IntervalSelector(attribute_name.strip(), int(lb), float("inf"), include_lower=True)
            except ValueError:
                return IntervalSelector(attribute_name.strip(), float(lb), float("inf"), include_lower=True)
        elif ">" in s:
            attribute_name, lb = s.split(">")
            try:
                return IntervalSelector(attribute_name.strip(), int(lb), float("inf"), include_lower=False)
            except ValueError:
                return IntervalSelector(
                    attribute_name.strip(), float(lb), float("inf"), include_lower=False
                )
        if s.count(":") == 2:
            attribute_name, lb, ub = s.split(":")
            lower_end = lb[0]
            upper_end = ub[-1]
            lb = lb.strip()[1:]
            ub = ub.strip()[:-1]

            if lower_end == "[":
                include_lower = False
            else:
                include_lower = True
            if upper_end == "[":
                include_upper = True
            else:
                include_upper = False

            try:
                return IntervalSelector(
                    attribute_name.strip(),
                    int(lb),
                    int(ub),
                    include_lower=include_lower,
                    include_upper=include_upper
                )
            except ValueError:
                return IntervalSelector(
                    attribute_name.strip(),
                    float(lb),
                    float(ub),
                    include_lower=include_lower,
                    include_upper=include_upper
                )
        else:
            raise ValueError(f"string {s} could not be converted to IntervalSelector")

    @property
    def selectors(self):
        return (self,)


def create_selectors(
        data,
        nbins=5,
        intervals_only=True,
        ignore=None,
        ignore_null=False,
        include_lower=True,
        include_upper=False
):
    if ignore is None:
        ignore = []
    sels = create_nominal_selectors(data, ignore, ignore_null)
    sels.extend(create_numeric_selectors(
        data,
        nbins,
        intervals_only,
        ignore=ignore,
        include_lower=include_lower,
        include_upper=include_upper
    ))
    return sels


def create_nominal_selectors(data, ignore=None, ignore_null=False):
    if ignore is None:
        ignore = []
    nominal_selectors = []
    # for attr_name in [
    #       x for x in data.select_dtypes(exclude=['number']).columns.values
    #       if x not in ignore]:
    #    nominal_selectors.extend(
    #       create_nominal_selectors_for_attribute(data, attr_name))
    nominal_dtypes = data.select_dtypes(exclude=["number"])
    dtypes = data.dtypes
    # print(dtypes)
    for attr_name in [x for x in nominal_dtypes.columns.values if x not in ignore]:
        nominal_selectors.extend(
            create_nominal_selectors_for_attribute(data, attr_name, dtypes, ignore_null)
        )
    return nominal_selectors


def create_nominal_selectors_for_attribute(data, attribute_name, dtypes=None, ignore_null=False):
    import pandas as pd  # pylint: disable=import-outside-toplevel

    nominal_selectors = []
    for val in pd.unique(data[attribute_name]):
        if ignore_null and pd.isna(val):
            continue

        nominal_selectors.append(ps.EqualitySelector(attribute_name, val))
    # setting the is_bool flag for selector
    if dtypes is None:
        dtypes = data.dtypes
    if dtypes[attribute_name] == "bool":
        for s in nominal_selectors:
            s.is_bool = True
    return nominal_selectors


def create_numeric_selectors(
        data,
        nbins=5,
        intervals_only=True,
        weighting_attribute=None,
        ignore=None,
        ignore_null=False,
        include_lower=True,
        include_upper=False
):
    if ignore is None:
        ignore = []  # pragma: no cover
    numeric_selectors = []
    for attr_name in [
        x
        for x in data.select_dtypes(include=["number"]).columns.values
        if x not in ignore
    ]:
        numeric_selectors.extend(
            create_numeric_selectors_for_attribute(
                data,
                attr_name,
                nbins,
                intervals_only,
                weighting_attribute,
                ignore_null,
                include_lower,
                include_upper
            )
        )
    return numeric_selectors


def create_numeric_selectors_for_attribute(
        data,
        attr_name,
        nbins=5,
        intervals_only=True,
        weighting_attribute=None,
        ignore_null=False,
        include_lower=True,
        include_upper=False
):
    numeric_selectors = []
    data_not_null = data[data[attr_name].notnull()]

    uniqueValues = np.unique(data_not_null[attr_name])
    if (not ignore_null) and len(data_not_null.index) < len(data.index):
        numeric_selectors.append(ps.EqualitySelector(attr_name, np.nan))

    if len(uniqueValues) <= nbins:
        for val in uniqueValues:
            numeric_selectors.append(ps.EqualitySelector(attr_name, val))
    else:
        if nbins == -1:
            cutpoints = uniqueValues
        else:
            cutpoints = ps.equal_frequency_discretization(
                data, attr_name, nbins, weighting_attribute
            )

        if intervals_only:
            old_cutpoint = float("-inf")
            for c in cutpoints:
                numeric_selectors.append(IntervalSelector(
                    attr_name,
                    old_cutpoint,
                    c,
                    include_lower=include_lower,
                    include_upper=include_upper
                ))
                old_cutpoint = c
            numeric_selectors.append(IntervalSelector(
                attr_name,
                old_cutpoint,
                float("inf"),
                include_lower=include_lower,
                include_upper=include_upper
            ))
        else:
            for c in cutpoints:
                numeric_selectors.append(IntervalSelector(
                    attr_name,
                    c,
                    float("inf"),
                    include_lower=include_lower,
                    include_upper=include_upper
                ))
                numeric_selectors.append(IntervalSelector(
                    attr_name,
                    float("-inf"),
                    c,
                    include_lower=include_lower,
                    include_upper=include_upper
                ))

    return numeric_selectors


def remove_target_attributes(selectors, target):
    return [
        sel for sel in selectors if sel.attribute_name not in target.get_attributes()
    ]

