import numpy as np
import pysubgroup as ps


class MinimumPerClassSupportConstraint:
    def __init__(self, min_support, gt_name, data, labels):
        self.min_support = min_support
        self.num_classes = len(labels)
        self.selectors = [ps.EqualitySelector(gt_name, label) for label in labels]

        if data is not None:
            for i in range(len(self.selectors)):
                self.selectors[i].representation = self.selectors[i].covers(data)

    @property
    def is_monotone(self):
        return True

    def is_satisfied(self, subgroup, statistics=None, data=None):
        # Restrict the subgroup to each class and count the included samples, then count the cases were at least one
        # sample was left.
        return all(np.array([ps.get_size(subgroup & selector, len(data), data) for selector in self.selectors]) >= self.min_support)
    
    def update(self, data):
        if data is not None:
            for i in range(len(self.selectors)):
                self.selectors[i].representation = None
                self.selectors[i].representation = self.selectors[i].covers(data)


class MinClassesConstraint:
    def __init__(self, min_classes, gt_name, data, labels=None):
        self.min_classes = min_classes

        if labels is None:
            labels = range(min_classes)
        self.selectors = [ps.EqualitySelector(gt_name, label) for label in labels]

        if data is not None:
            for i in range(len(self.selectors)):
                self.selectors[i].representation = self.selectors[i].covers(data)

    @property
    def is_monotone(self):
        return True

    def is_satisfied(self, subgroup, statistics=None, data=None):
        # Restrict the subgroup to each class and count the included samples, then count the cases were at least one
        # sample was left.
        return (sum(np.array([ps.get_size(subgroup & selector, len(data), data) for selector in self.selectors]) > 0)
                >= self.min_classes)
    
    def update(self, data):
        if data is not None:
            for i in range(len(self.selectors)):
                self.selectors[i].representation = None
                self.selectors[i].representation = self.selectors[i].covers(data)


class ContainsValueConstraint:
    def __init__(self, attribute_name, value, data):
        self.attribute_name = attribute_name
        self.value = value

        self.selector = ps.EqualitySelector(attribute_name, value)
        self.selector.representation = self.selector.covers(data)

    @property
    def is_monotone(self):
        return True

    def is_satisfied(self, subgroup, statistics=None, data=None):
        return ps.get_size(subgroup & self.selector, len(data), data) > 0
    
    def update(self, data):
        self.selector.representation = None
        self.selector.representation = self.selector.covers(data)


class ConstraintDisjunction:
    def __init__(self, constraints):
        self.constraints = constraints

    @property
    def is_monotone(self):
        return all([constraint.is_monotone for constraint in self.constraints])

    def is_satisfied(self, subgroup, statistics=None, data=None):
        return any([constraint.is_satisfied(subgroup, statistics, data) for constraint in self.constraints])

