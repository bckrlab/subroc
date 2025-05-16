import pandas as pd
import pysubgroup as ps

from subroc.quality_functions.soft_classifier_target import SoftClassifierTarget


class BoundedGeneralizationAwareQF(ps.GeneralizationAwareQF, ps.BoundedInterestingnessMeasure):
    def optimistic_estimate(self, subgroup, target: SoftClassifierTarget, data: pd.DataFrame, statistics=None):
        return self.qf.optimistic_estimate(subgroup, target, data, statistics)

