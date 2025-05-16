import pandas as pd


class SoftClassifierTarget:
    def __init__(self, gt_name="gt", score_name="score"):
        self.gt_name = gt_name
        self.score_name = score_name

    def get_target_columns(self, data: pd.DataFrame):
        return data[:, [self.gt_name, self.score_name]]

