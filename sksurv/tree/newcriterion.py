from ._criterion import LogrankCriterion
from sklearn.tree._classes import CRITERIA_REG, CRITERIA_CLF

def new_criterion(y_numeric,  n_outputs_, n_samples, unique_times_, is_event_time_):
    criterion = 0
    for _, status in y_numeric.items():
        if status == 0:
            criterion = LogrankCriterion(n_outputs=n_outputs_, n_samples=n_samples,
                                    uniqumes=unique_times_, is_event_time=isnt_time_)
        else:
            criterion = LogrankCriterion(n_outputs=n_outputs_, n_samples=n_samples,
                                    uniqumes=unique_times_, is_event_time=self.isnt_time_) + CRITERIA_REG["squared_error"](n_outputs=n_outputs_, n_samples=n_samples)

    return criterion
