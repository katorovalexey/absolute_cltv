from typing import List, Tuple

import pandas as pd
from pandas.tseries import offsets


def generate_config_for_epk_date_filters(
        count_predict_months=2,
        count_history_months=2,
        report_dt_test_left_border='2021-04-30',
        report_dt_test_count=1,
        report_dt_train_left_border='2020-01-31',
        report_dt_train_count=12,
        number_of_epk_bins=500,
        epk_left_bin_border_train=1,
        epk_bins_count_train=5,
        epk_left_bin_border_test=100,
        epk_bins_count_test=1,
):
    def next_month(dt: str, n: int) -> str:
        return (pd.to_datetime(dt, format="%Y-%m-%d") +
                offsets.MonthEnd(n)).strftime('%Y-%m-%d')

    def report_dt_list(report_dt_left_border: str, report_dt_count: int) -> List[Tuple[str, str]]:
        return [(
            next_month(report_dt_left_border, a),
            next_month(report_dt_left_border, a)
        ) for a in range(report_dt_count)]

    config = dict()
    config['count_predict_months'] = count_predict_months
    config['count_history_months'] = count_history_months
    config['number_of_epk_bins'] = number_of_epk_bins

    report_dt_train_list = report_dt_list(report_dt_train_left_border, report_dt_train_count)
    report_dt_test_list = report_dt_list(report_dt_test_left_border, report_dt_test_count)

    epk_bins_train_list = [
        (a, a + epk_bins_count_train - 1) for a in
        [epk_left_bin_border_train + epk_bins_count_train * a for a in range(report_dt_train_count)]]
    epk_bins_test_list = [
        (a, a + epk_bins_count_test - 1) for a in
        [epk_left_bin_border_test + epk_bins_count_test * a for a in range(report_dt_test_count)]]

    config['report_dt_borders_list'] = report_dt_train_list + report_dt_test_list
    config['epk_bins_borders_list'] = epk_bins_train_list + epk_bins_test_list

    train_dict = config.copy()
    test_dict = config.copy()
    config['train'] = train_dict
    config['test'] = test_dict

    config['train']['report_dt_borders_list'] = report_dt_train_list
    config['train']['epk_bins_borders_list'] = epk_bins_train_list

    config['test']['report_dt_borders_list'] = report_dt_test_list
    config['test']['epk_bins_borders_list'] = epk_bins_test_list

    return config
