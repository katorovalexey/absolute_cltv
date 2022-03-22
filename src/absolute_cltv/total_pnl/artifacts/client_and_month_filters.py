from typing import List, Tuple, Dict

import pandas as pd
from pandas.tseries import offsets
from pyspark.sql import SparkSession, DataFrame, functions as F


class ArtifactsClientMonth:

    def __init__(
            self,
            spark: SparkSession,
            epk_bins_and_report_dt_pairs: Dict,
    ):

        self._epk_bins_borders_list = epk_bins_and_report_dt_pairs['epk_bins_borders_list']
        self._number_of_epk_bins = epk_bins_and_report_dt_pairs['number_of_epk_bins']

        self._report_dt_borders_list = epk_bins_and_report_dt_pairs['report_dt_borders_list']

        self._count_predict_months = epk_bins_and_report_dt_pairs['count_predict_months']
        self._count_history_months = epk_bins_and_report_dt_pairs['count_history_months']
        self._spark = spark

        # _month_list_for_current_report_dt(self, report_dt_borders: Tuple[str]) -> List[str]:

    def _next_month(self, dt: str, n: int):
        return (pd.to_datetime(dt, format="%Y-%m-%d") +
                offsets.MonthEnd(n)).strftime('%Y-%m-%d')

    def _month_list_for_current_report_dt(self, report_dt: str) -> List[str]:
        return [
            (pd.to_datetime(report_dt, format="%Y-%m-%d") + offsets.MonthEnd(a)).strftime('%Y-%m-%d')
            for a in range(-(self._count_history_months - 1), self._count_predict_months + 1)
        ]

    def _month_list_for_current_bin(self, report_dt_borders: Tuple[str]) -> List[str]:
        report_dt_list = []
        n_month = 0
        while self._next_month(report_dt_borders[0], n_month) <= self._next_month(report_dt_borders[1], 0):
            report_dt_list += [self._next_month(report_dt_borders[0], n_month)]
            n_month += 1

        all_month = set()
        for report_dt in report_dt_list:
            for month in self._month_list_for_current_report_dt(report_dt):
                all_month.add(month)

        all_month_list = list(all_month)
        all_month_list.sort()
        return all_month_list

    def history_and_predict_months(self) -> List[str]:

        report_dt_list = []

        for report_dt_borders in self._report_dt_borders_list:
            n_month = 0
            while (
                    self._next_month(report_dt_borders[0], n_month) <=
                    self._next_month(report_dt_borders[1], 0)
            ):
                report_dt_list += self._month_list_for_current_bin(report_dt_borders)
                n_month += 1

        all_month = list(
            set(report_dt_list)
        )
        all_month.sort()

        return all_month

    def spark_df_history_and_predict_months(self) -> DataFrame:
        return (
            self._spark
                .createDataFrame(zip(self.history_and_predict_months()))
                .toDF('report_dt')
        )

    def client_group_and_last_history_months_pairs(self, table='pnl') -> List[Tuple[Tuple[int, int], Tuple[str, str]]]:

        epk_date_pairs = []
        for i in range(len(self._report_dt_borders_list)):
            if table == 'pnl':
                report_dt_list = self._month_list_for_current_bin(self._report_dt_borders_list[i])
            elif table == 'mega':
                report_dt_list = [self._report_dt_borders_list[i][a] for a in [0,1]]
            epk_date_pairs += [(
                self._epk_bins_borders_list[i],
                (report_dt_list[0], report_dt_list[-1])
            )]

        return epk_date_pairs

    def condition_to_filter_epk_month_pairs(self, table='pnl'):

        epk_date_pairs = self.client_group_and_last_history_months_pairs(table=table)

        condition_to_filter_epk_month_pairs = None

        for i in range(len(epk_date_pairs)):
            condition = (
                    (F.col('epk_hash') % self._number_of_epk_bins >= epk_date_pairs[i][0][0]) &
                    (F.col('epk_hash') % self._number_of_epk_bins <= epk_date_pairs[i][0][1]) &
                    (F.col('report_dt') >= epk_date_pairs[i][1][0]) &
                    (F.col('report_dt') <= epk_date_pairs[i][1][1])
            )
            if i == 0:
                condition_to_filter_epk_month_pairs = condition
            else:
                condition_to_filter_epk_month_pairs |= condition

        return condition_to_filter_epk_month_pairs