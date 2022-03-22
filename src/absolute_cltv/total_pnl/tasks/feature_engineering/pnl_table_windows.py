import logging

import luigi
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

from cltv_tools.luigi.spark.tasks import PySparkTask
from absolute_cltv.total_pnl.tasks.feature_engineering.pnl_table_all_attributes_for_client_ import \
    PnlTableAllAttributesForClient

logger = logging.getLogger(__name__)


class PnlTableWindows(PySparkTask):
    count_history_months: int = luigi.IntParameter()
    count_predict_months: int = luigi.IntParameter()

    def requires(self):
        return PnlTableAllAttributesForClient()

    def output(self):
        return HdfsTarget(self.hdfs_target_path())

    def main(self, spark: SparkSession):

        # load data from last step

        df_pnl_table_all_attributes_for_client = spark.read.parquet(self.input().path)

        window_pl_date_total = Window.partitionBy('epk_id', 'product', 'report_dt')
        window_lag_pl_item = Window.partitionBy('epk_id', 'product', 'item').orderBy('report_dt')
        window_lag_pl_total = Window.partitionBy('epk_id', 'product', 'item').orderBy('report_dt')

        history_lags_cols = [F.lag('pl_value', 1 + mnth).over(window_lag_pl_item).alias(f'item_-{2 + mnth}') for mnth in
                             range(self.count_history_months - 1)]
        total_pl_cols = [
            F.lead(
                (F.sum('pl_value').over(window_pl_date_total)),
                mnth
            )
                .over(window_lag_pl_total)
                .alias(f'pl_{mnth - 1}')
            for mnth in range(self.count_predict_months + 1)
        ]

        df_pnl_with_lags = (
            df_pnl_table_all_attributes_for_client
                .select('*',
                        *history_lags_cols,
                        *total_pl_cols
                        )
                .withColumnRenamed('pl_value', 'item_-1')
                .na.drop('any')
        )

        if self.count_history_months < 2:
            aggr_cols_1 = []
            aggr_cols_2 = []
            aggr_cols_3 = []
        else:
            columns_to_aggregate = [f'item_{mnth}' for mnth in range(-self.count_history_months, 0)]
            row_mean = sum([F.col(c) for c in columns_to_aggregate]) / F.lit(len(columns_to_aggregate))

            aggr_cols_1 = [
                row_mean.alias('mean'),
                F.least(*[F.col(c) for c in columns_to_aggregate]).alias('min'),
                F.greatest(*[F.col(c) for c in columns_to_aggregate]).alias('max')
            ]

            aggr_cols_2 = [
                ((F.col('max') - F.col('min')) / (F.when(F.col(f'mean') == 0,
                                                         F.lit(0.0001))
                                                  .otherwise(F.col('mean')))).alias('scatter'),
                (F.col('item_-1') - F.col(f'mean')).alias('delta_abs_mean'),
                (F.col('item_-1') - F.col(f'item_-2')).alias('delta_abs_min'),
                (F.col('item_-1') - F.col(f'item_-{self.count_history_months}')).alias('delta_abs_max')
            ]

            aggr_cols_3 = [
                (F.col('delta_abs_mean') / (F.when(F.col(f'mean') == 0,
                                                   F.lit(0.0001))
                                            .otherwise(F.col('mean')))).alias('delta_relative_mean'),
                (F.col('delta_abs_min') / (F.when(F.col(f'mean') == 0,
                                                  F.lit(0.0001))
                                           .otherwise(F.col('mean')))).alias('delta_relative_min'),
                (F.col('delta_abs_max') / (F.when(F.col(f'mean') == 0,
                                                  F.lit(0.0001))
                                           .otherwise(F.col('mean')))).alias('delta_relative_max')
            ]

        df_pnl_table_windows = (
            df_pnl_with_lags
                .select('*', *aggr_cols_1)
                .select('*', *aggr_cols_2)
                .select('*', *aggr_cols_3)
                .repartition(200)
        )

        df_pnl_table_windows.write.parquet(str(self.output().path))
