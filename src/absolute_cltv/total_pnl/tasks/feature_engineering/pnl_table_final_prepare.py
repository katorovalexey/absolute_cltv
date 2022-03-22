import logging
import os

import luigi
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json, dump_json
from absolute_cltv.total_pnl.artifacts.products_and_pnl_items import ArtifactsProductPnlItems
from absolute_cltv.total_pnl.tasks.feature_engineering.pnl_table_pivot_items_and_products import \
    PnlTablePivotItemsAndProducts

logger = logging.getLogger(__name__)


class PnlTableFinalPrepare(PySparkTask):
    pnl_products_info_path = luigi.Parameter()
    pnl_items_info_path = luigi.Parameter()
    count_predict_months = luigi.IntParameter()
    local_result_path = luigi.Parameter()

    def requires(self):
        return PnlTablePivotItemsAndProducts()

    def output(self):
        return {
            'df_pnl': HdfsTarget(self.hdfs_target_path()),
            'pnl_dataset_columns_groups': luigi.LocalTarget(os.path.join(
                str(self.local_result_path),
                'pnl_dataset_columns_groups.json'
            ))
        }

    def _save_output(self, pnl_dataset_columns_groups, df_pnl):
        dump_json('', self.output()['pnl_dataset_columns_groups'].path, pnl_dataset_columns_groups)
        df_pnl.write.parquet(self.output()['df_pnl'].path)

    def main(self, spark: SparkSession):

        # load info about products and pnl items

        pnl_products_info = load_json('', str(self.pnl_products_info_path))
        pnl_items_info = load_json('', str(self.pnl_items_info_path))

        artifacts_product_pnl_items = ArtifactsProductPnlItems(
            spark=spark,
            pnl_products_info=pnl_products_info,
            pnl_items_info=pnl_items_info,
        )

        products = artifacts_product_pnl_items.products()

        # load data from last step

        df_pnl_table_pivot = spark.read.parquet(self.input().path)

        # calculating sum of targets and columns replacement

        exceptions_target_sum = []
        expression_target_sum = None
        start_columns = ['epk_id', 'report_dt']
        select_target_sum = [F.col(a) for a in start_columns]

        for product in products:
            for month in range(self.count_predict_months):
                exceptions_target_sum += [f'{product}_pl_{month}']

                if month == 0:
                    expression_target_sum = F.col(f'{product}_pl_{month}')
                else:
                    expression_target_sum += F.col(f'{product}_pl_{month}')

            if expression_target_sum != None:
                select_target_sum += [(expression_target_sum).alias(f'{product}_pl_target')]

        other_columns_target_sum = [
            a for a in df_pnl_table_pivot.columns
            if a not in (exceptions_target_sum + start_columns)
        ]
        other_columns_target_sum.sort()

        select_target_sum += [F.col(a) for a in other_columns_target_sum]

        df_pnl_products_real_target = (
            df_pnl_table_pivot
                .select(*select_target_sum)
        )

        # separating columns

        pnl_table_features = [
            column for column in df_pnl_products_real_target.columns if
            column not in ['epk_id', 'report_dt'] and
            column.find('target') < 0
        ]
        pnl_table_targets_products_pnl_value = [
            column for column in df_pnl_products_real_target.columns if
            column.find('target') >= 0
        ]

        pnl_table_naive_products_pnl_value = [
            column for column in df_pnl_products_real_target.columns if
            column.find('_pl_-1') >= 0
        ]

        # targets as sales flag for clients with 0 pnl in last month

        select_sales_pl_target = ['*']
        for product in products:
            select_sales_pl_target += [
                F.when(
                    (F.col(f'{product}_pl_-1') == 0) &
                    (F.col(f'{product}_pl_target') != 0),
                    F.lit(1)
                ).otherwise(F.lit(0)).alias(f'{product}_sales_pl_target')
            ]

        df_pnl_products_flag_target = (
            df_pnl_products_real_target
                .select(*select_sales_pl_target)
                .na.fill(0)
        )

        pnl_table_targets_products_pnl_flag = [
            column for column in df_pnl_products_flag_target.columns if
            column.find('sales_pl_target') >= 0
        ]

        # targets as total pnl value (all products) for client

        total_sum = None
        total_naive = None

        for i, targ in enumerate(pnl_table_targets_products_pnl_value):

            if i == 0:
                total_sum = F.col(targ)
                total_naive = F.col(targ.split('target')[0] + '-1')
            else:
                total_sum += F.col(targ)
                total_naive += F.col(targ.split('target')[0] + '-1')

        df_pnl = (
            df_pnl_products_flag_target
                .select('*', total_naive.alias(f'total_pl_-1'), total_sum.alias('total_pl_target'))
                .na.fill(0)
                .repartition(200)
        )

        # dictionary with columns params

        pnl_dataset_columns_groups = {
            'features': pnl_table_features,
            'targets_total_pnl': ['total_pl_target'],
            'naive_total_pnl': ['total_pl_-1'],
            'targets_new_products_of_clients_with_null_pnl_flag': pnl_table_targets_products_pnl_flag,
            'targets_products_pnl_value': pnl_table_targets_products_pnl_value,
            'naive_products_pnl_value': pnl_table_naive_products_pnl_value
        }

        self._save_output(pnl_dataset_columns_groups, df_pnl)
