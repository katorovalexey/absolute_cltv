import logging
import os

import luigi
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import SparkSession, functions as F

from absolute_cltv.total_pnl.artifacts.client_and_month_filters import ArtifactsClientMonth
from absolute_cltv.total_pnl.tasks.feature_collection import FeatureCollection
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json

logger = logging.getLogger(__name__)


class TrainTestSplit(PySparkTask):
    hdfs_project_path = luigi.Parameter()
    local_result_path = luigi.Parameter()

    def requires(self):
        return FeatureCollection()

    def output(self):
        return {
            'train': HdfsTarget(os.path.join(str(self.hdfs_project_path), 'dataset_train')),
            'test': HdfsTarget(os.path.join(str(self.hdfs_project_path), 'dataset_test')),
        }

    def main(self, spark: SparkSession):
        df_dataset = (spark.read.parquet(self.input().path)
                      .select(F.col('*'), F.abs(F.hash(F.col('epk_id').cast('string'))).alias('epk_hash')))

        # load info about epk_id hash groups and report_dt values

        epk_bins_and_report_dt_pairs = load_json(str(self.local_result_path), 'epk_date_config_generating.json')

        artifacts_client_month_train = ArtifactsClientMonth(
            spark=spark,
            epk_bins_and_report_dt_pairs=epk_bins_and_report_dt_pairs['train']
        )
        condition_to_filter_train = artifacts_client_month_train.condition_to_filter_epk_month_pairs(table='mega')

        artifacts_client_month_test = ArtifactsClientMonth(
            spark=spark,
            epk_bins_and_report_dt_pairs=epk_bins_and_report_dt_pairs['test']
        )
        condition_to_filter_test = artifacts_client_month_test.condition_to_filter_epk_month_pairs(table='mega')

        # filter clients/dates pairs by hash

        df_train_without_valid = (
            df_dataset
                .filter(condition_to_filter_train)
                .drop(F.col('epk_hash'))
        )

        df_train = (
            df_train_without_valid
                .select(
                    F.col('*'),
                    F.abs(
                        F.hash(
                            F.concat(F.col('report_dt').cast('string'), F.col('epk_id').cast('string'))
                        )
                    ).alias('id_hash')
                )
                .withColumn(
                    'is_valid',
                    F.when(
                        F.col('id_hash') % 5 == 4, F.lit(True)
                    )
                    .otherwise(F.lit(False))
                )
                .drop(F.col('id_hash'))
                .repartition(200)
        )

        df_test = (
            df_dataset
                .filter(condition_to_filter_test)
                .drop(F.col('epk_hash'))
                .repartition(200)
        )

        df_train.write.parquet(self.output()['train'].path)
        df_test.write.parquet(self.output()['test'].path)
