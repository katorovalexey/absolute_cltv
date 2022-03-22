import logging

import luigi
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import SparkSession

from cltv_tools.luigi.spark.tasks import PySparkTask
from absolute_cltv.total_pnl.tasks.feature_engineering.mega_table_transform import MegaTableTransform
from absolute_cltv.total_pnl.tasks.feature_engineering.pnl_table_final_prepare import PnlTableFinalPrepare

logger = logging.getLogger(__name__)


class FeatureCollection(PySparkTask):

    def requires(self):
        return {
            'mega_dataset': MegaTableTransform(),
            'pnl_dataset': PnlTableFinalPrepare()
        }

    def output(self):
        return HdfsTarget(self.hdfs_target_path())

    def _input_dataset(self, spark: SparkSession):

        df_mega = spark.read.parquet(self.input()['mega_dataset']['df_mega_table_mapped'].path)
        df_pnl = spark.read.parquet(self.input()['pnl_dataset']['df_pnl'].path)
        return df_mega, df_pnl

    def main(self, spark: SparkSession):
        df_mega, df_pnl = self._input_dataset(spark)


        df_full_pnl_mega_dataset = (
            df_mega
                .join(
                df_pnl,
                ['epk_id', 'report_dt'],
                'left'
            )
                .na.fill(0)
                .repartition(200)
        )

        df_full_pnl_mega_dataset.write.parquet(
            self.output().path
        )
