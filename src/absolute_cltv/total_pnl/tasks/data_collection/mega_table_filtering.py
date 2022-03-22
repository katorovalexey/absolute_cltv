import logging

import luigi
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import functions as F, SparkSession

from absolute_cltv.total_pnl.tasks.epk_date_config_generating import EpkDateConfigGenerating
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json
from absolute_cltv.total_pnl.artifacts.client_and_month_filters import ArtifactsClientMonth

logger = logging.getLogger(__name__)


class MegaTableFiltering(PySparkTask):

    sample_clients_hdfs_path = luigi.Parameter()
    mega_table = luigi.Parameter()

    sample_features_from_mega_path = luigi.Parameter()
    mega_features_postfix_groups_path = luigi.Parameter()

    def requires(self):
        return EpkDateConfigGenerating()

    def output(self):
        return HdfsTarget(self.hdfs_target_path())

    def main(self, spark: SparkSession):

        # filter features list

        mega_features_postfix_groups = load_json('', str(self.mega_features_postfix_groups_path))

        true_postfix_list = [
            postfix for postfix_list in mega_features_postfix_groups.values() for postfix in postfix_list
        ]

        if str(self.sample_features_from_mega_path).startswith('no_path'):

            df_read_full_mega_table = spark.table(self.mega_table)
            features = [
                column for column in df_read_full_mega_table.columns if (
                        column not in ['epk_id', 'report_dt'] and
                        column.split('_')[-1] in true_postfix_list
                )
            ]

        else:

            features = [
                feature for feature in load_json('', str(self.sample_features_from_mega_path)) if
                feature.split('_')[-1] in true_postfix_list
            ]

        df_read_mega_table = spark.table(self.mega_table).select(
            *[F.col(column) for column in (['epk_id', 'report_dt'] + features)]
        )

        # filter clients by join df from sample_clients_path if not None

        if str(self.sample_clients_hdfs_path).startswith('no_path'):

            df_mega_table = df_read_mega_table

        else:

            df_read_sample_clients = spark.read.parquet(self.sample_clients_hdfs_path).coalesce(500)

            df_mega_table = (
                df_read_mega_table
                    .join(
                    df_read_sample_clients,
                    'epk_id'
                )
                    .coalesce(500)
            )


        # load info about epk_id hash groups and report_dt values

        epk_bins_and_report_dt_pairs = load_json('', self.input().path)

        artifacts_client_month = ArtifactsClientMonth(
            spark=spark,
            epk_bins_and_report_dt_pairs=epk_bins_and_report_dt_pairs
        )

        condition_to_filter_epk_month_pairs = artifacts_client_month.condition_to_filter_epk_month_pairs(table='mega')

        # filter clients/dates pairs by hash

        df_mega_table_filtered = (
            df_mega_table
                .select(F.col('*'), F.abs(F.hash(F.col('epk_id').cast('string'))).alias('epk_hash'))
                .filter(condition_to_filter_epk_month_pairs)
                .drop(F.col('epk_hash'))
                .repartition(200)
        )

        df_mega_table_filtered.write.parquet(str(self.output().path))
