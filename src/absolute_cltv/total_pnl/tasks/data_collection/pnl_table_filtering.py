import logging

import luigi
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import functions as F, SparkSession

from absolute_cltv.total_pnl.tasks.epk_date_config_generating import EpkDateConfigGenerating
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json
from absolute_cltv.total_pnl.artifacts.client_and_month_filters import ArtifactsClientMonth
from absolute_cltv.total_pnl.artifacts.products_and_pnl_items import ArtifactsProductPnlItems

logger = logging.getLogger(__name__)


class PnlTableFiltering(PySparkTask):
    sample_clients_hdfs_path = luigi.Parameter()
    pnl_detail_table = luigi.Parameter()

    pnl_products_info_path = luigi.Parameter()
    pnl_items_info_path = luigi.Parameter()

    def requires(self):
        return EpkDateConfigGenerating()

    def output(self):
        return HdfsTarget(self.hdfs_target_path())

    def main(self, spark: SparkSession):


        # filter clients by join df from sample_clients_path if not None

        df_read_pnl_table = spark.table(self.pnl_detail_table)

        if str(self.sample_clients_hdfs_path).startswith('hdfs'):

            df_read_sample_clients = spark.read.parquet(self.sample_clients_hdfs_path).coalesce(2000)

            df_pnl_table = (
                df_read_pnl_table
                    .join(
                    df_read_sample_clients,
                    df_read_sample_clients.epk_id == df_read_pnl_table.client_id
                )
                    .drop("client_id")
                    .withColumnRenamed('report_date', 'report_dt')
                    .coalesce(500)
            )

        else:
            df_pnl_table = (
                df_read_pnl_table
                    .withColumnRenamed('client_id', 'epk_id')
                    .withColumnRenamed('report_date', 'report_dt')
            )

        # load info about epk_id hash groups and report_dt values

        epk_bins_and_report_dt_pairs = load_json('', self.input().path)

        artifacts_client_month = ArtifactsClientMonth(
            spark=spark,
            epk_bins_and_report_dt_pairs=epk_bins_and_report_dt_pairs
        )

        condition_to_filter_epk_month_pairs = artifacts_client_month.condition_to_filter_epk_month_pairs()

        # load info about products and pnl items

        pnl_products_info = load_json('', str(self.pnl_products_info_path))
        pnl_items_info = load_json('', str(self.pnl_items_info_path))

        artifacts_product_pnl_items = ArtifactsProductPnlItems(
            spark=spark,
            pnl_products_info=pnl_products_info,
            pnl_items_info=pnl_items_info,
        )

        product_cd_list = artifacts_product_pnl_items.product_cd_list()
        spark_df_pnl_items = artifacts_product_pnl_items.spark_df_pnl_items()
        spark_df_products = artifacts_product_pnl_items.spark_df_products()

        # filter clients/dates pairs by hash and products/pnl_items mapping

        df_pnl_table_filtered = (
            df_pnl_table
                .select(F.col('*'), F.abs(F.hash(F.col('epk_id').cast('string'))).alias('epk_hash'))
                .filter(condition_to_filter_epk_month_pairs)
                .coalesce(1000)
                .filter(
                F.col('product_cd').isin(product_cd_list)
            )
                .coalesce(500)
                .groupby('epk_id', 'product_cd', 'report_dt', 'pl_item')
                .agg(F.sum(F.col('pl_val')).alias('pl_value'))
                .drop(F.col('epk_hash'))
                .join(F.broadcast(spark_df_products), ['product_cd'], 'left')
                .join(F.broadcast(spark_df_pnl_items), ['pl_item'], 'left')
                .groupby('epk_id', 'product', 'item', 'report_dt')
                .agg(F.sum(F.col('pl_value')).alias('pl_value'))
                .repartition(200)
        )

        df_pnl_table_filtered.write.parquet(str(self.output().path))
