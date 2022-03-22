import logging

import luigi
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from absolute_cltv.total_pnl.tasks.epk_date_config_generating import EpkDateConfigGenerating
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json
from absolute_cltv.total_pnl.artifacts.client_and_month_filters import ArtifactsClientMonth
from absolute_cltv.total_pnl.artifacts.products_and_pnl_items import ArtifactsProductPnlItems
from absolute_cltv.total_pnl.tasks.data_collection.pnl_table_filtering import PnlTableFiltering

logger = logging.getLogger(__name__)


class PnlTableAllAttributesForClient(PySparkTask):
    pnl_products_info_path = luigi.Parameter()
    pnl_items_info_path = luigi.Parameter()

    def requires(self):
        return {
            'config': EpkDateConfigGenerating(),
            'dataset': PnlTableFiltering()
        }

    def output(self):
        return HdfsTarget(self.hdfs_target_path())

    def main(self, spark: SparkSession):

        # load data from last step

        df_pnl_table_filtered = spark.read.parquet(self.input()['dataset'].path)

        # load info about epk_id hash groups and report_dt values

        epk_bins_and_report_dt_pairs = load_json('', self.input()['config'].path)

        artifacts_client_month = ArtifactsClientMonth(
            spark=spark,
            epk_bins_and_report_dt_pairs=epk_bins_and_report_dt_pairs
        )

        condition_to_filter_epk_month_pairs = artifacts_client_month.condition_to_filter_epk_month_pairs()
        spark_df_history_and_predict_months = artifacts_client_month.spark_df_history_and_predict_months()

        # load info about products and pnl items

        pnl_products_info = load_json('', str(self.pnl_products_info_path))
        pnl_items_info = load_json('', str(self.pnl_items_info_path))

        artifacts_product_pnl_items = ArtifactsProductPnlItems(
            spark=spark,
            pnl_products_info=pnl_products_info,
            pnl_items_info=pnl_items_info,
        )

        spark_df_pnl_items = artifacts_product_pnl_items.spark_df_pnl_items()
        spark_df_products = artifacts_product_pnl_items.spark_df_products()

        # generation attributes grid

        df_template_without_epk = (
            spark_df_products
                .select('product').distinct()
                .crossJoin(
                F.broadcast(spark_df_pnl_items)
                    .select('item').distinct()
            )
                .repartition(10)
                .crossJoin(
                F.broadcast(spark_df_history_and_predict_months)
                    .select('report_dt').distinct()
            )
                .coalesce(100)
        )

        df_template_with_epk = (
            df_pnl_table_filtered
                .select(F.col('epk_id')).distinct()
                .crossJoin(F.broadcast(df_template_without_epk)
                           )
                .coalesce(200)
        )

        # filter clients/dates pairs in attributes grid

        df_template_with_epk_filtered = (
            df_template_with_epk
                .select(
                '*',
                F.abs(F.hash(F.col('epk_id').cast('string'))).alias('epk_hash')
            )
                .filter(
                condition_to_filter_epk_month_pairs
            )
                .drop(F.col('epk_hash'))
                .coalesce(200)
        )

        # join to get table with all attributes for client

        df_pnl_table_all_attributes_for_client = (
            df_template_with_epk_filtered
                .join(
                df_pnl_table_filtered,
                ['epk_id', 'product', 'item', 'report_dt'],
                'left'
            )
                .select(*[F.col(a) for a in
                          ['epk_id', 'product', 'item', 'report_dt', 'pl_value']])
                .na.fill(0)
                .sort(*[F.col(a) for a in
                        ['epk_id', 'product', 'item', 'report_dt']])
                .repartition(200)
        )

        df_pnl_table_all_attributes_for_client.write.parquet(str(self.output().path))
