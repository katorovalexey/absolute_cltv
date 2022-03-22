import logging

import luigi
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json
from absolute_cltv.total_pnl.artifacts.products_and_pnl_items import ArtifactsProductPnlItems
from absolute_cltv.total_pnl.tasks.feature_engineering.pnl_table_windows import PnlTableWindows

logger = logging.getLogger(__name__)


class PnlTablePivotItemsAndProducts(PySparkTask):
    pnl_products_info_path = luigi.Parameter()
    pnl_items_info_path = luigi.Parameter()

    def requires(self):
        return PnlTableWindows()

    def output(self):
        return HdfsTarget(self.hdfs_target_path())

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
        pnl_items = artifacts_product_pnl_items.pnl_items()

        # load data from last step

        df_pnl_table_windows = spark.read.parquet(self.input().path)

        # pivot pl items

        cols_to_group_1 = ['epk_id', 'report_dt', 'product']
        col_to_pivot_1 = 'item'
        col_to_aggr_1 = [a for a in df_pnl_table_windows.columns if a not in (cols_to_group_1 + [col_to_pivot_1])]

        df_pnl_item_pivot = (
            df_pnl_table_windows
                .groupby(*[F.col(a) for a in cols_to_group_1])
                .pivot(col_to_pivot_1, pnl_items)
                .agg(*[F.first(F.col(a)) for a in col_to_aggr_1])
        )

        old_columns_names_1 = []
        new_columns_names_1 = []

        for col_name in df_pnl_item_pivot.columns:
            if col_name not in cols_to_group_1:
                if col_name.find('(') >= 0 and col_name.find(')') >= 0:

                    if col_name.find('_first(item_') >= 0:
                        old_columns_names_1 += [col_name]
                        new_columns_names_1 += [col_name.split('_first(item_')[0] +
                                                '_' +
                                                col_name.split('_first(item_')[1].split(')')[0]]

                    elif col_name.find('_first(pl_') >= 0:
                        if col_name.split('_first(pl_')[0] == pnl_items[0]:
                            old_columns_names_1 += [col_name]
                            new_columns_names_1 += [col_name.split('(')[1].split(')')[0]]

                    else:
                        old_columns_names_1 += [col_name]
                        new_columns_names_1 += [col_name.split('(')[0].split('_first')[0] +
                                                '_' +
                                                col_name.split('(')[1].split(')')[0]]

        select_pnl_item_pivot_renamed = (
                [F.col(a) for a in cols_to_group_1] +
                [F.col(old_columns_names_1[a]).alias(new_columns_names_1[a]) for a in range(len(old_columns_names_1))]
        )

        df_pnl_item_pivot_renamed = (
            df_pnl_item_pivot
                .select(*select_pnl_item_pivot_renamed)
        )

        # pivot products

        cols_to_group_2 = ['epk_id', 'report_dt']
        col_to_pivot_2 = 'product'
        col_to_aggr_2 = new_columns_names_1

        df_pnl_product_item_pivot = (
            df_pnl_item_pivot_renamed
                .groupby(*[F.col(a) for a in cols_to_group_2])
                .pivot(col_to_pivot_2, products)
                .agg(*[F.first(F.col(a)) for a in col_to_aggr_2])
        )

        old_columns_names_2 = []
        new_columns_names_2 = []

        for col_name in df_pnl_product_item_pivot.columns:
            if col_name not in cols_to_group_2:
                if col_name.find('(') >= 0 and col_name.find(')') >= 0:
                    old_columns_names_2 += [col_name]
                    new_columns_names_2 += [col_name.split('(')[0].split('_first')[0] +
                                            '_' +
                                            col_name.split('(')[1].split(')')[0]]

        select_pnl_product_item_pivot_renamed = (
                [F.col(a) for a in cols_to_group_2] +
                [F.col(old_columns_names_2[a]).alias(new_columns_names_2[a]) for a in range(len(old_columns_names_2))]
        )

        df_pnl_table_pivot = (
            df_pnl_product_item_pivot
                .select(*select_pnl_product_item_pivot_renamed)
                .repartition(200)
        )

        df_pnl_table_pivot.write.parquet(str(self.output().path))
