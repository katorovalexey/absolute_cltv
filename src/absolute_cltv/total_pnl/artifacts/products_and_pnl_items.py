from typing import Dict, List

from pyspark.sql import SparkSession, DataFrame


class ArtifactsProductPnlItems:

    def __init__(
            self,
            spark: SparkSession,
            pnl_products_info: Dict[str, List[any]],
            pnl_items_info: Dict[str, List[int]]
    ):
        self._pnl_items_info = pnl_items_info
        self._pnl_products_info = pnl_products_info
        self._spark = spark

    def products(self) -> List[str]:
        return list(self._pnl_products_info.keys())

    def product_cd_list(self) -> List[int]:
        return [
            cd for value in self._pnl_products_info.values() for cd in value[1]
        ]

    def spark_df_products(self) -> DataFrame:

        all_products_cd = []
        all_products_cd_groups = []
        for product_cd_key in self._pnl_products_info.keys():
            product_cd_list = self._pnl_products_info[product_cd_key][1]
            all_products_cd += product_cd_list
            all_products_cd_groups += [product_cd_key for a in product_cd_list]

        product_cd_data = list(zip(all_products_cd, all_products_cd_groups))
        product_cd_columns = ['product_cd', 'product']

        return (
            self._spark
                .createDataFrame(product_cd_data)
                .toDF(*product_cd_columns)
        )

    def pnl_items(self) -> List[str]:
        return list(self._pnl_items_info.keys())

    def spark_df_pnl_items(self) -> DataFrame:

        all_pl_items = []
        all_pl_items_groups = []
        for pl_items_key in self._pnl_items_info.keys():
            pl_items_list = self._pnl_items_info[pl_items_key]
            all_pl_items += pl_items_list
            all_pl_items_groups += [pl_items_key for a in pl_items_list]

        pl_items_data = list(zip(all_pl_items, all_pl_items_groups))
        pl_items_columns = ['pl_item', 'item']

        return (
            self._spark
                .createDataFrame(pl_items_data)
                .toDF(*pl_items_columns)
        )
