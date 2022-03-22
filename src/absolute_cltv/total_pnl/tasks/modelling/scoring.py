import luigi
import os
from luigi import LocalTarget
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import SparkSession, functions as F

from absolute_cltv.total_pnl.artifacts.generate_config_for_epk_date_filters import generate_config_for_epk_date_filters
from absolute_cltv.total_pnl.artifacts.products_and_pnl_items import ArtifactsProductPnlItems
from absolute_cltv.total_pnl.tasks.data_collection.mega_table_filtering import MegaTableFiltering
from absolute_cltv.total_pnl.tasks.data_collection.pnl_table_filtering import PnlTableFiltering
from absolute_cltv.total_pnl.tasks.epk_date_config_generating import EpkDateConfigGenerating
from absolute_cltv.total_pnl.tasks.feature_collection import FeatureCollection
from absolute_cltv.total_pnl.tasks.feature_engineering.mega_table_train_transformers import MegaTableTrainTransformers
from absolute_cltv.total_pnl.tasks.feature_engineering.mega_table_transform import MegaTableTransform
from absolute_cltv.total_pnl.tasks.feature_engineering.pnl_table_all_attributes_for_client_ import \
    PnlTableAllAttributesForClient
from absolute_cltv.total_pnl.tasks.feature_engineering.pnl_table_pivot_items_and_products import \
    PnlTablePivotItemsAndProducts
from absolute_cltv.total_pnl.tasks.feature_engineering.pnl_table_windows import PnlTableWindows
from absolute_cltv.total_pnl.tasks.modelling.calibration import Calibration
from absolute_cltv.total_pnl.tasks.modelling.combine_predictions import CombinePredictions
from absolute_cltv.total_pnl.tasks.modelling.make_predictions import MakePredictions
from cltv_tools.utils.json import dump_json, load_json


class ScoringEpkDateConfigGenerating(EpkDateConfigGenerating):
    count_predict_months = luigi.IntParameter()

    report_dt_scoring_left_border = luigi.Parameter()
    report_dt_scoring_count = luigi.IntParameter()

    number_of_epk_bins_scoring = luigi.IntParameter()
    epk_left_bin_border_scoring = luigi.IntParameter()
    epk_bins_count_scoring = luigi.IntParameter()

    def requires(self):
        return Calibration()

    def output(self):
        return LocalTarget(os.path.join(
            str(self.local_result_path),
            'scoring_epk_date_config_generating.json'
        ))

    def main(self, **kwargs):
        config = generate_config_for_epk_date_filters(
            count_predict_months=self.count_predict_months,
            count_history_months=self.count_history_months,
            report_dt_test_left_border=self.report_dt_scoring_left_border,
            report_dt_test_count=self.report_dt_scoring_count,
            report_dt_train_left_border=self.report_dt_scoring_left_border,
            report_dt_train_count=self.report_dt_scoring_count,
            number_of_epk_bins=self.number_of_epk_bins_scoring,
            epk_left_bin_border_train=self.epk_left_bin_border_scoring,
            epk_left_bin_border_test=self.epk_left_bin_border_scoring,
            epk_bins_count_train=self.epk_bins_count_scoring,
            epk_bins_count_test=self.epk_bins_count_scoring
        )

        path = os.path.join(self.hdfs_project_path, 'scores')
        os.popen(f'hdfs dfs -mkdir {path}').read()
        dump_json('', self.output().path, config)


class ScoringMegaTableFiltering(MegaTableFiltering):

    def requires(self):
        return ScoringEpkDateConfigGenerating()


class ScoringMegaTableTransform(MegaTableTransform):

    def requires(self):
        return {
            'transformers': MegaTableTrainTransformers(),
            'dataset': ScoringMegaTableFiltering()
        }

    def output(self):
        return {
            'df_mega_table_mapped': HdfsTarget(self.hdfs_target_path())
        }

    def _save_output(self, mega_features_groups, df_mega_table_mapped):
        df_mega_table_mapped.write.parquet(self.output()['df_mega_table_mapped'].path)


class ScoringPnlTableFiltering(PnlTableFiltering):

    def requires(self):
        return ScoringEpkDateConfigGenerating()


class ScoringPnlTableAllAttributesForClient(PnlTableAllAttributesForClient):

    def requires(self):
        return {
            'config': ScoringEpkDateConfigGenerating(),
            'dataset': ScoringPnlTableFiltering()
        }


class ScoringPnlTableWindows(PnlTableWindows):

    def requires(self):
        return ScoringPnlTableAllAttributesForClient()


class ScoringPnlTablePivotItemsAndProducts(PnlTablePivotItemsAndProducts):

    def requires(self):
        return ScoringPnlTableWindows()


class ScoringFeatureCollection(FeatureCollection):

    def requires(self):
        return {
            'mega_dataset': ScoringMegaTableTransform(),
            'pnl_dataset': ScoringPnlTablePivotItemsAndProducts()
        }

    def _input_dataset(self, spark: SparkSession):
        df_mega = spark.read.parquet(self.input()['mega_dataset']['df_mega_table_mapped'].path)
        df_pnl = spark.read.parquet(self.input()['pnl_dataset'].path)
        return df_mega, df_pnl


class ScoringMakePredictions(MakePredictions):
    pnl_items_info_path = luigi.Parameter()

    def requires(self):
        return ScoringFeatureCollection()

    def _dataset_task(self):
        return ScoringFeatureCollection()

    def _input_dataset(self, spark: SparkSession):

        dataset = spark.read.parquet(self.input().path)

        pnl_products_info = load_json('', str(self.pnl_products_info_path))
        pnl_items_info = load_json('', str(self.pnl_items_info_path))

        artifacts_product_pnl_items = ArtifactsProductPnlItems(
            spark=spark,
            pnl_products_info=pnl_products_info,
            pnl_items_info=pnl_items_info,
        )

        products = artifacts_product_pnl_items.products()

        total_naive = None

        for i, product in enumerate(products):

            if i == 0:
                total_naive = F.col(f'{product}_pl_-1')
            else:
                total_naive += F.col(f'{product}_pl_-1')

        return (
            dataset
                .select('*', total_naive.alias(f'total_pl_-1'))
                .na.fill(0)
                .repartition(200)
        )

    def _save_output(self, pnl_dataset_columns_groups, df_pnl):
        dump_json('', self.output()['pnl_dataset_columns_groups'].path, pnl_dataset_columns_groups)
        df_pnl.write.parquet(self.output()['df_pnl'].path)

    def main(self, spark: SparkSession):

        dataset, columns = self._main_without_saving(spark)
        scoring_columns = [a for a in columns if a.find('target') < 0]
        dataset.select(*[F.col(column) for column in scoring_columns]).write.parquet(self.output()['predictions'].path)


class ScoringCombinePredictions(CombinePredictions):
    scores_directory_name = luigi.Parameter()

    def requires(self):
        return {
            'predictions': ScoringMakePredictions(),
            'calibration': Calibration()
        }

    def output(self):
        return {
            'combine_predictions': HdfsTarget(os.path.join(
                str(self.hdfs_project_path),
                str(self.scores_directory_name))
            )
        }
