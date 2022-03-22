import logging

import luigi
import os

from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import SparkSession, functions as F

from absolute_cltv.total_pnl.tasks.modelling.calibration import Calibration
from absolute_cltv.total_pnl.tasks.modelling.make_predictions import MakePredictions
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json

logger = logging.getLogger(__name__)


class CombinePredictions(PySparkTask):
    local_result_path = luigi.Parameter()
    hdfs_project_path = luigi.Parameter()
    pnl_products_info_path = luigi.Parameter()

    def requires(self):
        return {
            'predictions': MakePredictions(),
            'calibration': Calibration()
        }

    def output(self):
        return {
            'combine_predictions': HdfsTarget(os.path.join(str(self.hdfs_project_path), 'dataset_test_predictions'))
        }

    def main(self, spark: SparkSession):

        # load data

        dataset = spark.read.parquet(self.input()['predictions']['predictions'].path)
        pnl_products_info = load_json('', str(self.pnl_products_info_path))

        # calibrate classification prediction

        calibration_coefficients = load_json('', self.input()['calibration']['calibration_coefficients'].path)

        calibrated_select = []
        poor_clients_select = []
        product_total_select = []
        total_pl_expr = F.lit(0)

        for product in list(pnl_products_info.keys()):

            probability_column = f'{product}_sales_pl_predict_probability'
            probability_column_calibrated = f'{product}_sales_pl_predict_probability_calibrated'
            poor_mix_column = f'{product}_poor_mix_pl_predict'
            history_column = f'{product}_pl_-1'
            total_product_column = f'{product}_total_pl_predict'

            expr = None
            n_bins = len(calibration_coefficients['coefficients'][product])
            for i, bin_values in enumerate(calibration_coefficients['coefficients'][product]):
                if i == 0:
                    expr = F.when(
                        (F.col(probability_column) >= bin_values[0]) & (F.col(probability_column) < bin_values[1]),
                        F.col(probability_column) * F.lit(bin_values[2])
                    )
                elif i == n_bins - 1:
                    expr = expr.when(
                        (F.col(probability_column) >= bin_values[0]) & (F.col(probability_column) <= bin_values[1]),
                        F.col(probability_column) * F.lit(bin_values[2])
                    ).otherwise(0).alias(probability_column_calibrated)
                else:
                    expr = expr.when(
                        (F.col(probability_column) >= bin_values[0]) & (F.col(probability_column) < bin_values[1]),
                        F.col(probability_column) * F.lit(bin_values[2])
                    )

            calibrated_select += [expr]

            poor_clients_select += [
                (F.col(probability_column_calibrated) * F.col(f'{product}_poor_pl_predict')).alias(poor_mix_column)
            ]

            product_total_select += [
                F.when(F.col(history_column) == 0, F.col(poor_mix_column))
                    .when(F.col(history_column) != 0, F.col(f'{product}_rich_pl_predict'))
                    .otherwise(0).alias(total_product_column)
            ]

            total_pl_expr += F.col(total_product_column)

        combined_predictions_df = (
            dataset
                .select('*', *calibrated_select)
                .select('*', *poor_clients_select)
                .select('*', *product_total_select)
                .select('*', (total_pl_expr).alias('total_pl_predict'))
        )

        combined_predictions_df.write.parquet(self.output()['combine_predictions'].path)
