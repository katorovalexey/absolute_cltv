import logging

import luigi
import os
from luigi import LocalTarget
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F

from absolute_cltv.total_pnl.artifacts.calibration_coefficients import calibration_coefficients
from absolute_cltv.total_pnl.tasks.modelling.train_models import TrainPoorClassificationModel
from absolute_cltv.total_pnl.tasks.modelling.train_test_split import TrainTestSplit
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json, dump_json

logger = logging.getLogger(__name__)


class Calibration(PySparkTask):
    local_result_path = luigi.Parameter()
    hdfs_project_path = luigi.Parameter()
    pnl_products_info_path = luigi.Parameter()

    def requires(self):
        return_dict = dict()
        return_dict['dataset'] = TrainTestSplit()

        pnl_products_info = load_json('', str(self.pnl_products_info_path))
        for product in list(pnl_products_info.keys()):
            return_dict[f'poor_classification_{product}'] = TrainPoorClassificationModel(**{
                'product': product,
                'model_name': 'poor_classification',
                'model_type': 'gb',
                'loss': None
            })

        return return_dict

    def output(self):
        return {
            'calibration_coefficients': LocalTarget(
                os.path.join(
                    self._local_models_params_path(),
                    'calibration_coefficients.json'
                )
            )
        }

    def _local_models_params_path(self) -> str:
        return os.path.join(str(self.local_result_path), 'models_params')

    def main(self, spark: SparkSession):

        dataset = spark.read.parquet(self.input()['dataset']['train'].path)
        dataset = dataset.filter(F.col('is_valid') == True)

        # make predictions

        pnl_products_info = load_json('', str(self.pnl_products_info_path))
        model_path = os.path.join(str(self.hdfs_project_path), 'models')

        calibration_coefficients_dict = dict()
        calibration_coefficients_dict['read_me'] = (
            'List of [left_score_bin_border, right_score_bin_border, coefficient (score * coefficient)] for each bin'
        )
        calibration_coefficients_dict['coefficients'] = dict()

        for product in list(pnl_products_info.keys()):
            model_part = 'poor_classification'

            model = PipelineModel.load(os.path.join(model_path, f'model_{product}_{model_part}'))
            dataset_predict = model.transform(dataset)

            prob_col = model.stages[1].getProbabilityCol()

            pd_df = (
                dataset_predict
                    .filter(F.col(f'{product}_pl_-1') == 0)
                    .select(
                    F.col(f'{product}_sales_pl_target').alias('target'),
                    vector_to_array(prob_col).getItem(1).alias('predict')
                ).toPandas()
            )

            product_coefficients = calibration_coefficients(pd_df)
            calibration_coefficients_dict['coefficients'][product] = product_coefficients

        dump_json('', self.output()['calibration_coefficients'].path, calibration_coefficients_dict)
