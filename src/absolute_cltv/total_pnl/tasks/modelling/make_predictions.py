import logging

import luigi
import os
from luigi.contrib.hdfs import HdfsTarget
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F

from absolute_cltv.total_pnl.tasks.modelling.train_models import TrainRichRegressionModel, TrainPoorRegressionModel, \
    TrainPoorClassificationModel, TrainTotalRegressionModel
from absolute_cltv.total_pnl.tasks.modelling.train_test_split import TrainTestSplit
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json

logger = logging.getLogger(__name__)


class MakePredictions(PySparkTask):
    local_result_path = luigi.Parameter()
    hdfs_project_path = luigi.Parameter()
    pnl_products_info_path = luigi.Parameter()

    def requires(self):
        pnl_products_info = load_json('', str(self.pnl_products_info_path))

        return_dict = dict()
        return_dict['dataset'] = self._dataset_task()

        for product in list(pnl_products_info.keys()):
            return_dict[f'rich_regression_{product}'] = TrainRichRegressionModel(**{
                'product': product,
                'model_name': 'rich_regression',
                'model_type': 'gb',
                'loss': 'mse'
            })

            return_dict[f'poor_regression_{product}'] = TrainPoorRegressionModel(**{
                'product': product,
                'model_name': 'poor_regression',
                'model_type': 'gb',
                'loss': 'mse'
            })

            return_dict[f'poor_classification_{product}'] = TrainPoorClassificationModel(**{
                'product': product,
                'model_name': 'poor_classification',
                'model_type': 'gb',
                'loss': None
            })

            return_dict[f'total_regression_AllProducts'] = TrainTotalRegressionModel(**{
                'product': 'AllProducts',
                'model_name': 'total_regression',
                'model_type': 'gb',
                'loss': 'mse'
            })

        return return_dict

    def output(self):
        return {'predictions': HdfsTarget(self.hdfs_target_path())}

    def _dataset_task(self):
        return TrainTestSplit()

    def _input_dataset(self, spark: SparkSession):
        return spark.read.parquet(self.input()['dataset']['test'].path)

    def _main_without_saving(self, spark: SparkSession):

        # load data

        dataset = self._input_dataset(spark)

        # filter columns

        pnl_dataset_columns_groups = load_json(
            str(self.local_result_path),
            'pnl_dataset_columns_groups.json'
        )
        del pnl_dataset_columns_groups['features']

        columns = ['epk_id', 'report_dt']

        for columns_group in pnl_dataset_columns_groups.values():
            columns += columns_group

        # make predictions

        pnl_products_info = load_json('', str(self.pnl_products_info_path))
        model_path = os.path.join(str(self.hdfs_project_path), 'models')

        for product in list(pnl_products_info.keys()):
            for model_part in ['rich_regression', 'poor_regression', 'poor_classification']:

                model = PipelineModel.load(os.path.join(model_path, f'model_{product}_{model_part}'))
                dataset = model.transform(dataset)
                dataset = dataset.drop('all_features')

                if model_part.find('regression') >= 0:
                    columns += [model.stages[1].getPredictionCol()]
                else:
                    prob_col = model.stages[1].getProbabilityCol()

                    dataset = (
                        dataset
                            .select(
                            '*',
                            vector_to_array(prob_col).getItem(1).alias(f'{prob_col}_probability')
                        )
                            .drop(prob_col)
                            .withColumnRenamed(f'{prob_col}_probability', prob_col)
                    )

                    columns += [prob_col]

        model = PipelineModel.load(os.path.join(model_path, f'model_AllProducts_total_regression'))
        columns += [model.stages[1].getPredictionCol()]
        dataset = model.transform(dataset).drop('all_features')

        return dataset, columns

    def main(self, spark: SparkSession):

        dataset, columns = self._main_without_saving(spark)
        dataset.select(*[F.col(column) for column in columns]).write.parquet(self.output()['predictions'].path)
