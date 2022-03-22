import logging

import luigi
import os
from luigi import LocalTarget
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import SparkSession, DataFrame

from absolute_cltv.total_pnl.artifacts.spark_regressor import SparkRegressor
from absolute_cltv.total_pnl.tasks.modelling.train_test_split import TrainTestSplit
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json, dump_json

logger = logging.getLogger(__name__)


class TrainModelTemplate(PySparkTask):
    product = luigi.Parameter()
    model_name = luigi.Parameter()
    model_type = luigi.Parameter()
    loss = luigi.Parameter()
    local_result_path = luigi.Parameter()
    hdfs_project_path = luigi.Parameter()

    def requires(self):
        return TrainTestSplit()

    def output(self):
        return {
            'feature_importance': LocalTarget(
                os.path.join(
                    self._local_models_params_path(),
                    f'feature_importance_{str(self.product)}_{str(self.model_name)}.xlsx'
                )
            ),
            'metrics': LocalTarget(
                os.path.join(
                    self._local_models_params_path(),
                    f'metrics_{str(self.product)}_{str(self.model_name)}.json'
                )
            ),
            'model': HdfsTarget(
                os.path.join(
                    self._hdfs_models_path(),
                    f'model_{str(self.product)}_{str(self.model_name)}'
                )
            )
        }

    def _get_dataset(self, spark: SparkSession) -> DataFrame:
        return spark.read.parquet(self.input()['train'].path)

    def _hdfs_models_path(self) -> str:
        return os.path.join(str(self.hdfs_project_path), 'models')

    def _local_models_params_path(self) -> str:
        return os.path.join(str(self.local_result_path), 'models_params')

    def _target(self) -> str:
        return f'{str(self.product)}_pl_target'

    def _predict(self) -> str:
        return f'{str(self.product)}_pl_predict'

    def _loss(self) -> str:
        return(str(self.loss))

    def _model(self, dataset, features, valid_column, target_column, predict_column):
        return SparkRegressor(
            dataset=dataset,
            features=features,
            valid_column=valid_column,
            target_column=target_column,
            predict_column=predict_column
        )

    def main(self, spark: SparkSession):
        # load info about features

        pnl_dataset_columns_groups = load_json(
            str(self.local_result_path),
            'pnl_dataset_columns_groups.json'
        )
        pnl_features = pnl_dataset_columns_groups['features'] + pnl_dataset_columns_groups['naive_total_pnl']

        mega_features = load_json(
            str(self.local_result_path),
            'mega_features_groups.json'
        )['features']

        features = mega_features + pnl_features

        # fit model

        model = self._model(
            dataset=self._get_dataset(spark),
            features=features,
            valid_column='is_valid',
            target_column=str(self._target()),
            predict_column=str(self._predict())
        )

        model_transformer_old, model_transformer_new, metrics_old, metrics_new, feature_importances_old, \
        feature_importances_new = model.fit_model(
            model_type=str(self.model_type),
            loss=self._loss()
        )

        metrics = {
            'all_features': metrics_old,
            'only_important_features': metrics_new
        }

        os.popen(f" mkdir {self._local_models_params_path()}")
        os.popen(f" hdfs dfs -mkdir {self._hdfs_models_path()}").read()

        feature_importances_new.to_excel(self.output()['feature_importance'].path)
        dump_json('', self.output()['metrics'].path, metrics)
        model_transformer_new.write().save(self.output()['model'].path)
