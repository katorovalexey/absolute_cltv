import logging
from typing import Dict

import luigi
import os

import pandas as pd
from luigi import LocalTarget
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import SparkSession, functions as F, DataFrame
from sklearn.metrics import roc_auc_score, roc_curve, auc

from absolute_cltv.total_pnl.tasks.modelling.combine_predictions import CombinePredictions
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json, dump_json

logger = logging.getLogger(__name__)


class Metrics(PySparkTask):
    local_result_path = luigi.Parameter()
    hdfs_project_path = luigi.Parameter()
    pnl_products_info_path = luigi.Parameter()

    def requires(self):
        return {
            'predictions': CombinePredictions()
        }

    def output(self):
        return {
            'metrics': LocalTarget(
                os.path.join(
                    str(self.local_result_path),
                    'models_params',
                    'metrics_all.json'
                )
            )
        }

    def _regression_metrics(self, spark_df: DataFrame, target_col: str, predict_col: str) -> Dict:
        metrics = dict()
        evaluator = RegressionEvaluator(labelCol=target_col, predictionCol=predict_col)

        evaluator.setMetricName('mae')
        metrics['mae'] = evaluator.evaluate(spark_df)
        evaluator.setMetricName('rmse')
        metrics['rmse'] = evaluator.evaluate(spark_df)
        metrics['sum'] = (
            spark_df
                .select((F.sum(F.col(predict_col)) / F.sum(F.col(target_col))).alias('sum'))
                .collect()[0]['sum']
        )
        return metrics

    def _classification_metrics(self, pd_df: pd.DataFrame, target_col: str, predict_col: str) -> Dict:
        metrics = dict()
        metrics['ROC_AUC'] = roc_auc_score(pd_df[target_col], pd_df[predict_col])
        fpr, tpr, thresholds = roc_curve(pd_df[target_col], pd_df[predict_col])
        metrics['Precision_AUC'] = auc(fpr, tpr)
        metrics['sum'] = pd_df[predict_col].sum() / pd_df[target_col].sum()

        return metrics

    def main(self, spark: SparkSession):
        # load data

        dataset = spark.read.parquet(self.input()['predictions']['combine_predictions'].path)
        pnl_products_info = load_json('', str(self.pnl_products_info_path))

        # predictions

        metrics = {
            'products': dict(),
            'total': dict()
        }

        for product in list(pnl_products_info.keys()):
            metrics['products'][product] = dict()

            # 1

            model_part = 'rich_regression'

            valid_metrics = load_json(
                os.path.join(str(self.local_result_path), 'models_params'),
                f'metrics_{product}_{model_part}.json'
            )

            metrics['products'][product][f'{model_part}_valid_all_features'] = valid_metrics['all_features']
            metrics['products'][product][f'{model_part}_valid_only_important_features'] = valid_metrics[
                'only_important_features']

            predict = f'{product}_rich_pl_predict'
            target = f'{product}_pl_target'
            naive = f'{product}_pl_-1'
            df = dataset.filter(F.col(f'{product}_pl_-1') != 0)
            metrics['products'][product][f'{model_part}_test_model'] = self._regression_metrics(df, target, predict)
            metrics['products'][product][f'{model_part}_test_naive'] = self._regression_metrics(df, target, naive)

            # 2

            model_part = 'poor_regression'

            valid_metrics = load_json(
                os.path.join(str(self.local_result_path), 'models_params'),
                f'metrics_{product}_{model_part}.json'
            )

            metrics['products'][product][f'{model_part}_valid_all_features'] = valid_metrics['all_features']
            metrics['products'][product][f'{model_part}_valid_only_important_features'] = valid_metrics[
                'only_important_features'
            ]

            predict = f'{product}_rich_pl_predict'
            target = f'{product}_pl_target'
            df = dataset.filter(
                (F.col(f'{product}_pl_-1') == 0) &
                (F.col(f'{product}_pl_target') != 0)
            )
            metrics['products'][product][f'{model_part}_test'] = self._regression_metrics(df, target, predict)

            # 3

            model_part = 'poor_classification'

            valid_metrics = load_json(
                os.path.join(str(self.local_result_path), 'models_params'),
                f'metrics_{product}_{model_part}.json'
            )

            metrics['products'][product][f'{model_part}_valid_all_features'] = valid_metrics['all_features']
            metrics['products'][product][f'{model_part}_valid_only_important_features'] = valid_metrics[
                'only_important_features'
            ]

            predict = f'{product}_sales_pl_predict_probability'
            target = f'{product}_sales_pl_target'
            df = (
                dataset
                    .filter(F.col(f'{product}_pl_-1') == 0)
                    .select(
                    F.col(predict).alias('predict'),
                    F.col(target).alias('target')
                )
                    .toPandas()
            )

            metrics['products'][product][f'{model_part}_test'] = self._classification_metrics(df, 'target', 'predict')

            # 4

            model_part = 'poor'

            predict = f'{product}_poor_mix_pl_predict'
            target = f'{product}_pl_target'
            naive = f'{product}_pl_-1'
            df = dataset.filter(F.col(f'{product}_pl_-1') == 0)

            metrics['products'][product][f'{model_part}_test_model'] = self._regression_metrics(df, target, predict)
            metrics['products'][product][f'{model_part}_test_naive'] = self._regression_metrics(df, target, naive)

            # 5

            model_part = 'product_total'

            predict = f'{product}_total_pl_predict'
            target = f'{product}_pl_target'
            naive = f'{product}_pl_-1'
            df = dataset

            metrics['products'][product][f'{model_part}_test_model'] = self._regression_metrics(df, target, predict)
            metrics['products'][product][f'{model_part}_test_naive'] = self._regression_metrics(df, target, naive)

        # 6 total PnL

        'AllProducts'

        predict_mix = 'total_pl_predict'
        predict_total = 'AllProducts_total_pl_predict'
        target = f'total_pl_target'
        naive = f'total_pl_-1_year'

        df = dataset.select(
            '*',
            (F.col('total_pl_-1') * 12).alias(naive)
        )

        metrics['total']['test_model_mix'] = self._regression_metrics(df, target, predict_mix)
        metrics['total']['test_model_total'] = self._regression_metrics(df, target, predict_total)
        metrics['total']['test_naive'] = self._regression_metrics(df, target, naive)

        dump_json('', self.output()['metrics'].path, metrics)
