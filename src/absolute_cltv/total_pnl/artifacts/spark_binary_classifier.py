from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.sql import functions as F

from absolute_cltv.total_pnl.artifacts.spark_regressor import SparkRegressor


class SparkBinaryClassifier(SparkRegressor):

    def _random_forest(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            featuresCol='all_features',
            labelCol=self._target_column,
            predictionCol=self._predict_column,
            rawPredictionCol=f"{self._predict_column}_raw",
            probabilityCol=f"{self._predict_column}_probability",
            seed=23,
            maxDepth=25,
            cacheNodeIds=False,
            numTrees=200,
            featureSubsetStrategy='0.05'
        )

    def _get_loss(self, loss: str) -> [str, str]:
        return None, 'areaUnderPR'

    def _gb(self, loss_name=None) -> GBTClassifier:
        gb_classifier = GBTClassifier(
            featuresCol='all_features',
            labelCol=self._target_column,
            predictionCol=self._predict_column,
            seed=23,
            maxIter=5
        )

        gb_classifier.setRawPredictionCol(f"{self._predict_column}_raw")
        gb_classifier.setProbabilityCol(f"{self._predict_column}_probability")

        return gb_classifier

    def _evaluator(self, metric_name: str) -> BinaryClassificationEvaluator:
        return BinaryClassificationEvaluator(
            rawPredictionCol=f"{self._predict_column}_raw",
            labelCol=self._target_column,
            metricName=metric_name
        )

    def _metrics(self, evaluator, valid_data_with_predictions):
        metrics = dict()

        evaluator.setMetricName('areaUnderROC')
        metrics['ROC_AUC'] = evaluator.evaluate(valid_data_with_predictions)
        evaluator.setMetricName('areaUnderPR')
        metrics['Precision_AUC'] = evaluator.evaluate(valid_data_with_predictions)

        metrics['sum'] = (
            valid_data_with_predictions
                .select(
                F.col(self._target_column).alias('target'),
                vector_to_array(f'{self._predict_column}_probability').getItem(1).alias('positive_probability')
            )
                .select((F.sum(F.col('positive_probability')) / F.sum(F.col('target'))).alias('sum'))
                .collect()[0]['sum']
        )
        return metrics

    def _subsampling_rate(self):
        return 0.1
