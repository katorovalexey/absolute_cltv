from typing import List, Dict

import pandas as pd
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, TrainValidationSplit
from pyspark.sql import DataFrame, functions as F


class SparkRegressor:
    def __init__(
            self,
            dataset: DataFrame,
            features: List[str],
            valid_column: str,
            target_column: str,
            predict_column: str
    ):
        self._predict_column = predict_column
        self._target_column = target_column
        self._valid_column = valid_column
        self._features = features
        self._dataset = dataset

    def _get_loss(self, loss: str) -> [str, str]:

        if loss == 'mae':
            loss_name, metric_name = 'absolute', 'mae'
        else:
            loss_name, metric_name = 'squared', 'rmse'
        return loss_name, metric_name

    def _assembler(self, input_cols) -> VectorAssembler:
        return VectorAssembler(
            inputCols=input_cols,
            outputCol='all_features'
        )

    def _random_forest(self) -> RandomForestRegressor:
        return RandomForestRegressor(
            featuresCol='all_features',
            labelCol=self._target_column,
            predictionCol=self._predict_column,
            seed=23,
            maxDepth=25,
            cacheNodeIds=False,
            numTrees=200,
            subsamplingRate=0.05,
            featureSubsetStrategy='0.05'
        )

    def _gb(self, loss_name: str) -> GBTRegressor:
        return GBTRegressor(
            featuresCol='all_features',
            labelCol=self._target_column,
            predictionCol=self._predict_column,
            seed=23,
            lossType=loss_name,
            maxIter=10
        )

    def _evaluator(self, metric_name: str) -> RegressionEvaluator:
        return RegressionEvaluator(
            labelCol=self._target_column,
            predictionCol=self._predict_column,
            metricName=metric_name
        )

    def _metrics(self, evaluator, valid_data_with_predictions):

        metrics = dict()

        evaluator.setMetricName('mae')
        metrics['mae'] = evaluator.evaluate(valid_data_with_predictions)
        evaluator.setMetricName('rmse')
        metrics['rmse'] = evaluator.evaluate(valid_data_with_predictions)
        metrics['sum'] = (
            valid_data_with_predictions
                .select((F.sum(F.col(self._predict_column)) / F.sum(F.col(self._target_column))).alias('sum'))
                .collect()[0]['sum']
        )
        return metrics

    def _feature_importances(self, features, model_estimator):

        feature_importances = pd.DataFrame()
        feature_importances['features'] = features
        feature_importances['importance'] = model_estimator.featureImportances.toArray()
        return feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

    def _subsampling_rate(self):
        return 0.1

    def _filter_features(self):
        train_data = self._dataset.filter(F.col(self._valid_column) == F.lit(False)).sample(0.1, seed=23).repartition(50)
        assembler = self._assembler(input_cols=self._features)

        estimator = self._random_forest()
        estimator.setMaxDepth(5)
        estimator.setFeatureSubsetStrategy('0.2')
        estimator.setSubsamplingRate(0.1)
        estimator.setNumTrees(100)

        model = Pipeline(stages=[assembler, estimator])
        rf_transformer = model.fit(train_data)
        return self._feature_importances(self._features, rf_transformer.stages[1]).head(300)['features'].to_list()

    def _fit_model_with_filtered_features(
            self,
            model_type: str = 'rf',
            loss: str = None
    ) -> [Pipeline, Dict, pd.DataFrame]:

        train_data = self._dataset.filter(F.col(self._valid_column) == F.lit(False)).sample(0.1, seed=46).repartition(50)
        valid_data = self._dataset.filter(F.col(self._valid_column) == F.lit(True)).sample(0.1, seed=46).repartition(50)
        features = self._filter_features()

        assembler = self._assembler(input_cols=features)
        loss_name, metric_name = self._get_loss(loss)
        evaluator = self._evaluator(metric_name)

        if model_type == 'gb':

            estimator = self._gb(loss_name)

            paramGrid = (
                ParamGridBuilder()
                    .addGrid(estimator.maxDepth, [3, 6, 9])
                    .addGrid(estimator.stepSize, [0.1, 0.01, 0.001])
                    .addGrid(estimator.maxIter, [10, 50, 120])
                    .build()
            )

            pipeline = Pipeline(stages=[assembler, estimator])

            model = TrainValidationSplit(
                estimator=pipeline,
                estimatorParamMaps=paramGrid,
                evaluator=evaluator,
                trainRatio=0.7
            )

        elif model_type == 'rf':
            estimator = self._random_forest()
            estimator.setMaxDepth(5)
            estimator.setFeatureSubsetStrategy('0.8')
            estimator.setSubsamplingRate(0.1)
            estimator.setNumTrees(100)
            model = Pipeline(stages=[assembler, estimator])

        else:
            model = None

        fitted_model = model.fit(train_data)

        if model_type == 'gb':
            model_estimator = fitted_model.bestModel.stages[1]
            model_transformer = fitted_model.bestModel
        elif model_type == 'rf':
            model_estimator = fitted_model.stages[1]
            model_transformer = fitted_model
        else:
            model_estimator = None
            model_transformer = None

        valid_data_with_predictions = model_transformer.transform(valid_data)
        metrics = self._metrics(evaluator, valid_data_with_predictions)
        feature_importances = self._feature_importances(features, model_estimator)

        return model_transformer, metrics, feature_importances

    def fit_model(self, model_type, loss=None):

        model_transformer, metrics, feature_importances = self._fit_model_with_filtered_features(model_type, loss)

        # feature selection

        feature_importances_sorted = feature_importances.sort_values('importance', ascending=False).reset_index(
            drop=True)
        imp_values = feature_importances_sorted['importance'].cumsum().copy()

        border = 0.95
        features_number_border = 100

        if imp_values[imp_values < border].shape[0] < features_number_border:
            while border < 0.998 and imp_values[imp_values < border].shape[0] < features_number_border:
                border += 0.001
        else:
            while border > 0.9 and imp_values[imp_values < border].shape[0] >= features_number_border:
                border -= 0.001

        number_of_top_features = imp_values[imp_values < border].shape[0]

        feature_importances_new = feature_importances_sorted.head(number_of_top_features)
        short_features_list = feature_importances_new['features'].to_list()

        # refitting

        valid_data = self._dataset.filter(F.col(self._valid_column) == True).repartition(50)
        assembler = self._assembler(input_cols=short_features_list)
        loss_name, metric_name = self._get_loss(loss)
        evaluator = self._evaluator(metric_name)

        if model_type == 'gb':

            train_data = self._dataset
            estimator = self._gb(loss_name)
            estimator.setMaxDepth(model_transformer.stages[1]._java_obj.getMaxDepth())
            estimator.setStepSize(model_transformer.stages[1]._java_obj.getStepSize())
            estimator.setMaxIter(model_transformer.stages[1]._java_obj.getMaxIter())

        elif model_type == 'rf':

            train_data = self._dataset.filter(F.col(self._valid_column) == F.lit(False)).repartition(150)
            estimator = self._random_forest()

            estimator.setMaxDepth(25)
            estimator.setFeatureSubsetStrategy('0.9')
            estimator.setSubsamplingRate(self._subsampling_rate())
            estimator.setNumTrees(100)

        else:
            train_data = None
            estimator = None

        model = Pipeline(stages=[assembler, estimator])
        fitted_model = model.fit(train_data)

        model_estimator_new = fitted_model.stages[1]
        model_transformer_new = fitted_model

        valid_data_with_predictions = model_transformer_new.transform(valid_data)
        metrics_new = self._metrics(evaluator, valid_data_with_predictions)
        feature_importances_new = self._feature_importances(short_features_list, model_estimator_new)

        return model_transformer, model_transformer_new, \
               metrics, metrics_new, \
               feature_importances, feature_importances_new
