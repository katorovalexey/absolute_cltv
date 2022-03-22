import logging

from pyspark.sql import SparkSession, DataFrame, functions as F
from absolute_cltv.total_pnl.artifacts.spark_binary_classifier import SparkBinaryClassifier
from absolute_cltv.total_pnl.tasks.modelling.train_model_template import TrainModelTemplate
from absolute_cltv.total_pnl.tasks.modelling.train_test_split import TrainTestSplit

logger = logging.getLogger(__name__)


class TrainRichRegressionModel(TrainModelTemplate):

    def _get_dataset(self, spark: SparkSession) -> DataFrame:
        return spark.read.parquet(self.input()['train'].path).filter(F.col(f'{str(self.product)}_pl_-1') != 0)

    def _predict(self) -> str:
        return f'{str(self.product)}_rich_pl_predict'


class TrainPoorRegressionModel(TrainModelTemplate):

    def _get_dataset(self, spark: SparkSession) -> DataFrame:
        return spark.read.parquet(self.input()['train'].path).filter(
            (F.col(f'{str(self.product)}_pl_-1') == 0) &
            (F.col(f'{str(self.product)}_pl_target') != 0)
        )

    def _predict(self) -> str:
        return f'{str(self.product)}_poor_pl_predict'


class TrainPoorClassificationModel(TrainModelTemplate):

    def _get_dataset(self, spark: SparkSession) -> DataFrame:
        return spark.read.parquet(self.input()['train'].path).filter(
            F.col(f'{str(self.product)}_pl_-1') == 0
        )

    def _target(self) -> str:
        return f'{str(self.product)}_sales_pl_target'

    def _predict(self) -> str:
        return f'{str(self.product)}_sales_pl_predict'

    def _loss(self) -> None:
        return None

    def _model(self, dataset, features, valid_column, target_column, predict_column):
        return SparkBinaryClassifier(
            dataset=dataset,
            features=features,
            valid_column='is_valid',
            target_column=str(self._target()),
            predict_column=str(self._predict())
        )
class TrainTotalRegressionModel(TrainModelTemplate):

    def requires(self):
        return TrainTestSplit()

    def _get_dataset(self, spark: SparkSession) -> DataFrame:
        return spark.read.parquet(self.input()['train'].path)

    def _target(self) -> str:
        return 'total_pl_target'

    def _predict(self) -> str:
        return f'{str(self.product)}_total_pl_predict'




