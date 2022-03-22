import logging
import os
from itertools import chain

import luigi
from luigi.contrib.hdfs import HdfsTarget
from pyspark.sql import functions as F, SparkSession
from pyspark.sql.types import FloatType

from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json, dump_json
from absolute_cltv.total_pnl.tasks.data_collection.mega_table_filtering import MegaTableFiltering
from absolute_cltv.total_pnl.tasks.feature_engineering.mega_table_train_transformers import \
    MegaTableTrainTransformers

logger = logging.getLogger(__name__)


class MegaTableTransform(PySparkTask):
    local_result_path = luigi.Parameter()

    def requires(self):
        return {
            'transformers': MegaTableTrainTransformers(),
            'dataset': MegaTableFiltering()
        }

    def output(self):
        return {
            'df_mega_table_mapped': HdfsTarget(self.hdfs_target_path()),
            'mega_features_groups': luigi.LocalTarget(os.path.join(
                str(self.local_result_path),
                'mega_features_groups.json'
            ))
        }

    def _save_output(self, mega_features_groups, df_mega_table_mapped):
        dump_json('', self.output()['mega_features_groups'].path, mega_features_groups)
        df_mega_table_mapped.write.parquet(self.output()['df_mega_table_mapped'].path)

    def main(self, spark: SparkSession):

        mega_features_categorical_mapping = load_json(
            str(self.local_result_path),
            'mega_features_categorical_mapping.json'
        )

        mega_features_fill_na = load_json(
            str(self.local_result_path),
            'mega_features_fill_na.json'
        )

        df_mega_table_filtered = spark.read.parquet(self.input()['dataset'].path)

        features_date = [a for a in mega_features_fill_na.keys() if a.split('_')[-1] == 'dt']

        select_date = [
                          F.col(feature)
                          for feature in mega_features_fill_na if feature not in features_date
                      ] + [
                          F.datediff(F.to_timestamp(F.col('report_dt')), F.col(feature)).alias(feature)
                          for feature in mega_features_fill_na if feature in features_date
                      ]

        mega_features_fill_na_new = dict()
        mean_features_to_fill = []
        no_mean_features_to_fill = []
        for feature in mega_features_fill_na.keys():
            value = mega_features_fill_na[feature]
            if len(value) > 1:
                mega_features_fill_na_new[feature] = value[0]
                mega_features_fill_na_new[f'{feature}_mean_na'] = value[1]
                mean_features_to_fill += [feature]
            else:
                mega_features_fill_na_new[feature] = value[0]
                no_mean_features_to_fill += [feature]

        select_rename = [
                            F.col(a) for a in no_mean_features_to_fill + mean_features_to_fill
                        ] + [
                            F.col(feature).alias(f'{feature}_mean_na') for feature in mean_features_to_fill
                        ]

        features = (
                no_mean_features_to_fill +
                mean_features_to_fill +
                [f'{feature}_mean_na' for feature in mean_features_to_fill]
        )

        select_no_mapping = [
            F.col(feature) for feature in features if feature not in mega_features_categorical_mapping.keys()
        ]

        select_new_values = select_no_mapping.copy()
        select_mapping = select_no_mapping.copy()

        for feature in mega_features_categorical_mapping.keys():
            feature_mapping = mega_features_categorical_mapping[feature]

            select_new_values += [
                F.when(F.col(feature).isin(list(feature_mapping.keys())), F.col(feature))
                    .otherwise('other_value').alias(feature)
            ]

            select_mapping += [
                F.create_map(
                    [F.lit(a) for a in chain(*mega_features_categorical_mapping[feature].items())]
                )[F.col(feature)].cast(FloatType()).alias(feature)
            ]

        index_columns = ['epk_id', 'report_dt']
        df_mega_table_mapped = (
            df_mega_table_filtered
                .select(*index_columns, *select_date)
                .select(*index_columns, *select_rename)
                .na.fill(mega_features_fill_na_new)
                .select(*index_columns, *select_new_values)
                .select(*index_columns, *select_mapping)
        )

        features_numeric = [
                               f'{feature}_mean_na' for feature in mean_features_to_fill
                           ] + mean_features_to_fill

        features_categorical = [
            a for a in features if a not in (features_numeric + features_date)
        ]

        mega_features_groups = {
            'features':
                features_numeric + features_date + features_categorical,
            'types': {
                'numeric': features_numeric,
                'date': features_date,
                'categorical': features_categorical
            }
        }

        self._save_output(mega_features_groups, df_mega_table_mapped)
