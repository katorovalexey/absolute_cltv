import logging
import os

import luigi
from luigi import LocalTarget
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from absolute_cltv.total_pnl.artifacts.client_and_month_filters import ArtifactsClientMonth
from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import load_json, dump_json
from absolute_cltv.total_pnl.tasks.data_collection.mega_table_filtering import MegaTableFiltering

logger = logging.getLogger(__name__)


class MegaTableTrainTransformers(PySparkTask):
    mega_features_postfix_groups_path = luigi.Parameter()
    local_result_path = luigi.Parameter()

    def requires(self):
        return MegaTableFiltering()

    def output(self):
        return {
            'mega_features_categorical_mapping': LocalTarget(
                os.path.join(
                str(self.local_result_path),
                'mega_features_categorical_mapping.json'
            )
            ),
            'mega_features_fill_na': LocalTarget(os.path.join(
                str(self.local_result_path),
                'mega_features_fill_na.json'
            ))
        }

    def main(self, spark: SparkSession):

        # load data from last step

        df_mega_table_filtered_full = spark.read.parquet(self.input().path)

        # load info about epk_id hash groups and report_dt values

        epk_bins_and_report_dt_pairs = load_json(str(self.local_result_path), 'epk_date_config_generating.json')

        artifacts_client_month = ArtifactsClientMonth(
            spark=spark,
            epk_bins_and_report_dt_pairs=epk_bins_and_report_dt_pairs['train']
        )

        condition_to_filter_epk_month_pairs_train = \
            artifacts_client_month.condition_to_filter_epk_month_pairs(table='mega')

        # filter clients/dates pairs by hash

        df_mega_table_filtered = (
            df_mega_table_filtered_full
                .select(F.col('*'), F.abs(F.hash(F.col('epk_id').cast('string'))).alias('epk_hash'))
                .filter(condition_to_filter_epk_month_pairs_train)
                .drop(F.col('epk_hash'))
                .repartition(200)
        )

        # features types preprocessing

        mega_features_postfix_groups = load_json('', str(self.mega_features_postfix_groups_path))
        features = [column for column in df_mega_table_filtered.columns if column not in ['epk_id', 'report_dt']]
        mega_max_uniq_values_categorical_str_features = 500

        features_types = dict(
            zip([a[0] for a in df_mega_table_filtered.dtypes],
                [a[1] for a in df_mega_table_filtered.dtypes])
        )

        # fill_na values for numeric features

        mega_features_fill_na = dict()
        select_means_numeric_features = []

        for feature in features:
            if feature.split('_')[-1] in mega_features_postfix_groups['numeric']:
                select_means_numeric_features += [F.mean(F.col(feature)).alias(feature)]
                mega_features_fill_na[feature] = None

        row_of_numeric_means = df_mega_table_filtered.select(*select_means_numeric_features).rdd.collect()[0]

        for feature in mega_features_fill_na.keys():

            feature_mean = row_of_numeric_means[feature]
            if feature_mean == None:
                mean_insert = 0
            else:
                mean_insert = float(feature_mean)
            mega_features_fill_na[feature] = [0, mean_insert]

        # fill_na values for data and categorical

        features_categorical_str = []
        for feature in features:
            if feature.split('_')[-1] in mega_features_postfix_groups['date']:
                mega_features_fill_na[feature] = [0]
            elif feature.split('_')[-1] in mega_features_postfix_groups['categorical']:
                if features_types[feature] == 'string':
                    mega_features_fill_na[feature] = ['other_value']
                    features_categorical_str += [feature]
                else:
                    mega_features_fill_na[feature] = [-9876]

        # mapping for categorical string features

        mega_features_categorical_mapping = dict()

        for feature in features_categorical_str:
            df_map = (
                df_mega_table_filtered
                    .select(F.col(feature).alias('values'))
                    .na.fill('other_value')
                    .groupby(F.col('values'))
                    .agg(F.count('*').alias('value_count'))
                    .cache()
            )

            if df_map.count() < mega_max_uniq_values_categorical_str_features:

                mapping = dict()
                df_map_pd = df_map.toPandas().sort_values('value_count')
                counts_sum = df_map_pd['value_count'].sum()

                for value_count in df_map_pd['value_count'].unique():

                    df_short = df_map_pd[df_map_pd['value_count'] == value_count].copy()
                    n_rows = df_short.shape[0]

                    if n_rows == 1:
                        df_short['delta'] = df_short['value_count'] / counts_sum
                        mapping.update(
                            zip(df_short['values'], df_short['delta'].astype(str))
                        )

                    elif n_rows > 1:
                        df_short['delta'] = ((df_short.reset_index(drop=True).index / (df_short.shape[0]))
                                             + df_short['value_count']) / counts_sum
                        mapping.update(zip(df_short['values'], df_short['delta'].astype(str)))

                if 'other_value' in mapping.keys():
                    if float(mapping['other_value']) == 0:
                        mapping['other_value'] = str(-1)
                else:
                    mapping['other_value'] = str(-1)

                mega_features_categorical_mapping[feature] = mapping
            else:
                del mega_features_fill_na[feature]

            df_map.unpersist()

        dump_json('', self.output()['mega_features_categorical_mapping'].path, mega_features_categorical_mapping)
        dump_json('', self.output()['mega_features_fill_na'].path, mega_features_fill_na)