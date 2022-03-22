import logging
import os

import luigi
from luigi import LocalTarget

from cltv_tools.luigi.spark.tasks import PySparkTask
from cltv_tools.utils.json import dump_json
from absolute_cltv.total_pnl.artifacts.generate_config_for_epk_date_filters import generate_config_for_epk_date_filters

logger = logging.getLogger(__name__)


class EpkDateConfigGenerating(PySparkTask):

    local_result_path = luigi.Parameter()
    hdfs_project_path = luigi.Parameter()
    hdfs_result_path = luigi.Parameter()

    count_history_months = luigi.IntParameter()
    count_predict_months = luigi.IntParameter()

    report_dt_test_left_border = luigi.Parameter()
    report_dt_test_count = luigi.IntParameter()
    report_dt_train_left_border = luigi.Parameter()
    report_dt_train_count = luigi.IntParameter()

    number_of_epk_bins = luigi.IntParameter()
    epk_left_bin_border_train = luigi.IntParameter()
    epk_left_bin_border_test = luigi.IntParameter()
    epk_bins_count_train = luigi.IntParameter()
    epk_bins_count_test = luigi.IntParameter()

    def output(self):
        return LocalTarget(os.path.join(
            str(self.local_result_path),
            'epk_date_config_generating.json'
        ))

    def main(self, **kwargs):

        config = generate_config_for_epk_date_filters(
            count_predict_months=self.count_predict_months,
            count_history_months=self.count_history_months,
            report_dt_test_left_border=self.report_dt_test_left_border,
            report_dt_test_count=self.report_dt_test_count,
            report_dt_train_left_border=self.report_dt_train_left_border,
            report_dt_train_count=self.report_dt_train_count,
            number_of_epk_bins=self.number_of_epk_bins,
            epk_left_bin_border_train=self.epk_left_bin_border_train,
            epk_left_bin_border_test=self.epk_left_bin_border_test,
            epk_bins_count_train=self.epk_bins_count_train,
            epk_bins_count_test=self.epk_bins_count_test
        )

        for path in [self.hdfs_project_path, self.hdfs_result_path]:
            os.popen(f'hdfs dfs -mkdir {path}').read()
        os.popen(f'mkdir {self.local_result_path}').read()

        dump_json('', self.output().path, config)

    def run(self):
        self.main()
