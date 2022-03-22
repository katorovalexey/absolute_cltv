import numpy as np
import pandas as pd


def calibration_coefficients(df: pd.DataFrame, n_bins=10, target_col='target', predict_col='predict'):
    df['bin'] = pd.qcut(df[predict_col], n_bins, labels=False, duplicates='drop')

    if df[predict_col].sum() == 0 or df[target_col].sum() == 0:
        default_coef = 1
    else:
        default_coef = df[target_col].sum() / df[predict_col].sum()

    left_border_max_bin = df[df['bin'] == n_bins][predict_col].min()
    right_border_min_bin = df[df['bin'] == 0][predict_col].max()

    if (
            df['bin'].nunique() != n_bins or
            (left_border_max_bin <= 0 and left_border_max_bin >= 1) or
            (right_border_min_bin <= 0 and right_border_min_bin >= 1)
    ):
        calibration_coefficients = [
            [round(a * 1 / n_bins, 2), round(a * 1 / n_bins + 1 / n_bins, 2), default_coef] for a in range(n_bins)
        ]

    else:
        calibration_coefficients = []

        for bin_number in range(n_bins):

            bin_df = df[df['bin'] == bin_number]

            left_prob_border = bin_df[predict_col].min()
            right_prob_border = bin_df[predict_col].max()

            mean_prob_predict = bin_df[predict_col].mean()
            mean_prob_target = bin_df[target_col].sum() / bin_df.shape[0]

            if mean_prob_predict == 0:
                coef = 1
            else:
                coef = mean_prob_target / mean_prob_predict
                if np.abs(1 - coef) >= 0.9999:
                    coef = 1

            if bin_number == 0:
                left_prob_border = 0
            if bin_number == n_bins - 1:
                right_prob_border = 1

            calibration_coefficients += [[left_prob_border, right_prob_border, coef]]
    return calibration_coefficients
