"""
7_cycle_autocorrelations.py
Compute the autocorrelation of some cycle-by-cycle features
for each shank recording.

$ time python 7_cycle_autocorrelations.py

Time: ~1.5 minutes
"""

from collections import defaultdict
import pickle
from scipy import stats
import numpy as np
import pandas as pd
from config import config_dict
import util


def compute_cycle_autocorr(df_shape, feat_name,
                           autocorr_delays=np.arange(1, 101),
                           corr_type='spearman'):
    """Compute the autocorrelation for a single feature"""

    # Compute time series of cycle rdsyms with nans in place for invalid
    # cycles
    max_cycle = df_shape.index.values[-1] + 1
    feat_by_cycle = np.ones(max_cycle) * np.nan
    feat_by_cycle[df_shape.index.values] = df_shape[feat_name].values

    # Compute autocorrelation
    autocorr_rs = np.zeros(len(autocorr_delays))
    for i, d in enumerate(autocorr_delays):
        # Determine pairs of interest for each time lag (neither NaN)
        x1 = feat_by_cycle[:-d]
        x2 = feat_by_cycle[d:]
        nan_idxs = np.logical_or(np.isnan(x1), np.isnan(x2))
        x1 = x1[~nan_idxs]
        x2 = x2[~nan_idxs]

        # Compute correlation at this delay
        if corr_type == 'spearman':
            r, _ = stats.spearmanr(x1, x2)
        elif corr_type == 'pearson':
            r, _ = stats.pearsonr(x1, x2)
        else:
            raise ValueError('invalid corr_type parameter')
        autocorr_rs[i] = r

    return autocorr_rs


# Load shank df
df_shanks_stats = pd.read_csv(config_dict['processed_path'] + 'df_shanks_stats.csv', index_col=0)
df_shanks_stats = df_shanks_stats[df_shanks_stats['burst_type'] == 0].reset_index()

# Initialize autocorr output
autocorr_feats = ['volt_amp', 'period', 'time_rdsym', 'time_ptsym']
autocorr_rs_by_shank = defaultdict(lambda: np.zeros((len(df_shanks_stats), len(config_dict['autocorr_delays']))))

# For each shank, compute autocorrs of lfp features
for i_shank, row in df_shanks_stats.iterrows():
    print(row['rat'], row['sess'], str(row['sh']))

    # Load shape dataframe
    path_shape = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + str(row['sh']) + '/cycle_by_cycle_shape.csv'
    df_shape = pd.read_csv(path_shape, index_col=0)
    df_shape = df_shape[df_shape['is_burst0'] == 1]

    for feat in autocorr_feats:
        autocorr_rs_by_shank[feat][i_shank] = compute_cycle_autocorr(df_shape, feat,
                                         autocorr_delays=config_dict['autocorr_delays'],
                                         corr_type=config_dict['autocorr_type'])

pickle.dump(dict(autocorr_rs_by_shank),
            open(config_dict['processed_path']+'cycle_autocorrelations.pkl', 'wb'))
