"""
8_burst_stats.py
Compute statistics about the bursting
of the theta rhythm

$ time python 8_burst_stats.py

Time: ~2 minutes
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from scipy import stats
from config import config_dict
import glob
import statsmodels.formula.api as smf

# Load shank statistics
df_shanks_stats = pd.read_csv(config_dict['processed_path'] + 'df_shanks_stats.csv', index_col=0)

# Determine sessions with position data
position_files = glob.glob(config_dict['position_path'] + '*')
sess_with_position = [p[-13:-4] for p in position_files]

feats = ['mv_amp', 'ms_period', 'time_rdsym', 'time_ptsym']
burst_len_distributions = defaultdict(list)
burst_stats = defaultdict(list)
burst_dfs = []
for _, row in df_shanks_stats[df_shanks_stats['burst_type'] == 0].iterrows():
    print(row['rat'], row['sess'], row['sh'])

    # Also process speed if it is present
    if row['sess'] in sess_with_position:
        feats_process = feats + ['speed']
    else:
        feats_process = feats

    # Load shape dataframe
    finame = config_dict['mat_path'] + '/' + row['rat'] + '/' + row['sess'] + '/' + str(row['sh']) + '/cycle_by_cycle_shape.csv'
    df_shape = pd.read_csv(finame, index_col=0)
    df_shape['mv_amp'] = df_shape['volt_amp'] / 1000
    df_shape['ms_period'] = df_shape['period'] / row['Fs'] * 1000

    for i_burst in range(len(config_dict['burst_kwargs_list'])):

        # Initialize saving
        burst_stats['rat'].append(row['rat'])
        burst_stats['sess'].append(row['sess'])
        burst_stats['sh'].append(row['sh'])
        burst_stats['burst_type'].append(i_burst)

        # Limit shape dataframe to bursting cycles
        df_cycles = df_shape[df_shape['is_burst' + str(i_burst)] == 1]
        df_cycles = df_cycles[feats_process + ['burst' + str(i_burst) + '_len',
                                               'burst' + str(i_burst) + '_number',
                                               'sample_next_peak', 'sample_last_peak']]

        # Compute distribution of burst lengths
        df_bursts = df_cycles.groupby('burst' + str(i_burst) + '_number').mean()
        burst_len_distributions[i_burst].append(df_bursts['burst' + str(i_burst) + '_len'].values)

        # Create columns for values of first cycle feature in burst
        df_temp = df_cycles.groupby('burst' + str(i_burst) + '_number').first().drop('burst' + str(i_burst) + '_len', axis=1)
        df_temp.rename({k: k + '_first' for k in df_temp.columns}, axis=1, inplace=True)
        df_bursts = df_bursts.merge(df_temp, left_index=True, right_index=True)

        # Zscore cycle features
        for feat in feats_process:
            df_bursts['z_' + feat] = (df_bursts[feat] - df_bursts[feat].mean()) / df_bursts[feat].std(ddof=0)

        # Burst timing
        df_bursts['burst_start'] = df_cycles.groupby('burst' + str(i_burst) + '_number')['sample_last_peak'].first()
        df_bursts['burst_end'] = df_cycles.groupby('burst' + str(i_burst) + '_number')['sample_next_peak'].last()
        df_bursts['time_since_last_burst'] = np.insert(df_bursts['burst_start'].values[1:] - df_bursts['burst_end'].values[:-1], 0, 0)

        # Get feature of previous burst
        for feat in feats_process:
            df_bursts['last_' + feat] = np.insert(df_bursts[feat].values[:-1], 0, 0)
        df_bursts.drop(0, axis=0, inplace=True)
        burst_dfs.append(df_bursts)

        # Save number of bursts
        burst_stats['N_bursts'].append(len(df_bursts))

        # Compute correlation between cycle features of adjacent bursts
        # Force bursts to be apart from each other by some minimum time (results same if 0 or 1 seconds)
        df_bursts = df_bursts[df_bursts['time_since_last_burst'] > row['Fs']]
        for feat in feats_process:
            r, p = stats.spearmanr(df_bursts['last_' + feat], df_bursts[feat])
            burst_stats[feat + '_adjburst_r'].append(r)
            burst_stats[feat + '_adjburst_p'].append(p)

        if 'speed' not in feats_process:
            burst_stats['speed_adjburst_r'].append(np.nan)
            burst_stats['speed_adjburst_p'].append(np.nan)

        # Zscore first cycle features
        for feat in feats_process:
            df_bursts['z_' + feat + '_first'] = (df_bursts[feat + '_first'] - df_bursts[feat + '_first'].mean()) / df_bursts[feat + '_first'].std(ddof=0)

        # Train linear model to predict burst length from first cycle features
        try:
            results = smf.glm('burst' + str(i_burst) + '_len ~ ' + ' + '.join(['z_' + s + '_first' for s in feats]),
                              data=df_bursts).fit()
            for feat in feats:
                burst_stats[feat + '_first_coef'].append(results.params['z_' + feat + '_first'])
                burst_stats[feat + '_first_p'].append(results.pvalues['z_' + feat + '_first'])

            # Compute R2
            sse = np.sum(results.resid_response**2)
            sst = np.sum((df_bursts['burst' + str(i_burst) + '_len'] - np.mean(df_bursts['burst' + str(i_burst) + '_len']))**2)
            r2 = 1 - (sse / sst)
            burst_stats['glm_first_r2'].append(r2)
        except:
            for feat in feats:
                burst_stats[feat+'_first_coef'].append(np.nan)
                burst_stats[feat+'_first_p'].append(np.nan)
            burst_stats['glm_first_r2'].append(np.nan)


        # If position data is available, repeat GLM with speed
        if row['sess'] in sess_with_position:
            try:
                # Run GLM
                results = smf.glm('burst' + str(i_burst) + '_len ~ ' + ' + '.join(['z_'+ s + '_first' for s in feats_process]),
                              data=df_bursts).fit()
                for feat in feats_process:
                    burst_stats[feat+'_first_speed_coef'].append(results.params['z_'+feat+'_first'])
                    burst_stats[feat+'_first_speed_p'].append(results.pvalues['z_'+feat+'_first'])

                # Compute R2
                sse = np.sum(results.resid_response**2)
                sst = np.sum((df_bursts['burst' + str(i_burst) + '_len'] - np.mean(df_bursts['burst' + str(i_burst) + '_len']))**2)
                r2 = 1 - (sse / sst)
                burst_stats['glm_first_speed_r2'].append(r2)
            except:
                for feat in feats + ['speed']:
                    burst_stats[feat+'_first_speed_coef'].append(np.nan)
                    burst_stats[feat+'_first_speed_p'].append(np.nan)
                burst_stats['glm_first_speed_r2'].append(np.nan)
        else:
            for feat in feats + ['speed']:
                burst_stats[feat+'_first_speed_coef'].append(np.nan)
                burst_stats[feat+'_first_speed_p'].append(np.nan)
            burst_stats['glm_first_speed_r2'].append(np.nan)

# Save burst stats
df_burst_stats = pd.DataFrame(burst_stats)
df_burst_stats.to_csv(config_dict['processed_path'] + 'df_burst_stats.csv')

# Save burst distributions
for i_burst in range(len(config_dict['burst_kwargs_list'])):
    np.save(config_dict['processed_path'] + 'burst' + str(i_burst) + '_lengths.npy', np.hstack(burst_len_distributions[i_burst]))
