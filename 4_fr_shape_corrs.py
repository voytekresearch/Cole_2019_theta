"""
4_fr_shape_corrs.py
Compute correlation between firing rate and cycle features

$ time python 4_fr_shape_corrs.py

Time: ~60 min
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from config import config_dict
import glob

pd.options.mode.chained_assignment = None

# Load shank and neuron dfs
df_shanks_stats = pd.read_csv(config_dict['processed_path'] + 'df_shanks_stats.csv', index_col=0)
df_neus_stats = pd.read_csv(config_dict['processed_path'] + 'df_neus_stats.csv', index_col=0)

# Determine sessions with position data
position_files = glob.glob(config_dict['position_path'] + '*')
sess_with_position = [p[-13:-4] for p in position_files]

# Define features to correlate with FR
feat_analyze = ['volt_amp', 'period', 'time_rdsym', 'time_ptsym']

# For each shank
dict_stats = defaultdict(list)
for _, row in df_shanks_stats[df_shanks_stats['burst_type'] == 0].iterrows():
    print(row['rat'], row['sess'], str(row['sh']))

    # Load shape dataframe
    path_shape = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + str(row['sh']) + '/cycle_by_cycle_shape.csv'
    df_shape = pd.read_csv(path_shape, index_col=0)

    # Load spike times
    path_spikes = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + str(row['sh']) + '/spikes_per_cycle.json'
    df_spikes = pd.read_json(path_spikes)

    # Merge dfs
    df_shapespk = df_shape.merge(df_spikes, left_index=True, right_index=True).reset_index(drop=True)

    # Load neuron info
    df_neus_shank = df_neus_stats[(df_neus_stats['rat'] == row['rat']) &
                                  (df_neus_stats['sess'] == row['sess']) &
                                  (df_neus_stats['sh'] == row['sh'])]

    # For each burst detection condition
    for i_burst in range(len(config_dict['burst_kwargs_list'])):

        # For each neuron
        for i_neu, row_neu in df_neus_shank.iterrows():

            # Determine cycles of interest
            df_neu = df_shapespk[df_shapespk['is_burst' + str(i_burst)] == 1].reset_index(drop=True)

            # Determine FR of neuron of interest
            df_neu = df_neu.rename(columns={row_neu['clu']: 'spikes'})
            df_neu['Nspikes'] = df_neu['spikes'].str.len()
            df_neu['fr'] = df_neu['Nspikes'] / (df_neu['period'] / row['Fs'])
            if row['sess'] in (sess_with_position):
                df_bursts = df_neu.groupby('burst' + str(i_burst) + '_number')[feat_analyze + ['speed', 'fr']].mean()
            else:
                df_bursts = df_neu.groupby('burst' + str(i_burst) + '_number')[feat_analyze + ['fr']].mean()

            # Determine number of spikes in cycles of interest
            N_spikes_total = df_neu['Nspikes'].sum()

            if N_spikes_total >= config_dict['analysis_N_spikes_min']:

                # Save neuron metadata
                dict_stats['rat'].append(row['rat'])
                dict_stats['sess'].append(row['sess'])
                dict_stats['sh'].append(row['sh'])
                dict_stats['clu'].append(row_neu['clu'])
                dict_stats['burst_type'].append(i_burst)

                # Save correlation between FR and cycle features
                for feat in feat_analyze:
                    r, p = stats.spearmanr(df_neu['fr'], df_neu[feat])
                    dict_stats['corr_cycle_fr_{:s}_r'.format(feat)].append(r)
                    dict_stats['corr_cycle_fr_{:s}_p'.format(feat)].append(p)

                # Analyze burst relationships for each feature
                for feat in feat_analyze:
                    # For each burst, compute the average firing rate and average cycle feature
                    r, p = stats.spearmanr(df_bursts[feat], df_bursts['fr'])
                    dict_stats['corr_burst_across_fr_{:s}_r'.format(feat)].append(r)
                    dict_stats['corr_burst_across_fr_{:s}_p'.format(feat)].append(p)

                    # Compute correlation between cycle feature and fr within each burst
                    df_temp = df_neu[['burst' + str(i_burst) + '_number', feat, 'fr']]
                    df_temp = df_temp.groupby('burst' + str(i_burst) + '_number').corr().reset_index()
                    rs = df_temp[df_temp['level_1'] == 'fr'][feat].dropna()
                    _, p = stats.wilcoxon(rs)
                    dict_stats['corr_burst_within_fr_{:s}_avgr'.format(feat)].append(np.mean(rs))
                    dict_stats['corr_burst_within_fr_{:s}_p'.format(feat)].append(p)

                # Normalize features and run GLM for cycle and bursts
                for feat in feat_analyze:
                    df_neu[feat] = (df_neu[feat] - df_neu[feat].mean())/df_neu[feat].std(ddof=0)
                    df_bursts[feat] = (df_bursts[feat] - df_bursts[feat].mean())/df_bursts[feat].std(ddof=0)
                if row['sess'] in (sess_with_position):
                    df_neu['speed'] = (df_neu['speed'] - df_neu['speed'].mean())/df_neu['speed'].std(ddof=0)
                    df_bursts['speed'] = (df_bursts['speed'] - df_bursts['speed'].mean())/df_bursts['speed'].std(ddof=0)

                mdl = smf.glm('fr ~ ' + ' + '.join(feat_analyze), data=df_neu)
                res = mdl.fit()
                for feat in feat_analyze:
                    dict_stats['glm_cycle_fr_coef_' + feat].append(res.params[feat])
                    dict_stats['glm_cycle_fr_p_' + feat].append(res.pvalues[feat])

                # Compute and save R2
                sse = np.sum(res.resid_response**2)
                sst = np.sum((df_neu['fr'] - np.mean(df_neu['fr']))**2)
                r2 = 1 - (sse / sst)
                dict_stats['glm_cycle_fr_r2'].append(r2)

                try:
                    mdl = smf.glm('fr ~ ' + ' + '.join(feat_analyze), data=df_bursts)
                    res = mdl.fit()
                    for feat in feat_analyze:
                        dict_stats['glm_burst_fr_coef_' + feat].append(res.params[feat])
                        dict_stats['glm_burst_fr_p_' + feat].append(res.pvalues[feat])

                    sse = np.sum(res.resid_response**2)
                    sst = np.sum((df_bursts['fr'] - np.mean(df_bursts['fr']))**2)
                    r2 = 1 - (sse / sst)
                    dict_stats['glm_burst_fr_r2'].append(r2)
                except:
                    for feat in feat_analyze:
                        dict_stats['glm_burst_fr_coef_' + feat].append(np.nan)
                        dict_stats['glm_burst_fr_p_' + feat].append(np.nan)
                    dict_stats['glm_burst_fr_r2'].append(np.nan)

                # If the neuron is on a shank with position, then add a GLM accounting for speed
                if row['sess'] in (sess_with_position):
                    mdl = smf.glm('fr ~ ' + ' + '.join(feat_analyze + ['speed']), data=df_bursts)
                    res = mdl.fit()
                    for feat in feat_analyze + ['speed']:
                        dict_stats['glm_speed_fr_coef_' + feat].append(res.params[feat])
                        dict_stats['glm_speed_fr_p_' + feat].append(res.pvalues[feat])

                    # Compute and save R2
                    sse = np.sum(res.resid_response**2)
                    sst = np.sum((df_bursts['fr'] - np.mean(df_bursts['fr']))**2)
                    r2 = 1 - (sse / sst)
                    dict_stats['glm_speed_fr_r2'].append(r2)
                else:
                    for feat in feat_analyze + ['speed']:
                        dict_stats['glm_speed_fr_coef_' + feat].append(np.nan)
                        dict_stats['glm_speed_fr_p_' + feat].append(np.nan)
                    dict_stats['glm_speed_fr_r2'].append(np.nan)

df_spkshape_corrs = pd.DataFrame(dict_stats)
df_spkshape_corrs.to_csv(config_dict['processed_path'] + 'df_fr_shape_corrs.csv')
