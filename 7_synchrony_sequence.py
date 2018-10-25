"""
7_synchrony_sequence.py
Quantify the synchrony and sequence of pairs of neurons
and relate this to the LFP cycle parameters

$ time python 7_synchrony_sequence.py

Time: 1 hour
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from config import config_dict
import itertools
from bisect import bisect_left


def compute_sequence_col(last_spike, next_spike):
    if last_spike and not next_spike:
        return 'post'
    elif next_spike and not last_spike:
        return 'pre'
    else:
        return np.nan


# Load stats for each shank and each neuron
df_shanks_stats = pd.read_csv(config_dict['processed_path'] + 'df_shanks_stats.csv', index_col=0)
df_neus_stats = pd.read_csv(config_dict['processed_path'] + 'df_neus_stats.csv', index_col=0)

# Define time scales and features of interest
synchrony_ms = config_dict['synchrony_ms']
cycle_features = ['volt_amp', 'period', 'time_rdsym', 'time_ptsym']

# For each shank compute synchrony and sequence correlations
dict_stats = defaultdict(list)
for _, row in df_shanks_stats[df_shanks_stats['burst_type'] == 0].iterrows():

    # Load shape dataframe
    path_shape = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + str(row['sh']) + '/cycle_by_cycle_shape.csv'
    df_shape = pd.read_csv(path_shape, index_col=0)

    # Load spiking
    f = np.load(config_dict['mat_path'] + row['rat'] + '/' + row['sess'] + '/neu_raster.npz')
    raster_times = f['raster_times_by_shank'][row['sh']]
    raster_neus = f['raster_neus_by_shank'][row['sh']]

    # Load neuron info
    df_neus_shank = df_neus_stats[(df_neus_stats['rat'] == row['rat']) &
                                  (df_neus_stats['sess'] == row['sess']) &
                                  (df_neus_stats['sh'] == row['sh'])]

    # Only consider neurons with a minimum number of spikes
    df_neus_shank = df_neus_shank[df_neus_shank['N_spikes_recording'] >= config_dict['analysis_N_spikes_min']]
    shank_clus = list(df_neus_shank['clu'])

    # Update user on status
    if config_dict['verbose']:
        print(row['rat'], row['sess'], row['sh'], len(df_neus_shank), 'neus')

    # For each pair of neurons
    for pair in itertools.combinations(shank_clus, 2):
        # Make spkt2 be the one with fewer spikes
        spkt1 = raster_times[raster_neus == pair[0]]
        spkt2 = raster_times[raster_neus == pair[1]]
        if len(spkt2) > len(spkt1):
            spkt1, spkt2 = spkt2, spkt1

        # For each spike in spkt2, find the next closest and most recent spike in spkt1
        spkt1 = np.insert(np.append(spkt1, np.inf), 0, -np.inf)
        next_spikes = np.array([spkt1[bisect_left(spkt1, x)] for x in spkt2])
        last_spikes = np.array([spkt1[bisect_left(spkt1, x)-1] for x in spkt2])

        # Make df of closest spikes
        df_sync = pd.DataFrame({'spike_time': spkt2 / row['Fs'],
                                'last_spike': last_spikes / row['Fs'],
                                'next_spike': next_spikes / row['Fs'],
                                'last_spike_dt': (spkt2 - last_spikes) / row['Fs'],
                                'next_spike_dt': (next_spikes - spkt2) / row['Fs']})
        df_sync = df_sync.replace([np.inf, -np.inf], np.nan).dropna()
        df_sync['closest_spike'] = df_sync[['last_spike_dt', 'next_spike_dt']].min(axis=1)

        # Get cycle information for each spike
        df_sync['cycle_number'] = [bisect_left(df_shape['sample_next_peak'], x) for x in df_sync['spike_time']*row['Fs']]
        df_sync = df_sync.merge(df_shape[cycle_features], left_on='cycle_number', right_index=True)
        df_sync['period'] = df_sync['period'] / row['Fs']

        # For each time scale of synchrony and sequence
        for ms in synchrony_ms:
            # Determine if closest/next/last spike is within X ms
            df_sync['spike_within_{:d}ms'.format(ms)] = df_sync['closest_spike'] < (ms / 1000)
            df_sync['last_spike_within_{:d}ms'.format(ms)] = df_sync['last_spike_dt'] <= (ms / 1000)
            df_sync['next_spike_within_{:d}ms'.format(ms)] = df_sync['next_spike_dt'] <= (ms / 1000)

            # Only analyze synchrony/sequence if the number of synchronous spikes is above some threshold
            N_sync_spikes = df_sync['spike_within_{:d}ms'.format(ms)].sum()
            if (N_sync_spikes >= config_dict['sync_min_spikes']) and (len(df_sync) > 0):

                # Do computation for every burst setting
                for i_burst in range(len(config_dict['burst_kwargs_list'])):

                    # Save metadata
                    dict_stats['rat'].append(row['rat'])
                    dict_stats['sess'].append(row['sess'])
                    dict_stats['sh'].append(row['sh'])
                    dict_stats['synchrony_ms'].append(ms)
                    dict_stats['burst_type'].append(i_burst)
                    dict_stats['N_sync_spikes_recording'].append(N_sync_spikes)

                    if len(spkt2) > len(spkt1):
                        dict_stats['clu1'].append(pair[1])
                        dict_stats['clu2'].append(pair[0])
                    else:
                        dict_stats['clu1'].append(pair[0])
                        dict_stats['clu2'].append(pair[1])

                    # Keep only cycles of interest
                    if 'is_burst' + str(i_burst) in df_sync.columns:
                        df_sync.drop(columns='is_burst' + str(i_burst), inplace=True)
                    df_sync = df_sync.merge(df_shape[['is_burst' + str(i_burst)]],
                                            left_on='cycle_number', right_index=True)

                    # Test if difference in is_burst when neurons are synchronous vs asynchronous
                    x1 = df_sync[df_sync['spike_within_{:d}ms'.format(ms)]]['is_burst' + str(i_burst)]
                    x2 = df_sync[~df_sync['spike_within_{:d}ms'.format(ms)]]['is_burst' + str(i_burst)]
                    oddsratio, pval = stats.fisher_exact([[x1.sum(), len(x1) - x1.sum()],
                                                          [x2.sum(), len(x2) - x2.sum()]])
                    dict_stats['is_burst_sync_p'].append(pval)
                    dict_stats['is_burst_sync_mean'].append(np.mean(x1))
                    dict_stats['is_burst_nonsync_mean'].append(np.mean(x2))

                    dict_stats['N_is_burst_sync'].append(x1.sum())
                    dict_stats['N_is_nonburst_sync'].append(len(x1) - x1.sum())
                    dict_stats['N_is_burst_nonsync'].append(x2.sum())
                    dict_stats['N_is_nonburst_nonsync'].append(len(x2) - x2.sum())

                    # When testing synchrony for cycle features, count each cycle once,
                    # and remove cycle if spikes have different results
                    df_cycles = df_sync[df_sync['is_burst' + str(i_burst)]]
                    dict_stats['N_sync_spikes_bursting'].append(df_cycles['spike_within_{:d}ms'.format(ms)].sum())
                    df_cycle_temp = df_cycles[['cycle_number', 'spike_within_{:d}ms'.format(ms)]].groupby('cycle_number').mean().reset_index()
                    cycles_use = df_cycle_temp[df_cycle_temp['spike_within_{:d}ms'.format(ms)].isin([0, 1])]['cycle_number'].values
                    df_temp = df_cycles[df_cycles['cycle_number'].isin(cycles_use)]
                    df_temp = df_temp.drop_duplicates(subset=['cycle_number'])

                    # Test if difference in cycle features when neurons are synchronous vs asynchronous
                    for feat in cycle_features:
                        x1 = df_temp[df_temp['spike_within_{:d}ms'.format(ms)] == 1][feat]
                        x2 = df_temp[df_temp['spike_within_{:d}ms'.format(ms)] == 0][feat]
                        U, p = stats.mannwhitneyu(x1, x2)
                        dict_stats['{:s}_sync_p'.format(feat)].append(p)
                        dict_stats['{:s}_sync_mean'.format(feat)].append(np.mean(x1))
                        dict_stats['{:s}_nonsync_mean'.format(feat)].append(np.mean(x2))

                    # Determine the sequence of spiking at each spike
                    df_seq = df_sync.copy()
                    df_seq['sequence_{:d}ms'.format(ms)] = df_seq.apply(
                            lambda row: compute_sequence_col(row['last_spike_within_{:d}ms'.format(ms)],
                                                             row['next_spike_within_{:d}ms'.format(ms)]), axis=1)

                    # Do not do sequence analysis if all sequences are null
                    if df_seq['sequence_{:d}ms'.format(ms)].isnull().sum() == len(df_seq):
                        dict_stats['N_spikes_sequence'].append(N_spikes_seq)
                        dict_stats['is_burst_seqprepost_p'].append(np.nan)
                        dict_stats['is_burst_post_mean'].append(np.nan)
                        dict_stats['is_burst_pre_mean'].append(np.nan)
                        dict_stats['N_spikes_cycle_sequence'].append(np.nan)
                        dict_stats['N_is_burst_post'].append(np.nan)
                        dict_stats['N_is_nonburst_post'].append(np.nan)
                        dict_stats['N_is_burst_pre'].append(np.nan)
                        dict_stats['N_is_nonburst_pre'].append(np.nan)
                        dict_stats['preoverpost_frac_theta'].append(np.nan)
                        dict_stats['preoverpost_frac_nottheta'].append(np.nan)
                        for feat in cycle_features:
                            dict_stats['{:s}_seqprepost_p'.format(feat)].append(np.nan)
                            dict_stats['{:s}_seqpre_mean'.format(feat)].append(np.nan)
                            dict_stats['{:s}_seqpost_mean'.format(feat)].append(np.nan)
                    else:

                        # Remove some spikes for sequence analysis
                        # e.g. don't want a burst of spkt2 followed by a spkt1 to be counted as a bunch of instances of a post-sequence
                        # Determine which spikes occur within the window size of one another
                        df_seq['time_to_last_spike'] = df_seq['spike_time'].diff()
                        df_seq['last_spike_close'] = df_seq['time_to_last_spike'] <= (2 * ms / 1000)
                        df_seq['last_spike_previous'] = np.insert(df_seq['last_spike'].values[:-1], 0, np.nan)
                        df_seq['next_spike_previous'] = np.insert(df_seq['next_spike'].values[:-1], 0, np.nan)

                        # For each pair of spikes that occurs within the window size of another spike
                        # Remove both spikes if their pre/post designations are opposite
                        df_seq['last_spike_seq'] = np.insert(df_seq['sequence_{:d}ms'.format(ms)].values[:-1], 0, np.nan)
                        df_seq['last_spike_seq'] = np.insert(df_seq['sequence_{:d}ms'.format(ms)].values[:-1], 0, np.nan)
                        df_seq['remove_this_spike'] = (df_seq['last_spike_seq'] != df_seq['sequence_{:d}ms'.format(ms)]) & \
                            ~df_seq['last_spike_seq'].isnull() & ~df_seq['sequence_{:d}ms'.format(ms)].isnull() & df_seq['last_spike_close']
                        df_seq['remove_last_spike'] = df_seq['remove_this_spike'].copy()

                        # If pre/post designations are the same, remove the farther spike if both have the same closest_spike
                        df_both_pre = df_seq[(df_seq['sequence_{:d}ms'.format(ms)] == 'pre') &
                                             (df_seq['last_spike_seq'] == 'pre') &
                                             (df_seq['last_spike_close'] == True)] # NOTE NECESSARY PEP8 VIOLATION. Otherwise changes function
                        df_both_post = df_seq[(df_seq['sequence_{:d}ms'.format(ms)] == 'post') &
                                              (df_seq['last_spike_seq'] == 'post') &
                                              (df_seq['last_spike_close'] == True)] # NOTE NECESSARY PEP8 VIOLATION. Otherwise changes function

                        # For spikes which the current and previous were 'pre',
                        # mark last_spike for removal if last_spike==last_spike_previous
                        spk_index_remove_last_spk = df_both_post[df_both_post['last_spike'] == df_both_post['last_spike_previous']].index
                        df_seq.loc[spk_index_remove_last_spk, 'remove_last_spike'] = True

                        # For spikes which the current and previous were 'post',
                        # mark this_spike for removal if next_spike==next_spike_previous
                        spk_index_remove_this_spk = df_both_pre[df_both_pre['next_spike'] == df_both_pre['next_spike_previous']].index
                        df_seq.loc[spk_index_remove_this_spk, 'remove_this_spike'] = True

                        # Remove rows of df_seq as appropriate
                        rows_rmv1 = np.array(df_seq[df_seq['remove_this_spike']].index.tolist())
                        rows_rmv2 = np.array(df_seq[df_seq['remove_last_spike']].index.tolist()) - 1
                        rows_rmv = np.union1d(rows_rmv1, rows_rmv2)
                        df_seq = df_seq.drop(rows_rmv)

                        # Only do sequence analysis if have a minimum number of instances of sequence
                        N_spikes_seq = df_seq['sequence_{:d}ms'.format(ms)].value_counts().sum()
                        dict_stats['N_spikes_sequence'].append(N_spikes_seq)
                        if N_spikes_seq < config_dict['seq_min_spikes']:
                            dict_stats['is_burst_seqprepost_p'].append(np.nan)
                            dict_stats['is_burst_post_mean'].append(np.nan)
                            dict_stats['is_burst_pre_mean'].append(np.nan)
                            dict_stats['N_is_burst_post'].append(np.nan)
                            dict_stats['N_is_nonburst_post'].append(np.nan)
                            dict_stats['N_is_burst_pre'].append(np.nan)
                            dict_stats['N_is_nonburst_pre'].append(np.nan)
                            dict_stats['N_spikes_cycle_sequence'].append(np.nan)
                            dict_stats['preoverpost_frac_theta'].append(np.nan)
                            dict_stats['preoverpost_frac_nottheta'].append(np.nan)
                            for feat in cycle_features:
                                dict_stats['{:s}_seqprepost_p'.format(feat)].append(np.nan)
                                dict_stats['{:s}_seqpre_mean'.format(feat)].append(np.nan)
                                dict_stats['{:s}_seqpost_mean'.format(feat)].append(np.nan)

                        else:
                            # Test if sequence correlated to is_burst
                            x_post = df_seq[df_seq['sequence_{:d}ms'.format(ms)] == 'post']['is_burst' + str(i_burst)]
                            x_pre = df_seq[df_seq['sequence_{:d}ms'.format(ms)] == 'pre']['is_burst' + str(i_burst)]
                            oddsratio, pval = stats.fisher_exact([[x_pre.sum(), len(x_pre) - x_pre.sum()],
                                                                  [x_post.sum(), len(x_post) - x_post.sum()]])
                            dict_stats['is_burst_seqprepost_p'].append(pval)

                            # Save means for each class of sequence
                            dict_stats['is_burst_post_mean'].append(np.mean(x_post))
                            dict_stats['is_burst_pre_mean'].append(np.mean(x_pre))

                            dict_stats['N_is_burst_post'].append(x_post.sum())
                            dict_stats['N_is_nonburst_post'].append(len(x_post) - x_post.sum())
                            dict_stats['N_is_burst_pre'].append(x_pre.sum())
                            dict_stats['N_is_nonburst_pre'].append(len(x_pre) - x_pre.sum())

                            # Save fraction of pre over post for during theta and not during theta
                            N_pre_theta = x_pre.sum()
                            N_post_theta = x_post.sum()
                            N_pre_nottheta = len(x_pre) - N_pre_theta
                            N_post_nottheta = len(x_post) - N_post_theta
                            dict_stats['preoverpost_frac_theta'].append(N_pre_theta / N_post_theta)
                            dict_stats['preoverpost_frac_nottheta'].append(N_pre_nottheta / N_post_nottheta)

                            # Only analyze cycle sequences if enough unique sequences
                            df_cycles = df_seq[df_seq['is_burst' + str(i_burst)] == 1]
                            N_spikes_cycle_seq = df_cycles['sequence_{:d}ms'.format(ms)].value_counts().sum()
                            dict_stats['N_spikes_cycle_sequence'].append(N_spikes_cycle_seq)
                            if N_spikes_cycle_seq < config_dict['seq_min_spikes']:
                                for feat in cycle_features:
                                    dict_stats['{:s}_seqprepost_p'.format(feat)].append(np.nan)
                                    dict_stats['{:s}_seqpre_mean'.format(feat)].append(np.nan)
                                    dict_stats['{:s}_seqpost_mean'.format(feat)].append(np.nan)
                            else:
                                # Loop through other cycle parameters and compare sequences
                                for feat in cycle_features:
                                    x_pre = df_cycles[df_cycles['sequence_{:d}ms'.format(ms)] == 'pre'][feat]
                                    x_post = df_cycles[df_cycles['sequence_{:d}ms'.format(ms)] == 'post'][feat]
                                    U, p = stats.mannwhitneyu(x_pre, x_post)
                                    dict_stats['{:s}_seqprepost_p'.format(feat)].append(p)
                                    dict_stats['{:s}_seqpre_mean'.format(feat)].append(np.mean(x_pre))
                                    dict_stats['{:s}_seqpost_mean'.format(feat)].append(np.mean(x_post))

for k in dict_stats.keys():
    print(k, len(dict_stats[k]))

df_stats = pd.DataFrame(dict_stats)
df_stats.to_csv(config_dict['processed_path'] + 'df_neuron_pairs.csv')
