"""
3_Neu_shank_stats.py
Compute statistics of spike-field coupling and firing rate
for each neuron in the dataset.

$ time python 3_Neu_shank_stats.py

Time: ~1 hour
"""

from collections import defaultdict
import numpy as np
import h5py
import pandas as pd
import neurodsp
from bycycle.cyclepoints import extrema_interpolated_phase
import itertools
from scipy import stats
from config import config_dict
import util
from shutil import copyfile
import os

pd.options.mode.chained_assignment = None


def compute_2pi_pha(pha1):
    if pha1 > 0:
        return pha1
    else:
        return 2 * np.pi + pha1


# Create directory for processed output
if not os.path.exists(config_dict['processed_path']):
    os.makedirs(config_dict['processed_path'])

# Throw error if configured for peak-centered cycles
if config_dict['extrema_center'] != 'T':
    raise ValueError('Script only written for trough-centered cycles.')

# Load and process shank dataframe
df_shanks = util.create_shank_df()

# For each shank, compute features of LFP
feats_corr = ['volt_amp', 'period', 'time_rdsym', 'time_ptsym']
shank_feats = defaultdict(list)
for _, row in df_shanks.iterrows():
    # Load shape dataframe
    finame = config_dict['mat_path'] + '/' + row['rat'] + '/' + row['sess'] + '/' + row['sh'] + '/cycle_by_cycle_shape.csv'
    df_shape = pd.read_csv(finame, index_col=0)

    # Do computations for each burst detection parameter set
    for i_burst in range(len(config_dict['burst_kwargs_list'])):
        # Initiatlize row
        shank_feats['rat'].append(row['rat'])
        shank_feats['sess'].append(row['sess'])
        shank_feats['sh'].append(row['sh'])
        shank_feats['burst_type'].append(i_burst)

        # Save average cycle features
        df_shape_cycles = df_shape[df_shape['is_burst' + str(i_burst)] == 1]
        shank_feats['amp_mean'].append(df_shape_cycles['volt_amp'].mean())
        shank_feats['period_mean'].append(df_shape_cycles['period'].mean())
        shank_feats['rdsym_mean'].append(df_shape_cycles['time_rdsym'].mean())
        shank_feats['ptsym_mean'].append(df_shape_cycles['time_ptsym'].mean())

        # Save other signal features
        shank_feats['cycling_frac'].append(df_shape_cycles['period'].sum() / df_shape['period'].sum())
        shank_feats['N_cycles_all'].append(len(df_shape))
        shank_feats['N_cycles_burst'].append(len(df_shape_cycles))
        shank_feats['N_seconds'].append((df_shape['sample_next_peak'].values[-1] - df_shape['sample_last_peak'].values[0]) / row['Fs'])

        # Compute correlations between amp, period, and rdsym features
        for feat1, feat2 in itertools.combinations(feats_corr, 2):
            r, p = stats.spearmanr(df_shape_cycles[feat1], df_shape_cycles[feat2])
            shank_feats['corr_' + feat1 + '_' + feat2 + '_r'].append(r)
            shank_feats['corr_' + feat1 + '_' + feat2 + '_p'].append(p)

# Merge shank statistics into df of shanks
df_shanks_stats = df_shanks.merge(pd.DataFrame(shank_feats), on=['rat', 'sess', 'sh'])

# Compute period in milliseconds
df_shanks_stats['mv_amp_mean'] = df_shanks_stats['amp_mean'] / 1000
df_shanks_stats['ms_period_mean'] = df_shanks_stats['period_mean'] / df_shanks_stats['Fs'] * 1000

# Save shank info to csv
df_shanks_stats.to_csv(config_dict['processed_path'] + 'df_shanks_stats.csv')

print('Saved shank stats!')

# For each shank: compute phase over time and SFC for each neuron
dict_stats = defaultdict(list)
N_bins = config_dict['N_bins_sfc']
pha_bins = np.linspace(-np.pi, np.pi, N_bins + 1)
for _, row in df_shanks_stats[df_shanks_stats['burst_type'] == 0].iterrows():
    if config_dict['verbose']:
        print(row['rat'], row['sess'], str(row['sh']))

    # For some recordings, zeropadding is necessary for hilbert transform
    if (row['rat'] == 'gor01' and row['sess'] == '2006-6-13_15-44-7') or \
       (row['rat'] == 'vvp01' and row['sess'] == '2006-4-10_21-2-40') or \
       (row['rat'] == 'vvp01' and row['sess'] == '2006-4-18_21-22-11'):
        hilbert_increase_N = True
    else:
        hilbert_increase_N = False

    # Load lfp
    finame = config_dict['mat_path'] + '/' + row['rat'] + '/' + row['sess'] + '/' + row['sh'] + '/lfp' + str(row['elec']) + '.mat'
    f = h5py.File(finame)
    lfp = np.array(f['lfp']).T[0]

    # Load spike times
    f = np.load(config_dict['mat_path'] + row['rat'] + '/' + row['sess'] + '/neu_raster.npz')
    if len(f['raster_times_by_shank']) <= int(row['sh']):
        raise ValueError('The neuron data does not extend to this shank.')
    raster_times = f['raster_times_by_shank'][int(row['sh'])]
    raster_neus = f['raster_neus_by_shank'][int(row['sh'])]

    # Load shape dataframe
    path_shape = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + row['sh'] + '/cycle_by_cycle_shape.csv'
    df_shape = pd.read_csv(path_shape, index_col=0)

    # Load N spikes
    path_Nspikes = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + row['sh'] + '/Nspikes_per_cycle.csv'
    df_Nspikes = pd.read_csv(path_Nspikes, index_col=0)

    # Load spike times
    path_spikes = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + row['sh'] + '/spikes_per_cycle.json'
    df_spikes = pd.read_json(path_spikes)

    # Create combined dfs for shape and spiking info
    df_shapespk = df_shape.merge(df_Nspikes, left_index=True, right_index=True).reset_index(drop=True)
    df_shapespk_cycles = df_shapespk[df_shapespk['is_burst0'] == 1].reset_index(drop=True)

    # Create combined df for spike times
    df_shapespkt = df_shape.merge(df_spikes, left_index=True, right_index=True).reset_index(drop=True)
    df_shapespkt = df_shapespkt[df_shapespkt['is_burst0'] == 1].reset_index(drop=True)

    # Delete spike times before the first cycle's first peak and after the last cycle's rise zerox
    spks_keep = np.logical_and(raster_times >= df_shape['sample_last_peak'][0],
                               raster_times <= df_shape['sample_zerox_rise'].values[-1])
    raster_times = raster_times[spks_keep]
    raster_neus = raster_neus[spks_keep]

    # Compute waveform and hilbert phase for the lfp
    pha = extrema_interpolated_phase(lfp, np.append(df_shape['sample_last_peak'].values,
                                                    df_shape['sample_next_peak'].values[-1]),
                                     df_shape['sample_trough'],
                                     zeroxR=df_shape['sample_zerox_rise'],
                                     zeroxD=df_shape['sample_zerox_decay'])
    pha_hilbert = neurodsp.timefrequency.phase_by_time(lfp, row['Fs'],
                                                       config_dict['f_range'],
                                                       filter_kwargs={'compute_transition_band': False,
                                                                      'N_cycles': 3},
                                                       hilbert_increase_N=hilbert_increase_N)

    # Compute phase array for each cycle
    pha_by_cycle = np.zeros(len(df_shapespk_cycles), dtype=np.ndarray)
    pha_by_cycle_hilbert = np.zeros(len(df_shapespk_cycles), dtype=np.ndarray)
    for j, row_cycle in df_shapespk_cycles.iterrows():
        pha_by_cycle[j] = pha[row_cycle['sample_last_peak']:row_cycle['sample_next_peak']]
        pha_by_cycle_hilbert[j] = pha_hilbert[row_cycle['sample_last_peak']:row_cycle['sample_next_peak']]
    all_phases_cycle = np.hstack(pha_by_cycle)
    all_phases_cycle = all_phases_cycle[~np.isnan(all_phases_cycle)]
    all_phases_cycle_hilbert = np.hstack(pha_by_cycle_hilbert)
    all_phases_cycle_hilbert = all_phases_cycle_hilbert[~np.isnan(all_phases_cycle_hilbert)]

    # Load neuron info
    fn = config_dict['mat_path'] + '/' + row['rat'] + '/' + row['sess'] + '/neu_info.csv'
    df_neu_info_temp = pd.read_csv(fn, index_col=0)
    df_neu_info_temp = df_neu_info_temp[df_neu_info_temp['sh'] == int(row['sh'])]
    df_neu_info_temp = df_neu_info_temp[df_neu_info_temp['clu'].isin(np.unique(raster_neus))]

    # For each neuron
    for i_neu, row_neu in df_neu_info_temp.iterrows():
        # Get spike train
        spkT = raster_times[raster_neus == row_neu['clu']]

        # Initialize row
        dict_stats['rat'].append(row['rat'])
        dict_stats['sess'].append(row['sess'])
        dict_stats['sh'].append(row['sh'])
        dict_stats['clu'].append(row_neu['clu'])
        dict_stats['fr1'].append(row_neu['fr1'])
        dict_stats['fr2'].append(row_neu['fr2'])
        dict_stats['type'].append(row_neu['type'])

        # Save spiking stats for recording
        dict_stats['fr_recording'].append(len(spkT) / row['N_seconds'])
        dict_stats['N_spikes_recording'].append(len(spkT))

        # Rename current neuron to Nspikes
        df_neu = df_shapespk.rename(columns={str(row_neu['clu']): 'Nspikes'})

        # For each burst detection method, compute FR for theta vs nontheta
        for i_burst in range(len(config_dict['burst_kwargs_list'])):
            # Get dfs for cycles and noncycles
            df_isburst = df_neu[df_neu['is_burst' + str(i_burst)]]
            df_notburst = df_neu[~df_neu['is_burst' + str(i_burst)]]

            dict_stats['fr_burst' + str(i_burst)].append(df_isburst['Nspikes'].sum() / df_isburst['period'].sum() * row['Fs'])
            dict_stats['fr_notburst' + str(i_burst)].append(df_notburst['Nspikes'].sum() / df_notburst['period'].sum() * row['Fs'])
            dict_stats['N_spikes_burst' + str(i_burst)].append(df_isburst['Nspikes'].sum())

            # Compare firing rate for cycles and noncycles
            df_isburst['fr'] = df_isburst['Nspikes'] / (df_isburst['period'] / row['Fs'])
            df_notburst['fr'] = df_notburst['Nspikes'] / (df_notburst['period'] / row['Fs'])
            U, p = stats.mannwhitneyu(df_isburst['fr'], df_notburst['fr'])
            dict_stats['mwu_fr_burst' + str(i_burst) + '_notburst_p'].append(p)

        # No further analysis if too few spikes
        if dict_stats['N_spikes_burst0'][-1] <= config_dict['N_spikes_min_sfc']:
            dict_stats['sfc_magnitude_recording'].append(np.nan)
            dict_stats['sfc_phase_recording'].append(np.nan)
            dict_stats['sfc_magnitude_cycles'].append(np.nan)
            dict_stats['sfc_phase_cycles'].append(np.nan)
            dict_stats['sfc_magnitude_cycles_hilbert'].append(np.nan)
            dict_stats['sfc_phase_cycles_hilbert'].append(np.nan)

        else:
            # Compute phase of each spike
            spk_phases = pha[spkT]

            # Compute firing rate in each phase bin
            pha_bins_Nsamples, pha_bin_edges = np.histogram(pha[~np.isnan(pha)], bins=pha_bins)
            pha_bins_Nspikes, _ = np.histogram(spk_phases, bins=pha_bins)
            pha_bins_fr = pha_bins_Nspikes / (pha_bins_Nsamples / row['Fs'])

            # Compute SFC
            pha_bin_centers = [np.mean([x, y]) for x, y in zip(pha_bin_edges[:-1], pha_bin_edges[1:])]
            mean_vector = np.sum([fr * np.exp(1j * pha) for fr, pha in zip(pha_bins_fr, pha_bin_centers)]) / np.sum(pha_bins_fr)

            dict_stats['sfc_magnitude_recording'].append(np.abs(mean_vector))
            dict_stats['sfc_phase_recording'].append(np.angle(mean_vector))

            # Compute sfc for only oscillating cycles - waveform phase
            spk_phases = []
            for cycle_phases, spk_idxs in zip(pha_by_cycle, df_shapespkt[row_neu['clu']]):
                spk_phases.append(cycle_phases[spk_idxs])
            all_spk_phases_cycle = [x for xy in spk_phases for x in xy]
            pha_bins_Nsamples, _ = np.histogram(all_phases_cycle, bins=pha_bins)
            pha_bins_Nspikes, _ = np.histogram(all_spk_phases_cycle, bins=pha_bins)
            pha_bins_fr = pha_bins_Nspikes / (pha_bins_Nsamples / row['Fs'])
            pha_bin_centers = [np.mean([x, y]) for x, y in zip(pha_bin_edges[:-1], pha_bin_edges[1:])]
            mean_vector = np.sum([fr * np.exp(1j * pha) for fr, pha in zip(pha_bins_fr, pha_bin_centers)]) / np.sum(pha_bins_fr)
            dict_stats['sfc_magnitude_cycles'].append(np.abs(mean_vector))
            dict_stats['sfc_phase_cycles'].append(np.angle(mean_vector))

            # Compute sfc for only oscillating cycles - hilbert phase
            spk_phases_hilbert = []
            for cycle_phases_hilbert, spk_idxs in zip(pha_by_cycle_hilbert, df_shapespkt[row_neu['clu']]):
                spk_phases_hilbert.append(cycle_phases_hilbert[spk_idxs])
            all_spk_phases_cycle_hilbert = [x for xy in spk_phases_hilbert for x in xy]
            pha_bins_Nsamples, _ = np.histogram(all_phases_cycle_hilbert, bins=pha_bins)
            pha_bins_Nspikes, _ = np.histogram(all_spk_phases_cycle_hilbert, bins=pha_bins)
            pha_bins_fr = pha_bins_Nspikes / (pha_bins_Nsamples / row['Fs'])
            pha_bin_centers = [np.mean([x, y]) for x, y in zip(pha_bin_edges[:-1], pha_bin_edges[1:])]
            mean_vector = np.sum([fr * np.exp(1j * pha) for fr, pha in zip(pha_bins_fr, pha_bin_centers)]) / np.sum(pha_bins_fr)
            dict_stats['sfc_magnitude_cycles_hilbert'].append(np.abs(mean_vector))
            dict_stats['sfc_phase_cycles_hilbert'].append(np.angle(mean_vector))

# Convert neu stats dict to df
df_neus_stats = pd.DataFrame(dict_stats)

# Add some columns to neuron dataframe
for i_burst in range(len(config_dict['burst_kwargs_list'])):
    df_neus_stats['norm_fr_diff_burst' + str(i_burst)] = (df_neus_stats['fr_burst' + str(i_burst)] -
                                                            df_neus_stats['fr_notburst' + str(i_burst)]) / \
                                                         (df_neus_stats['fr_burst' + str(i_burst)] +
                                                            df_neus_stats['fr_notburst' + str(i_burst)])

df_neus_stats['sfc_phase_cycles_2pi'] = df_neus_stats['sfc_phase_cycles'].apply(compute_2pi_pha)

# Save neu dataframe
df_neus_stats.to_csv(config_dict['processed_path'] + 'df_neus_stats.csv')

# Save configuration file
copyfile('config.py', config_dict['processed_path'] + 'config.py')
