"""
5_spiketime_shape_corrs.py
Compute effect of rdsym on individual neuron
spike timing

$ time python 5_spiketime_shape_corrs.py

Time: ~22 min
"""

from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from config import config_dict
import h5py
from bycycle.cyclepoints import extrema_interpolated_phase


def samples_to_phases(spks, phase_array):
    phases = []
    for spk in spks:
        phases.append(phase_array[spk])
    return phases


def gaussian(x, mu, sig):
    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))


def circ_conv(x, kernel):
    x2 = np.hstack([x, x, x])
    x2_conv = np.convolve(x2, kernel, 'same')
    x_conv = x2_conv[len(x):2 * len(x)]
    return x_conv


# Prep spike time bins and gaussian kernels
norm_bin_size = config_dict['norm_spike_binsize']
normtime_bins = np.arange(0, 1.01, norm_bin_size)
normtime_bin_centers = [np.mean([x, y]) for (x, y) in zip(normtime_bins[1:], normtime_bins[:-1])]

norm_gauss_std = config_dict['norm_spike_gauss_std']
norm_gauss_t = np.arange(-norm_gauss_std * 3, norm_gauss_std * 3 + norm_bin_size, norm_bin_size)
norm_gauss_curve = gaussian(norm_gauss_t, 0, norm_gauss_std)
norm_gauss_curve = norm_gauss_curve / np.sum(norm_gauss_curve)

trough_bin_size = config_dict['trough_spike_binsize']
trough_bins = np.arange(-trough_bin_size * 10,
                        trough_bin_size * 11, trough_bin_size)
trough_bin_centers = [np.mean([x, y]) for (
    x, y) in zip(trough_bins[1:], trough_bins[:-1])]

trough_gauss_std = config_dict['trough_spike_gauss_std']
trough_gauss_t = np.arange(-trough_gauss_std * 3,
                           trough_gauss_std * 3 + trough_bin_size, trough_bin_size)
trough_gauss_curve = gaussian(trough_gauss_t, 0, trough_gauss_std)
trough_gauss_curve = trough_gauss_curve / np.sum(trough_gauss_curve)

pha_bin_size = config_dict['pha_spike_binsize']
pha_bins = np.arange(-np.pi, np.pi * 1.01, pha_bin_size)
pha_bin_centers = [np.mean([x, y])
                   for (x, y) in zip(pha_bins[1:], pha_bins[:-1])]

pha_gauss_std = config_dict['pha_spike_gauss_std']
pha_gauss_t = np.arange(-pha_gauss_std * 3,
                        pha_gauss_std * 3 + pha_bin_size, pha_bin_size)
pha_gauss_curve = gaussian(pha_gauss_t, 0, pha_gauss_std)
pha_gauss_curve = pha_gauss_curve / np.sum(pha_gauss_curve)

# Load shank and neuron dfs
df_shanks_stats = pd.read_csv(
    config_dict['processed_path'] + 'df_shanks_stats.csv', index_col=0)
df_neus_stats = pd.read_csv(
    config_dict['processed_path'] + 'df_neus_stats.csv', index_col=0)

# For each shank, compute spike time histograms
dict_stats = defaultdict(list)
for _, row in df_shanks_stats[df_shanks_stats['burst_type'] == 0].iterrows():
    if config_dict['verbose']:
        print(row['rat'], row['sess'], str(row['sh']))

    # Load shape dataframe
    path_shape = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + str(row['sh']) + '/cycle_by_cycle_shape.csv'
    df_shape = pd.read_csv(path_shape, index_col=0)

    # Load spike times
    path_spikes = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + str(row['sh']) + '/spikes_per_cycle.json'
    df_spikes = pd.read_json(path_spikes)

    # Load spike times around trough
    path_spikes_trough = config_dict['mat_path'] + '/' + row['rat'] + \
        '/' + row['sess'] + '/' + \
        str(row['sh']) + '/spikes_around_extrema.json'
    df_spikes_trough = pd.read_json(path_spikes_trough)

    # Merge dfs
    df_shapespk = df_shape.merge(df_spikes, left_index=True, right_index=True).reset_index(drop=True)
    df_shapespk = df_shapespk[df_shapespk['is_burst0'] == 1].reset_index(drop=True)
    df_shapespk_trough = df_shape.merge(df_spikes_trough, left_index=True, right_index=True).reset_index(drop=True)
    df_shapespk_trough = df_shapespk_trough[df_shapespk_trough['is_burst0'] == 1].reset_index(
        drop=True)

    # Load lfp
    finame = config_dict['mat_path'] + '/' + row['rat'] + '/' + row['sess'] + '/' + str(row['sh']) + '/lfp' + str(row['elec']) + '.mat'
    f = h5py.File(finame)
    lfp = np.array(f['lfp']).T[0]

    # Compute LFP phase
    pha = extrema_interpolated_phase(lfp, np.append(df_shape['sample_last_peak'].values, df_shape['sample_next_peak'].values[-1]),
                                                    df_shape['sample_trough'],
                                                    zeroxR=df_shape['sample_zerox_rise'],
                                                    zeroxD=df_shape['sample_zerox_decay'])

    # Compute phase array for each cycle
    pha_by_cycle = np.zeros(len(df_shapespk), dtype=np.ndarray)
    for j, row_cycle in df_shapespk.iterrows():
        pha_by_cycle[j] = pha[row_cycle['sample_last_peak']:row_cycle['sample_next_peak']]
    df_shapespk['phase_array'] = pha_by_cycle

    # For each neuron
    df_neus_shank = df_neus_stats[(df_neus_stats['rat'] == row['rat']) &
                                  (df_neus_stats['sess'] == row['sess']) &
                                  (df_neus_stats['sh'] == row['sh'])]
    for i_neu, row_neu in df_neus_shank.iterrows():
        if row_neu['N_spikes_burst0'] >= config_dict['analysis_N_spikes_min']:
            # Save neuron metadata
            dict_stats['rat'].append(row['rat'])
            dict_stats['sess'].append(row['sess'])
            dict_stats['sh'].append(row['sh'])
            dict_stats['clu'].append(row_neu['clu'])

            # Get neuron activity of interest
            df_neu = df_shapespk.rename(columns={row_neu['clu']: 'spikes'})
            df_neu_trough = df_shapespk_trough.rename(
                columns={row_neu['clu']: 'spikes_trough'})
            df_neu['spikes_trough'] = df_neu_trough['spikes_trough']

            # Compute each spike's phase
            spks_phases = []
            for i_cycle, row_cycle in df_neu.iterrows():
                spks_phases.append(samples_to_phases(
                    row_cycle['spikes'], row_cycle['phase_array']))
            df_neu['spikes_phases'] = spks_phases

            # Compute each spike's normalize peak-to-peak time
            spks_norm = []
            for i_cycle, row_cycle in df_neu.iterrows():
                spks_norm.append(
                    np.array(row_cycle['spikes']) / row_cycle['period'])
            df_neu['spikes_normtime'] = spks_norm

            # Test for uniformity of distribution of normalized spike times
            all_spikes_normtime = [
                x for y in df_neu['spikes_normtime'] for x in y]
            D, p = stats.kstest(all_spikes_normtime, 'uniform')
            dict_stats['nonuniform_norm_spk_times_D'].append(D)
            dict_stats['nonuniform_norm_spk_times_p'].append(p)

            # Test for difference between normalized spike times for sym and
            # asym
            df_asym = df_neu[df_neu['time_rdsym'] <=
                             config_dict['analysis_rdsym_bottom_max']]
            df_sym = df_neu[df_neu['time_rdsym'] >
                            config_dict['analysis_rdsym_top_min']]
            all_spikes_normtime_asym = [
                x for y in df_asym['spikes_normtime'] for x in y]
            all_spikes_normtime_sym = [
                x for y in df_sym['spikes_normtime'] for x in y]
            U, p = stats.mannwhitneyu(
                all_spikes_normtime_asym, all_spikes_normtime_sym)
            dict_stats['mwu_normtime_rdsym_p'].append(p)

            # Smooth normalized spike time distribution: asymmetrical
            hist, bins = np.histogram(
                all_spikes_normtime_asym, bins=normtime_bins)
            # Convert histogram to spikes per cycle
            hist = hist / len(df_asym)
            hist_asym = np.convolve(hist, norm_gauss_curve, mode='same')
            dict_stats['norm_spktime_asym_mode'].append(
                normtime_bin_centers[np.argmax(hist_asym)])

            # Smooth normalized spike time distribution: symmetrical
            hist, bins = np.histogram(
                all_spikes_normtime_sym, bins=normtime_bins)
            hist = hist / len(df_sym)
            hist_sym = np.convolve(hist, norm_gauss_curve, mode='same')
            dict_stats['norm_spktime_sym_mode'].append(
                normtime_bin_centers[np.argmax(hist_sym)])

            # Trough-centered spiking
            # Compare trough spike times for asym and sym
            all_spikes_trough_asym = np.array(
                [x for y in df_asym['spikes_trough'] for x in y]) / row['Fs']
            all_spikes_trough_sym = np.array(
                [x for y in df_sym['spikes_trough'] for x in y]) / row['Fs']
            all_spikes_trough_asym = all_spikes_trough_asym[all_spikes_trough_asym >= -
                                                            config_dict['trough_spike_mwu_maxt']]
            all_spikes_trough_sym = all_spikes_trough_sym[all_spikes_trough_sym >= -
                                                          config_dict['trough_spike_mwu_maxt']]
            all_spikes_trough_asym = all_spikes_trough_asym[all_spikes_trough_asym <=
                                                            config_dict['trough_spike_mwu_maxt']]
            all_spikes_trough_sym = all_spikes_trough_sym[all_spikes_trough_sym <=
                                                          config_dict['trough_spike_mwu_maxt']]
            U, p = stats.mannwhitneyu(
                all_spikes_trough_asym, all_spikes_trough_sym)
            dict_stats['mwu_trough_time_rdsym_p'].append(p)

            # Smooth trough spike time distribution: asymmetrical
            hist_asym, _ = np.histogram(
                all_spikes_trough_asym, bins=trough_bins)
            hist_asym = hist_asym / \
                (len(df_asym) * config_dict['trough_spike_binsize'])
            hist_asym = np.convolve(hist_asym, trough_gauss_curve, mode='same')
            dict_stats['trough_spike_asym_mode'].append(
                trough_bin_centers[np.argmax(hist_asym)])

            # Smooth trough spike time distribution: symmetrical
            hist_sym, _ = np.histogram(all_spikes_trough_sym, bins=trough_bins)
            hist_sym = hist_sym / \
                (len(df_sym) * config_dict['trough_spike_binsize'])
            hist_sym = np.convolve(hist_sym, trough_gauss_curve, mode='same')
            dict_stats['trough_spike_sym_mode'].append(
                trough_bin_centers[np.argmax(hist_sym)])

            # Get all spike phases
            all_spikes_pha_asym = np.array(
                [x for y in df_asym['spikes_phases'] for x in y])
            all_spikes_pha_sym = np.array(
                [x for y in df_sym['spikes_phases'] for x in y])

            # Smooth normalized spike phase distribution: asymmetric
            all_pha_asym = [x for y in df_asym['phase_array'] for x in y]
            pha_bins_Nsamples, _ = np.histogram(all_pha_asym, bins=pha_bins)
            hist_asym, _ = np.histogram(all_spikes_pha_asym, bins=pha_bins)
            hist_asym = hist_asym / pha_bins_Nsamples
            hist_asym = circ_conv(hist_asym, pha_gauss_curve)
            dict_stats['pha_spike_asym_mode'].append(
                pha_bin_centers[np.argmax(hist_asym)])

            # Smooth normalized spike phase distribution: symmetric
            all_pha_sym = [x for y in df_sym['phase_array'] for x in y]
            pha_bins_Nsamples, _ = np.histogram(all_pha_sym, bins=pha_bins)
            hist_sym, _ = np.histogram(all_spikes_pha_sym, bins=pha_bins)
            hist_sym = hist_sym / pha_bins_Nsamples
            hist_sym = circ_conv(hist_sym, pha_gauss_curve)
            dict_stats['pha_spike_sym_mode'].append(
                pha_bin_centers[np.argmax(hist_sym)])

df_spktime_corrs = pd.DataFrame(dict_stats)
df_spktime_corrs.to_csv(
    config_dict['processed_path'] + 'df_spktime_shape_corrs.csv')
