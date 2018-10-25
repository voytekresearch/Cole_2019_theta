"""
2_Compute_cyclebycycle_features.py
Step 2 in the data processing workflow for this project.
First, this script takes raw LFP files for each
rat/session/shank, and computes waveform shape features of
the hippocampal theta oscillation on a cycle-by-cycle basis.
It saves these features in a .csv file in the same folder
that the LFP file was found.
Second, this script goes through each neuron
and determines the spike times for each cycle.

$ time python 2_Compute_cyclebycycle_features.py

Time: ~2 hours
"""

# Load modules
import numpy as np
import h5py
import os
import pandas as pd
import time
from bycycle import burst
from bycycle.features import compute_features
from operator import itemgetter
from itertools import groupby
import neurodsp
import glob

import util

from config import config_dict

# Recompute is_burst with new osc kwargs
def recompute_is_burst(df_orig,
                       amplitude_fraction_threshold=0,
                       amplitude_consistency_threshold=0,
                       period_consistency_threshold=0,
                       monotonicity_threshold=0,
                       N_cycles_min=3):
    """Re-determine oscillating periods with a new choice of burst detection parameters"""
    df = df_orig.copy()
    cycle_good_amp = df['amp_fraction'] > amplitude_fraction_threshold
    cycle_good_amp_consist = df['amp_consistency'] > amplitude_consistency_threshold
    cycle_good_period_consist = df['period_consistency'] > period_consistency_threshold
    cycle_good_monotonicity = df['monotonicity'] > monotonicity_threshold
    is_burst = cycle_good_amp & cycle_good_amp_consist & cycle_good_period_consist & cycle_good_monotonicity
    is_burst[0] = False
    is_burst[-1] = False
    df['is_burst'] = is_burst
    df = burst._min_consecutive_cycles(df, N_cycles_min=N_cycles_min)
    return df['is_burst'].astype(bool)


def compute_shape(lfp, Fs, f_range, cf_low, cf_high, N_seconds_low, N_seconds_high,
                  hilbert_increase_N=False):
    """Compute shape features on a cycle-by-cycle basis"""

    # Make lfp values floats if not already
    lfp = lfp.astype(float)

    # Lowpass and highpass filter LFP
    lfp = neurodsp.filter(lfp, Fs, 'lowpass', cf_low, compute_transition_band=False,
                          N_seconds=N_seconds_low, remove_edge_artifacts=False)
    lfp = neurodsp.filter(lfp, Fs, 'highpass', cf_high, compute_transition_band=False,
                          N_seconds=N_seconds_high, remove_edge_artifacts=False)

    # Calculate shape features
    df = compute_features(lfp, Fs, f_range,
                          center_extrema=config_dict['extrema_center'],
                          hilbert_increase_N=hilbert_increase_N,
                          burst_detection_kwargs=config_dict['burst_kwargs_list'][0])

    return lfp, df


def add_position_features(lfp, df_shape, sess):
    """Add speed and position to cycle-by-cycle df"""
    # Define sampling rates
    Fs_lfp = 1252
    Fs_pos = 39.0625
    t_lfp = np.arange(0, len(lfp) / Fs_lfp, 1 / Fs_lfp)

    # Load and process position
    pos_mat = np.loadtxt('/gh/data2/hc3/behavior/{:s}.whl'.format(sess))
    x_nose_39, y_nose_39, x_body_39, y_body_39 = pos_mat.T
    x_body_39[x_body_39 == -1] = np.nan
    y_body_39[y_body_39 == -1] = np.nan
    t_pos = np.arange(0, len(x_body_39) / Fs_pos, 1 / Fs_pos)

    # Interpolate position
    x_body = np.interp(t_lfp, t_pos, x_body_39)
    y_body = np.interp(t_lfp, t_pos, y_body_39)

    # Compute average x and y coordinates for each cycle
    N_cycles = len(df_shape)
    x_by_cycle = np.zeros(N_cycles)
    y_by_cycle = np.zeros(N_cycles)
    for i_cyc, row_cyc in df_shape.iterrows():
        samp_start = int(row_cyc['sample_trough'] + row_cyc['sample_last_peak'])
        samp_end = int(row_cyc['sample_trough'] + row_cyc['sample_next_peak'])
        x_by_cycle[i_cyc] = np.nanmean(x_body[samp_start:samp_end])
        y_by_cycle[i_cyc] = np.nanmean(y_body[samp_start:samp_end])

    df_shape['x_pos'] = x_by_cycle
    df_shape['y_pos'] = y_by_cycle

    # Compute speed of rat for every cycle
    speed_by_cycle = np.zeros(len(df_shape))
    for i_cyc, row_cyc in df_shape.iterrows():
        dx = x_body[row_cyc['sample_next_peak']] - x_body[row_cyc['sample_last_peak']]
        dy = y_body[row_cyc['sample_next_peak']] - y_body[row_cyc['sample_last_peak']]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        speed_by_cycle[i_cyc] = dist / row_cyc['period'] * Fs_lfp
    df_shape['speed'] = speed_by_cycle
    return df_shape


# Determine all rats, shanks, and sessions
df_shanks = util.create_shank_df(drop_no_neu=False)

# Determine sess with position info
position_files = glob.glob(config_dict['position_path'] + '*')
sess_with_position = [p[-13:-4] for p in position_files]

# For each shank, compute cycle-by-cycle features
t_start = time.time()
for _, row in df_shanks.iterrows():
    # Define metadata
    rat, sess, sh, Fs, elec = row[['rat', 'sess', 'sh', 'Fs', 'elec']]
    spikelock_window_samps = int(np.ceil(config_dict['spikelock_window_time'] * Fs))

    # Define paths to files to save
    base_path = config_dict['mat_path'] + '/' + rat + '/' + sess + '/' + sh + '/'
    shape_path = base_path + 'cycle_by_cycle_shape.csv'
    spikes_around_extrema_path = base_path + 'spikes_around_extrema.json'
    spikes_per_cycle_path = base_path + 'spikes_per_cycle.json'
    Nspikes_per_cycle_path = base_path + 'Nspikes_per_cycle.csv'

    # Compute cycle by cycle features if not already computed
    if os.path.isfile(shape_path):
        already_computed_shape = True
        if config_dict['verbose']:
            print('Already computed shape features:', rat, sess, sh)
    else:
        already_computed_shape = False
        # Load LFP
        lfp_filename = base_path + 'lfp' + str(elec) + '.mat'
        f = h5py.File(lfp_filename)
        lfp = np.array(f['lfp']).T[0]

        # Update user on script status
        if config_dict['verbose']:
            print('Start computing shape features:', rat, sess, sh, 'time = {:d}'.format(int(time.time() - t_start)))

        try:
            # For some recordings, zeropadding is necessary for hilbert transform
            if (rat == 'gor01' and sess == '2006-6-13_15-44-7') or \
               (rat == 'vvp01' and sess == '2006-4-10_21-2-40') or \
               (rat == 'vvp01' and sess == '2006-4-18_21-22-11'):
                hilbert_increase_N = True
            else:
                hilbert_increase_N = False

            # Compute shape features
            lfp, df = compute_shape(lfp, Fs, config_dict['f_range'],
                                    config_dict['cf_low'], config_dict['cf_high'],
                                    config_dict['N_seconds_low'], config_dict['N_seconds_high'],
                                    hilbert_increase_N=hilbert_increase_N)
            df.rename({'is_burst': 'is_burst0'}, axis=1, inplace=True)

            # Recompute burst for other parameters
            for i_bk, burst_kwargs in enumerate(config_dict['burst_kwargs_list'][1:]):
                df['is_burst' + str(i_bk + 1)] = recompute_is_burst(df, **burst_kwargs)

            # Compute burst durations and numbers
            for i_burst in range(len(config_dict['burst_kwargs_list'])):
                df_temp = df[df['is_burst' + str(i_burst)] == 1]
                groups = []
                for k, g in groupby(enumerate(df_temp.index), lambda x: x[0] - x[1]):
                    groups.append(list(map(itemgetter(1), g)))
                burst_lens = [len(x) for x in groups]
                burst_lens_cycle = [burst_lens[burst_i] for burst_i in range(len(groups)) for cycle_i in groups[burst_i]]
                burst_nums_cycle = [burst_i for burst_i in range(len(groups)) for cycle_i in groups[burst_i]]
                df_temp['burst' + str(i_burst) + '_number'] = burst_nums_cycle
                df_temp['burst' + str(i_burst) + '_len'] = burst_lens_cycle


                df = df.merge(df_temp[['sample_trough', 'burst' + str(i_burst) + '_number', 'burst' + str(i_burst) + '_len']],
                              on='sample_trough', how='left')

            # Add position features if present
            if row['sess'] in sess_with_position:
                df = add_position_features(lfp, df, row['sess'])

            # Save shape features
            df.to_csv(shape_path)
            if config_dict['verbose']:
                print('saved shape features, time = {:d}'.format(int(time.time() - t_start)))

        except Exception as e:
            print('Error computing cycle-by-cycle features for', rat, sess, sh, e)

    # Compute spike features if not already computed
    if os.path.isfile(spikes_around_extrema_path) and os.path.isfile(spikes_per_cycle_path) and os.path.isfile(Nspikes_per_cycle_path):
        if config_dict['verbose']:
            print('Already computed spike times:', rat, sess, sh)
    else:
        try:
            # Load LFP
            if not already_computed_shape:
                lfp_filename = base_path + 'lfp' + str(elec) + '.mat'
                f = h5py.File(lfp_filename)
                lfp = np.array(f['lfp']).T[0]

            # Load spiking data
            f = np.load(config_dict['mat_path'] + rat + '/' + sess + '/neu_raster.npz')
            if len(f['raster_times_by_shank']) <= int(sh):
                raise ValueError('The neuron data does not extend to this shank.')
            raster_times = f['raster_times_by_shank'][int(sh)]
            raster_neus = f['raster_neus_by_shank'][int(sh)]

            # Make sure shank region in neu_info matches the prescribed region
            df_neu_info = pd.read_csv(config_dict['mat_path'] + rat + '/' + sess + '/neu_info.csv', index_col=0)
            df_neu_info = df_neu_info[df_neu_info['sh'] == int(sh)]
            if len(df_neu_info) == 0:
                raise ValueError('No neurons on this shank. Skip to next shank.')

            if config_dict['lfp_region'] != df_neu_info['region'].unique()[0]:
                raise ValueError('Spike region does not match desired region. Bug in shank index. ',
                                 config_dict['lfp_region'], df_neu_info['region'].unique()[0])

            # Determine the extrema timepoints to which to lock spikes
            if config_dict['extrema_center'] == 'T':
                df_T = pd.read_csv(shape_path)
                x_lock = df_T['sample_trough'].values
                x_lock_last_extrema = df_T['sample_last_peak'].values
                x_period = df_T['period'].values
            elif config_dict['extrema_center'] == 'P':
                print('Computing spikes on cycles centered on peaks. Not tested. Use with caution.')
                df_P = pd.read_csv(shape_path)
                x_lock = df_P['sample_peak'].values
                x_lock_last_extrema = df_P['sample_last_trough'].values
                x_period = df_P['period'].values
            else:
                raise ValueError('extrema_center option in config.py is invalid.')

            # For each neuron, compute spike times relative to each cycle's center extrema
            N_cycles = len(x_lock)
            cluster_idxs = np.unique(raster_neus)
            spikes_around_extrema = {}
            for clu_i in cluster_idxs:
                # Determine spikes for this neuron
                spkT = raster_times[raster_neus == clu_i]

                # Determine spike sample relative to each trough
                spikes_around_extrema[clu_i] = np.zeros(N_cycles, dtype=np.ndarray)
                for i, x in enumerate(x_lock):
                    spikes_around_extrema[clu_i][i] = spkT[np.logical_and(spkT >= x - spikelock_window_samps,
                                                                          spkT <= x + spikelock_window_samps)] - x

            # Compute spike samples relative to last extrema
            spikes_per_cycle = {}
            for clu_i in cluster_idxs:
                # Resave spike times to relative to last extrema
                spk_temp = spikes_around_extrema[clu_i] + (x_lock - x_lock_last_extrema)

                # Limit spike times to those in the cycle
                spk_temp = [x[x >= 0] for x in spk_temp]
                spikes_per_cycle[clu_i] = [x[x < y] for (x, y) in zip(spk_temp, x_period)]

            # Compute number of spikes in each cycle
            Nspikes_per_cycle = {}
            for clu_i in cluster_idxs:
                Nspikes_per_cycle[clu_i] = [len(x) for x in spikes_per_cycle[clu_i]]

            # Save dataframes: 1 for each feature
            df_spikes_around_extrema = pd.DataFrame(spikes_around_extrema)
            df_spikes_per_cycle = pd.DataFrame(spikes_per_cycle)
            df_Nspikes_per_cycle = pd.DataFrame(Nspikes_per_cycle)
            df_spikes_around_extrema.to_json(spikes_around_extrema_path)
            df_spikes_per_cycle.to_json(spikes_per_cycle_path)
            df_Nspikes_per_cycle.to_csv(Nspikes_per_cycle_path)

        except Exception as e:
            print('Error computing spiking features for', rat, sess, sh, e)
