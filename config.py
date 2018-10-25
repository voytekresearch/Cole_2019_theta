"""
config.py
Configuration settings (parameters) for data processing.
"""

import numpy as np

config_dict = {
     # Data loading
     'mat_path': '/gh/data2/hc3/matv3/',
     'rawdata_path': '/gh/data/hc3/',
     'metadata_path': '/gh/data2/hc3/docs/',
     'position_path': '/gh/data2/hc3/behavior/',

     # Metadata
     'behavior_Fs': 39.0625,
     'lfp_region': 'CA1',

     # Data saving
     'processed_path': 'processed_data/',
     'verbose': True,

     # Filtering
     'f_range': (4, 10),
     'cf_low': 25,
     'cf_high': 1,
     'N_seconds_low': .5,
     'N_seconds_high': 2,

     # Burst detection
     'extrema_center': 'T',
     'burst_kwargs_list': [
          {'amplitude_fraction_threshold': 0,
           'amplitude_consistency_threshold': .4,
           'period_consistency_threshold': .55,
           'monotonicity_threshold': .8,
           'N_cycles_min': 3},
          {'amplitude_fraction_threshold': 0,
           'amplitude_consistency_threshold': .6,
           'period_consistency_threshold': .6,
           'monotonicity_threshold': .9,
           'N_cycles_min': 3},
          {'amplitude_fraction_threshold': 0,
           'amplitude_consistency_threshold': .35,
           'period_consistency_threshold': .5,
           'monotonicity_threshold': .75,
           'N_cycles_min': 3},
          {'amplitude_fraction_threshold': 0,
           'amplitude_consistency_threshold': .65,
           'period_consistency_threshold': .65,
           'monotonicity_threshold': .95,
           'N_cycles_min': 3}
          ],

     # Spiking processing
     'spikelock_window_time': .2,
     'N_bins_sfc': 50,
     'N_spikes_min_sfc': 100,
     'analysis_N_spikes_min': 100,
     'analysis_N_sync_min': 100,
     'analysis_N_spikes_min_percat': 25,
     'analysis_synchrony_ms': 20,
     'isi_max': 625,
     'sync_min_spikes': 20,
     'seq_min_spikes': 20,
     'synchrony_ms': np.arange(10, 51, 10),

     # Binarize asymmetric and symmetric
     'analysis_rdsym_bottom_max': .4,
     'analysis_rdsym_top_min': .4,
     'analysis_ptsym_bottom_max': .4,
     'analysis_ptsym_top_min': .4,

     # Smoothing spike times
     'norm_spike_binsize': .05,
     'norm_spike_gauss_std': .05,
     'trough_spike_binsize': .01,
     'trough_spike_mwu_maxt': .1,
     'trough_spike_gauss_std': .005,
     'pha_spike_binsize': np.pi / 10,
     'pha_spike_gauss_std': np.pi / 10,

     # Autocorrelation
     'autocorr_delays': np.arange(1, 101),
     'autocorr_delays_neus_max': 500,
     'autocorr_type': 'spearman'
     }

# Define sampling freq for each session
Fs_by_rat = {'ec013': 1250, 'ec014': 1250, 'ec016': 1250,
             'f01': 1250, 'g01': 1250, 'i01': 1250,
             'gor01': 1252, 'pin01': 1252, 'vvp01': 1252}
