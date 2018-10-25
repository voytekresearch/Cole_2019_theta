"""
1_Reformat_spiketrain_files.py
Step 1 in the data processing workflow for this project.
First, reformat spiketrain rasters into 1 npz file per session
Second, Create 1 csv file per session containing info
about each cell from hc3-cell.csv

$ time python 1_Reformat_spiketrain_files.py
Runtime: ~15 min
"""

import numpy as np
import pandas as pd
import glob
import shutil

from config import config_dict
from config import Fs_by_rat

# Remove shank 8 from all ec013 files
dirs_rmv = glob.glob(config_dict['mat_path'] + 'ec013/*/8/')
for dirpath in dirs_rmv:
    shutil.rmtree(dirpath)

# Determine LFP file names
lfp_files = glob.glob(config_dict['mat_path'] + '*/*')
N_sessions = len(lfp_files)

# Determine rat and sess names
rat = [f.split('/')[-2] for f in lfp_files]
sess = [f.split('/')[-1] for f in lfp_files]

# Locate each clu and res file (spike trains)
res_files = [glob.glob(config_dict['rawdata_path'] + rat[i] + '*/' + sess[i] + '/*.res.*') for i in range(N_sessions)]
clu_files = [glob.glob(config_dict['rawdata_path'] + rat[i] + '*/' + sess[i] + '/*.clu.*') for i in range(N_sessions)]
N_shanks_per_sess = [len(res_files[i]) for i in range(N_sessions)]

# Load spike trains, downsample and resave in numpy format
for i in range(N_sessions):
    if config_dict['verbose']:
        print('rat', rat[i], 'session', sess[i])

    # Downsample 16x if the original Fs=20kHz; 26x if the original Fs=32552kHz
    Fs = Fs_by_rat[rat[i]]
    if Fs == 1250:
        spike_downsample_factor = 16
    elif Fs == 1252:
        spike_downsample_factor = 26
    else:
        raise ValueError('Unfamiliar Fs')

    # Downsample spike times for each shank
    raster_times_by_shank = np.zeros(N_shanks_per_sess[i], dtype=np.ndarray)
    raster_neus_by_shank = np.zeros(N_shanks_per_sess[i], dtype=np.ndarray)
    N_clusters_by_shank = np.zeros(N_shanks_per_sess[i], dtype=int)
    for j in range(N_shanks_per_sess[i]):
        shank_zeroidx = int(res_files[i][j].split('.')[-1]) - 1
        if config_dict['verbose']:
            print(j, shank_zeroidx)

        spike_times = np.loadtxt(res_files[i][j], dtype=int)
        spike_times = spike_times / spike_downsample_factor
        raw_clusters = np.loadtxt(clu_files[i][j], dtype=int)

        raster_times_by_shank[shank_zeroidx] = spike_times.astype(int)
        raster_neus_by_shank[shank_zeroidx] = raw_clusters[1:]
        N_clusters_by_shank[shank_zeroidx] = raw_clusters[0]

    # Save neuron info
    np.savez(config_dict['mat_path'] + rat[i] + '/' + sess[i] + '/neu_raster.npz',
             raster_times_by_shank=raster_times_by_shank,
             raster_neus_by_shank=raster_neus_by_shank,
             N_clusters_by_shank=N_clusters_by_shank)

# Determine day of recording for each session
dff = pd.read_csv(config_dict['metadata_path'] + 'sess_behavior.csv')
day_by_sess = {s: dff[dff['session'] == s]['topdir'].values[0] for s in sess}

# Determine neuron info for each session and save in a csv
df = pd.read_csv(config_dict['metadata_path'] + 'hc3-cell_src.csv')
dfs_by_session = [np.nan] * N_sessions
for i in range(N_sessions):
    dfs_by_session[i] = df[df['sess'] == day_by_sess[sess[i]]].reset_index(drop=True)
    dfs_by_session[i]['sh'] = dfs_by_session[i]['sh'] - 1
    dfs_by_session[i].to_csv(config_dict['mat_path'] + rat[i] + '/' + sess[i] + '/neu_info.csv')
