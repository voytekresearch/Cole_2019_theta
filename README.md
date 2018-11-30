# Cole_2018_theta
Analysis code to create the figures in ([SR Cole & B Voytek. 2018. Hippocampal theta bursting and waveform shape reflect CA1 spiking patterns. bioRxiv](https://www.biorxiv.org/content/early/2018/10/25/452987)).

Follow the steps below in order to replicate or investigate the analyses of this study.

## Activate environment

The precise python environment used to run this analysis is provided in the `environment.yml` file. In order to run the scripts and notebooks, it is a good idea to activate the environment by running the following command in the terminal:

```
$ conda env create -f environment.yml
```

## Raw data from CRCNS

The data for this study can be obtained from the [hc3 database on CRCNS](https://crcns.org/data-sets/hc/hc-3). The 27 sessions analyzed are listed in [the MATLAB script used to convert the files containing LFPs (.eeg) to MATLAB files](https://github.com/voytekresearch/Cole_2018_theta/blob/v2/binary_to_mat/reformat_lfps_to_mat.m). The files needed are those with the extensions `.eeg` (field potentials), `.clu` (spiking), `.res` (spiking), `.xml` (metadata), and `.whl` (rat position tracking). This is the first script that should be run after downloading the raw data

## Data processing scripts

The numbered (1-8) python scripts compute the waveform shape and spiking features from the raw data, and quantify how they are related. They should be run in order and currently yield the files in the `processed_data` directory.

## Analysis configuration

The settings used for running the analysis are aggregated in the `config.py` file. If you want to re-run the analyses with different parameter selections (e.g. different burst detection hyperparameters), change the parameters in this file.

## Processed data

Because the scripts take several hours to run on a laptop, and the raw data will need to be downloaded from crcns.org, intermediate data is saved in the `processed_data` directory, so that interested researchers can more easily replicate and tune the analyses performed in the notebooks.

## Notebooks

Each notebook corresponds to 1 figure in the paper, as numbered. The code heavily utilizes libraries in [bycycle](https://github.com/voytekresearch/bycycle), a toolbox for cycle-by-cycle analysis of brain rhythms, and [neurodsp](https://github.com/voytekresearch/neurodsp), a toolbox for analyzing neural oscillations.
