"""
util.py
Misc functions for loading, summarizing, and analyzing data
"""

import numpy as np
import glob
import pandas as pd
import os
from collections import defaultdict
from config import config_dict

from scipy.stats import pearsonr
from scipy.stats import chi2


def circ_corrcc(alpha, x):
    """Correlation coefficient between one circular and one linear random
    variable.

    # Circular-linear correlation code from:
    # https://etiennecmb.github.io/brainpipe/_modules/brainpipe/stat/circstat.html

    Args:
        alpha: vector
            Sample of angles in radians

        x: vector
            Sample of linear random variable

    Returns:
        rho: float
            Correlation coefficient

        pval: float
            p-value

    Code taken from the Circular Statistics Toolbox for Matlab
    By Philipp Berens, 2009
    Python adaptation by Etienne Combrisson
    """
    if len(alpha) != len(x):
        raise ValueError('The length of alpha and x must be the same')
    n = len(alpha)

    # Compute correlation coefficent for sin and cos independently
    rxs = pearsonr(x, np.sin(alpha))[0]
    rxc = pearsonr(x, np.cos(alpha))[0]
    rcs = pearsonr(np.sin(alpha), np.cos(alpha))[0]

    # Compute angular-linear correlation (equ. 27.47)
    rho = np.sqrt((rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2))

    # Compute pvalue
    pval = 1 - chi2.cdf(n * rho**2, 2)

    return rho, pval


def create_shank_df(drop_no_neu=True):
    """Output all sessions and shanks"""

    # Determine all rat names
    rat_paths = glob.glob(config_dict['mat_path'] + '*')
    rat_names = [p.split('/')[-1] for p in rat_paths if os.path.isdir(p)]

    # Determine sess and shank info for each rat
    dict_shank = defaultdict(list)
    for rat in rat_names:
        # Determine all session names
        sess_paths = glob.glob(config_dict['mat_path'] + rat + '/*')
        sess_names = [p.split('/')[-1] for p in sess_paths]
        sess_names = np.unique(sess_names)
        for sess in sess_names:
            # Determine all shanks
            sh_paths = glob.glob(config_dict['mat_path'] + rat + '/' + sess + '/*[0-9]')
            sh_names = [p.split('/')[-1] for p in sh_paths]
            for sh in sh_names:
                # Save each shank to a row
                dict_shank['rat'].append(rat)
                dict_shank['sess'].append(sess)
                dict_shank['sh'].append(sh)

                # Determine and save sampling rate
                if ('gor' in rat) or ('pin' in rat) or ('vvp' in rat):
                    dict_shank['Fs'].append(1252)
                else:
                    dict_shank['Fs'].append(1250)

                # Determine electrode number to use for LFP
                if rat == 'ec016' and sess == 'ec016.397' and sh == '2':
                    dict_shank['elec'].append(1)
                elif rat == 'ec016' and sess == 'ec016.397' and sh == '6':
                    dict_shank['elec'].append(1)
                elif rat == 'ec013' and sh == '2':
                    dict_shank['elec'].append(3)
                else:
                    dict_shank['elec'].append(0)

                # Determine brain region
                df_info = pd.read_csv(config_dict['mat_path'] + rat + '/' + sess + '/neu_info.csv')
                region = _determine_region_one_shank(rat, sess, sh, df_info)
                dict_shank['region'].append(region)

                # Determine if each shank has neurons
                if int(sh) in df_info['sh'].values:
                    dict_shank['has_neurons'].append(True)
                else:
                    dict_shank['has_neurons'].append(False)
    df_shank = pd.DataFrame.from_dict(dict_shank)

    df_shank = df_shank[df_shank['region'] == config_dict['lfp_region']]
    df_shank = df_shank.drop('region', 1)

    if drop_no_neu:
        df_shank = df_shank[df_shank['has_neurons']]
        df_shank = df_shank.drop('has_neurons', 1)
    df_shank.reset_index(inplace=True, drop=True)

    return df_shank


def _determine_region_one_shank(rat, session, shank, df_info):
    """Determine the region that a rat/session/shank is in"""

    shank = str(shank)
    try:
        temp_region = df_info[df_info['sh'] == int(
            shank)].region.values[0]
    except IndexError:
        if rat == 'gor01':
            if shank == '10':
                temp_region = 'CA1'
            elif shank == '4' or shank == '5' or shank == '6':
                temp_region = 'CA3'
            else:
                raise ValueError('New region found with no label. Check region for rat:',
                                 rat, ' session:', session, ' shank:', shank)
        elif rat == 'pin01':
            if shank == '0' or shank == '1':
                temp_region = 'CA1'
            else:
                raise ValueError('New region found with no label. Check region for rat:',
                                 rat, ' session:', session, ' shank:', shank)
        elif rat == 'vvp01':
            if session == '2006-4-10_21-2-40':
                if shank == '0':
                    temp_region = 'CA1'
                elif shank == '7':
                    temp_region = 'CA3'
                else:
                    raise ValueError('New region found with no label. Check region for rat:',
                                     rat, ' session:', session, ' shank:', shank)
            else:
                raise ValueError('New region found with no label. Check region for rat:',
                                 rat, ' session:', session, ' shank:', shank)
        elif rat == 'ec013':
            if session == 'ec013.426':
                if shank == '2':
                    temp_region = 'EC5'
                else:
                    raise ValueError('New region found with no label. Check region for rat:',
                                     rat, ' session:', session, ' shank:', shank)
            else:
                raise ValueError('New region found with no label. Check region for rat:',
                                 rat, ' session:', session, ' shank:', shank)
        elif rat == 'ec014':
            if session == 'ec014.277':
                if shank == '5' or shank == '6' or shank == '7':
                    temp_region = 'CA1'
                else:
                    raise ValueError('New region found with no label. Check region for rat:',
                                     rat, ' session:', session, ' shank:', shank)
            elif session == 'ec014.440':
                if shank == '12':
                    temp_region = 'X'
                elif shank == '6':
                    temp_region = 'CA1'
                elif shank == '9':
                    temp_region = 'EC3'
                else:
                    raise ValueError('New region found with no label. Check region for rat:',
                                     rat, ' session:', session, ' shank:', shank)
            else:
                raise ValueError('New region found with no label. Check region for rat:',
                                 rat, ' session:', session, ' shank:', shank)
        elif rat == 'ec016':
            if session == 'ec016.499':
                if shank == '0' or shank == '9':
                    temp_region = 'CA1'
                else:
                    raise ValueError('New region found with no label. Check region for rat:',
                                     rat, ' session:', session, ' shank:', shank)
            elif session == 'ec016.582':
                if shank == '0':
                    temp_region = 'CA1'
                elif shank == '4':
                    temp_region = 'EC3'
                else:
                    raise ValueError('New region found with no label. Check region for rat:',
                                     rat, ' session:', session, ' shank:', shank)
            else:
                raise ValueError('New region found with no label. Check region for rat:',
                                 rat, ' session:', session, ' shank:', shank)
        elif rat == 'f01' or rat == 'g01':
            temp_region = 'CA1'
        elif rat == 'i01':
            if shank == '6' or shank == '11' or shank == '12':
                temp_region = 'CA1'
            elif shank == '0' or shank == '1':
                temp_region = 'EC?'
            else:
                raise ValueError('New region found with no label. Check region for rat:',
                                 rat, ' session:', session, ' shank:', shank)
        else:
            raise ValueError('New region found with no label. Check region for rat:',
                             rat, ' session:', session, ' shank:', shank)

    return temp_region
