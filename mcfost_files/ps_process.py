import numpy as np
import xarray as xr
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse


# parameter space processing

# assumes you run this in the correct directory ... 

run_list = [s for s in os.listdir() if ('param_' in s and os.path.isdir(s))]
pickle_list = [pd.read_pickle(os.path.join(s, 'img_output.pkl')).set_index('wl') for s in run_list]
param_list = pd.read_csv('param_space.csv', header=None, index_col=0)[1].values

# import data values
sed_wavelength, sed_data = np.load('/data/jhyl3/windy-dust/mcfost_files/sed_data.npy')
wavelength_dlc, dlt_data, sett10_model = np.load('/data/jhyl3/windy-dust/mcfost_files/dlc_data.npy')
br_data = np.load('/data/jhyl3/windy-dust/mcfost_files/br_data.npy')[1]

sed_data_list = np.interp(wavelength_dlc, sed_wavelength, sed_data) # interpolate for comparison
data_df = pd.DataFrame({'wl': wavelength_dlc, 'sed': sed_data_list, 'br_ratio': br_data, 'dlt': dlt_data}).set_index('wl')


# generate xarray dataset
ds = xr.concat([s.to_xarray() for s in pickle_list], dim='param')
ds = ds.assign_coords(param=param_list)


# generate plot

parser = argparse.ArgumentParser(description="wavelength/data selection")
parser.add_argument("wl", type=float, choices=[0.8, 2.0, 4.4, 7.7, 12.8, 21.0])
parser.add_argument("data", type=str, choices=['sed', 'br_ratio', 'dlt', 'grad'])

args = parser.parse_args()
wvl = args.wl; data_choice = args.data

if data_choice != 'grad':
    
    rcParams['figure.figsize'] = (4,2)
    plt.scatter([s*1e4 for s in param_list], ds.sel(wl=wvl)[data_choice], marker='x', s=10)
    plt.xlabel(r'Minimum stalling grain size / $\mathrm{\mu m}$')

    if data_choice == 'sed': 
        plt.yscale('log')
        plt.ylabel(r'$\lambda\,F_{\lambda}$ / $\mathrm{W\,m^{-2}}$')
        plt.title(f'SED flux at {wvl} um', fontsize=10)
        plt.axhline(data_df.loc[wvl]['sed'], color='C3', linestyle='--', lw=0.8)
        file_name = f'sed_{wvl}um.png'

    elif data_choice == 'br_ratio':
        plt.ylabel(r'Asymmetry factor')
        plt.title(f'Asymmetry factor at {wvl} um', fontsize=10)
        plt.axhline(data_df.loc[wvl]['br_ratio'], color='C3', linestyle='--', lw=0.8)
        file_name = f'br_ratio_{wvl}um.png'

    elif data_choice == 'dlt':
        plt.ylabel(r'$d_\mathrm{neb}$ / arcsec')
        plt.title(f'Dark lane thickness at {wvl} um', fontsize=10)
        plt.axhline(data_df.loc[wvl]['dlt'], color='C3', linestyle='--', lw=0.8)
        file_name = f'dlt_{wvl}um.png'

    plt.savefig(file_name, dpi=300, bbox_inches='tight')


# gradient - still work in progress
# specifically, slope of dark lane thickness chromaticity between 0.8 and 4.4 um