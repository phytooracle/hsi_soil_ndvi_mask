#!/usr/bin/env python3
"""
Author : eg
Date   : 2021-06-07
Purpose: Rock the Casbah
"""

import argparse
import os
import sys
import h5py
from spectral import * 
import numpy as np
import glob
import warnings
import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import json
import glob
import multiprocessing
warnings.filterwarnings('ignore')


# --------------------------------------------------
def get_args():
    """Get command-line arguments"""

    parser = argparse.ArgumentParser(
        description='Rock the Casbah',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('h5_file',
                        metavar='h5_file',
                        help='Hyperspectral H5 file.')

    parser.add_argument('-b',
                        '--band',
                        help='Band to use for soil masking.',
                        metavar='band',
                        type=int,
                        default=700)

    parser.add_argument('-min',
                        '--min_x',
                        help='Minimum x value for cropping.',
                        metavar='min_x',
                        type=int,
                        default=0)   

    parser.add_argument('-max',
                        '--max_x',
                        help='Maximum x value for cropping.',
                        metavar='max_x',
                        type=int,
                        required=False)
                        # default=700)   

    return parser.parse_args()


# --------------------------------------------------
def closest(lst, K):

    return min(enumerate(lst), key=lambda x: abs(x[1]-K))


# --------------------------------------------------
def generate_soil_mask(f, x_lim, band = 700):
    
    original = f['hyperspectral'][:, x_lim[0]:x_lim[1],band]
    mask = f['hyperspectral'][:, x_lim[0]:x_lim[1], band]

    mask[mask<=3000]=0
    mask[mask>0]=255

    masked_array = f['hyperspectral'][:, x_lim[0]:x_lim[1],:]
    masked_array[np.where(mask==0)] = 0

    return mask, masked_array


# --------------------------------------------------
def generate_ndvi_mask(f, x_lim):
    
    wavelength_floats = f.attrs['wavelength'].astype(float)
    b1 = closest(wavelength_floats, 607.0)[0]
    b2 = closest(wavelength_floats, 802.5)[0]

    mask = ndvi(f['hyperspectral'][:, x_lim[0]:x_lim[1],:], b1, b2)
    mask[mask<0.4]=0
    mask[mask>=0.4]=255

    masked_array = f['hyperspectral'][:, x_lim[0]:x_lim[1],:]
    masked_array[np.where(mask==0)] = 0

    return mask, masked_array


# --------------------------------------------------
def get_mean_reflectance(masked_array):

    mean_refl_list = []
    mean_refl = np.zeros(masked_array.shape[2])

    for i in np.arange(masked_array.shape[2]):
        refl_band = masked_array[:,:,i]
        mean_refl[i] = np.ma.mean(refl_band[refl_band!=0])

    mean_refl_list.append(mean_refl)
    
    return mean_refl_list


# --------------------------------------------------
def plot_spectra(f, mean_refl_list):
    
    wavelength_floats = f.attrs['wavelength'].astype(float)

    sns.set_style("whitegrid")
    sns.set_context("paper")

    sns.relplot(x=wavelength_floats,
                y=mean_refl_list[0], 
                height=5,
                aspect=1.5)

    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Digital number (DN)')
    plt.xticks(rotation=45);


# --------------------------------------------------
def write_mask_to_h5(f, soil_mask, ndvi_mask):
    
    f.create_dataset('soil_mask', data=soil_mask)
    f.create_dataset('ndvi_mask', data=ndvi_mask)
    f.close()


# --------------------------------------------------
def generate_soil_mask(f, x_lim, band = 700):
    
    original = f['hyperspectral'][:, x_lim[0]:x_lim[1],band]
    mask = f['hyperspectral'][:, x_lim[0]:x_lim[1], band]

    mask[mask<=3000]=0
    mask[mask>0]=255

    masked_array = f['hyperspectral'][:, x_lim[0]:x_lim[1],:]
    masked_array[np.where(mask==0)] = 0

    return mask, masked_array


# --------------------------------------------------
def generate_ndvi_mask(f, x_lim):
    
    wavelength_floats = f.attrs['wavelength'].astype(float)
    b1 = closest(wavelength_floats, 607.0)[0]
    b2 = closest(wavelength_floats, 802.5)[0]

    mask = ndvi(f['hyperspectral'][:, x_lim[0]:x_lim[1],:], b1, b2)
    mask[mask<0.4]=0
    mask[mask>=0.4]=255

    masked_array = f['hyperspectral'][:, x_lim[0]:x_lim[1],:]
    masked_array[np.where(mask==0)] = 0

    return mask, masked_array


# --------------------------------------------------
def get_mean_reflectance(masked_array):

    mean_refl_list = []
    mean_refl = np.zeros(masked_array.shape[2])

    for i in np.arange(masked_array.shape[2]):
        refl_band = masked_array[:,:,i]
        mean_refl[i] = np.ma.mean(refl_band[refl_band!=0])

    mean_refl_list.append(mean_refl)
    
    return mean_refl_list


# --------------------------------------------------
def write_mask_to_h5(f, soil_mask, ndvi_mask, soil_mean_refl, ndvi_mean_refl):
    
    f.create_dataset('soil_mask', data=soil_mask)
    f.create_dataset('ndvi_mask', data=ndvi_mask)
    f.create_dataset('soil_mean_spectra', data=soil_mean_refl)
    f.create_dataset('ndvi_mean_spectra', data=ndvi_mean_refl)
    f.close()


# --------------------------------------------------
def main():
    """Make a jazz noise here"""

    args = get_args()

    # Open hyperspectral H5 file. 
    f = h5py.File(args.h5_file, 'a')

    # Determine x bounds.
    if args.max_x is None:

        n_row, n_col, n_bands = f['hyperspectral'].shape
        x_lim = (args.min_x, n_col)

    else: 
        x_lim = (args.min_x, args.max_x)

    # Selected band soil masking.
    soil_mask, soil_masked_array = generate_soil_mask(f, x_lim)
    soil_mean_refl = get_mean_reflectance(soil_masked_array)

    # NDVI soil masking.
    ndvi_mask, ndvi_masked_array = generate_ndvi_mask(f, x_lim)
    ndvi_mean_refl = get_mean_reflectance(ndvi_masked_array)

    # Append masks to H5 file.
    write_mask_to_h5(f, soil_mask, ndvi_mask, soil_mean_refl, ndvi_mean_refl)


# --------------------------------------------------
if __name__ == '__main__':
    main()
