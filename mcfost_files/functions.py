from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve_fft
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm, ticker
from matplotlib.colors import LogNorm
import matplotlib.colors as mcolors
import pymcfost as mcfost
import os
import scipy.constants as sc
from scipy.optimize import bisect, newton, curve_fit, fsolve
from scipy.signal import savgol_filter, find_peaks

# convolve the image with a PSF kernel
def psf_conv(img, psf_FWHM, i):
    image = img.image[0,0,i]
    sigma = psf_FWHM / (img.pixelscale * 2 * np.sqrt(2*np.log(2)))
    psf = Gaussian2DKernel(sigma, x_size=int(15*sigma), y_size=int(15*sigma))
    conv_img = convolve_fft(image, psf, boundary='wrap')
    conv_img /= np.max(conv_img) # normalisation
    return conv_img

def plot_img(img, psf_FWHM, scale, i):
    if psf_FWHM is not None:
        img = psf_conv(img, psf_FWHM, i)
    else:
        img = img.image[0,0,i] / np.max(img.image[0,0,i])
    
    vmax = np.nanmax(img)
    vmin = 1e-3 * vmax
    if scale == 'sqrt':
        norm = mcolors.PowerNorm(0.5, vmin=vmin, vmax=vmax, clip=True)
    elif scale == 'lin':
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    plt.imshow(img, norm=norm, cmap='cividis', origin='lower')
    plt.show()

# calculate the dark lane thickness and plot the brightness profile
def dlthick(im, i, psf_FWHM, fit_lim, window, plot:bool):
    if psf_FWHM is not None:
        img_array = psf_conv(im, psf_FWHM, i)
    else:
        img_array = im.image[0,0,i]
    lim_neg = im.extent[1]
    lim_pos = im.extent[0]
    x = np.arange(lim_neg, lim_pos, im.pixelscale)

    mp_idx_h = int((im.header['naxis1'] - 1) / 2)
    mp_idx_v = int((im.header['naxis2'] - 1) / 2)
    x1_data = x[:mp_idx_v]
    x2_data = x[mp_idx_v:]
    norm_data = img_array[:,mp_idx_h] / np.max(img_array[:,mp_idx_h])

    y1_data = norm_data[:mp_idx_v]
    y2_data = norm_data[mp_idx_v:]
    y1_data_smoothed = savgol_filter(y1_data, window, 3)
    y2_data_smoothed = savgol_filter(y2_data, window, 3)

    y1_data_max = y1_data[np.argmax(y1_data_smoothed)]
    x1_trun = x1_data[np.where(y1_data > y1_data_max*fit_lim)]
    y2_data_max = y2_data[np.argmax(y2_data_smoothed)]
    x2_trun = x2_data[np.where(y2_data > y2_data_max*fit_lim)]

    if plot:
        plt.scatter(x, norm_data, label='data', c='C0', marker='x', s=5)
        plt.plot(x1_data, y1_data_smoothed, label='smoothed', lw=0.8, c='C6')
        plt.plot(x2_data, y2_data_smoothed, lw=0.8, c='C6')
        plt.ylim(0, 1.05)
        xlim = np.max([np.abs(x1_trun[0]), np.abs(x2_trun[-1])])
        plt.xlim(-xlim*1.1, xlim*1.1)
        plt.xlabel(r'$\Delta z$ / arcsec')
        plt.ylabel('Normalised intensity')
        plt.legend(prop={'size': 8})

    # x1_data_max = x1_data[np.argmax(y1_data_smoothed)]    
    x1_data_max = x1_data[find_peaks(y1_data_smoothed, distance=300)[0]]
    # x2_data_max = x2_data[np.argmax(y2_data_smoothed)]
    x2_data_max = x2_data[find_peaks(y2_data_smoothed, distance=300)[0]]

    if (len(x1_data_max) == 0) or (len(x2_data_max) == 0):
        print('No peak found')
        plot = True
        return
    return np.abs(x2_data_max - x1_data_max)[0]

def brightratio(im, i, psf_FWHM, fit_lim, window, plot:bool):
    if psf_FWHM is not None:
        img_array = psf_conv(im, psf_FWHM, i)
    else:
        img_array = im.image[0,0,i]
    lim_neg = im.extent[1]
    lim_pos = im.extent[0]
    x = np.arange(lim_neg, lim_pos, im.pixelscale)

    mp_idx_h = int((im.header['naxis1'] - 1) / 2)
    mp_idx_v = int((im.header['naxis2'] - 1) / 2)
    x1_data = x[:mp_idx_v]
    x2_data = x[mp_idx_v:]
    norm_data = img_array[:,mp_idx_h] / np.max(img_array[:,mp_idx_h])

    y1_data = norm_data[:mp_idx_v]
    y2_data = norm_data[mp_idx_v:]
    y1_data_smoothed = savgol_filter(y1_data, window, 3)
    y2_data_smoothed = savgol_filter(y2_data, window, 3)

    y1_data_max = y1_data[np.argmax(y1_data_smoothed)]
    x1_trun = x1_data[np.where(y1_data > y1_data_max*fit_lim)]
    y2_data_max = y2_data[np.argmax(y2_data_smoothed)]
    x2_trun = x2_data[np.where(y2_data > y2_data_max*fit_lim)]

    if plot:
        plt.scatter(x, norm_data, label='data', c='C0', marker='x', s=5)
        plt.plot(x1_data, y1_data_smoothed, label='smoothed', lw=0.8, c='C6')
        plt.plot(x2_data, y2_data_smoothed, lw=0.8, c='C6')
        plt.ylim(0, 1.05)
        xlim = np.max([np.abs(x1_trun[0]), np.abs(x2_trun[-1])])
        plt.xlim(-xlim*1.1, xlim*1.1)
        plt.xlabel(r'$\Delta z$ / arcsec')
        plt.ylabel('Normalised intensity')
        plt.legend(prop={'size': 8})

    x1_data_max = x1_data[find_peaks(y1_data_smoothed, distance=300)[0]]
    x2_data_max = x2_data[find_peaks(y2_data_smoothed, distance=300)[0]]

    if (len(x1_data_max) == 0) or (len(x2_data_max) == 0):
        print('No peak found')
        plot = True
        return
    return y1_data_max / y2_data_max

def bisect_list(a):
    g = lambda a: 2 * 3**-1.5 / H_R[:-1] - k[0] / a
    lb_loc = k[0]/a # lower bound for root finding
    solution_check = np.sum(g(a)>0)
    root_R = []
    if solution_check == 0:
        print('No solution')
        return
    else:
        for j in range(solution_check):
            g_loc = lambda z: z / (1 + (H_R[j]*z)**2) ** (3/2) - k[0]/a
            result = bisect(g_loc, lb_loc, 2**-0.5 / H_R[j], maxiter=2000)
            lb_loc = result-2
            root_R.append(result)
        root_R = np.array(root_R)
        return root_R