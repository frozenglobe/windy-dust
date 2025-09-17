from astropy.io import fits
import pymcfost as mcfost
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm, ticker
from matplotlib.colors import LogNorm
import os
from astropy.convolution import Gaussian2DKernel, convolve_fft
from scipy.signal import savgol_filter, find_peaks
import pandas as pd
import stpsf
import copy
from matplotlib.patches import Ellipse, Circle

# Rundown of program:
# 1. identifies valid image directories
# 2. loads and plots image files - this is necessary to get
#    properties of the image itself, e.g. extent, pixelscale
# 3. uses STPSF to generate PSFs for JWST NIRCam, MIRI and performs
#    image convolution
# 4. plots images of six wavelengths and saves image
# 5. generates brightess profiles to calculate asymmetry factor and
#    dark lane thickness
# 6. outputs plots of SED, asymmetry factor and dark lane thickness
# 7. linearly interpolates model SED for total integrated flux at
#    each image wavelength
# 8. outputs SED flux, asymmetry factor and dark lane thickness
#    for each of the image wavelengths

# --- LOADING FILES --- #

# checks if a string is a float
def isFloat(string):
    try: float(string); return True;
    except: return False

# loads all data directories
data_list = [s for s in os.listdir() if 'data_' in s]
sed = fits.open("data_th/sed_rt.fits.gz") # load SED
temp = fits.open("data_th/Temperature.fits.gz") # load temperature structure
mass_density = fits.open("data_disk/dust_mass_density.fits.gz") # load mass density structure
grid = fits.open("data_disk/grid.fits.gz") # load grid

# loads mcfost parameter file
para_path = [s for s in os.listdir() if '.para' in s][0]
with open(para_path, 'r') as f: para_lines = f.readlines(); f.close()

# loads run parameter file
with open('_run_param.txt', 'r') as f: run_param_lines = f.readlines(); f.close()
a_st = float([s for s in run_param_lines if 'a_stl' in s][0].split()[1]) * 1e4 # in um
if_incl = float([s for s in run_param_lines if 'incl' in s][0].split()[1])

# building the grid
R = grid[0].data[0,0,0]
P = len(R)
N = len(grid[0].data[1,0,:,0]) # number of vertical cells
H_100 = float([s for s in para_lines if 'scale height, r' in s][0].split()[0]) # scale height at 100 au in au
beta = float([s for s in para_lines if 'flaring exponent' in s][0].split()[0])
q = 3 - 2*beta
H = H_100 * (R / 100)**beta # scale height in au
z = (grid[0].data[1,0] / np.tile(H, (N,1))).T[0]

# constant inclination for IF following CA16, HC21
if np.isnan(if_incl): z_IF = np.full((P,), 1e30) # effectively infinite height if no inclination is provided
else: z_IF = R/H * np.tan(if_incl)

# loading image files
# splits each directory string and returns those where the 2nd entry is a float
idx = np.nonzero([isFloat(t.split('_')[1]) for t in data_list])[0]
img_paths = np.array(data_list)[idx] # currently unsorted

# now sort by increasing wavelength
wl_list = [float(s.split('_')[1]) for s in img_paths]
sort_idx = np.array(wl_list).argsort()
img_paths = img_paths[sort_idx]

# load all images and plot first image
img_list = [mcfost.Image(s) for s in img_paths]
img_list[0].plot()

# extract image properties 
lim_pos, lim_neg = img_list[0].extent[:2]
pixelscale = img_list[0].pixelscale
ct_px = (np.array([img_list[0].nx, img_list[0].ny])/2).astype(int) # central pixel idxs
# ... and place into dictionary
img_param = {'lim_pos': lim_pos, 'lim_neg': lim_neg, 'pixelscale': pixelscale, 'ct_px': ct_px}

# --- #

# --- load data from Duchene et al. (2024) --- #
sed_wavelength, sed_data = np.load('/data/jhyl3/windy-dust/mcfost_files/sed_data.npy')
wavelength_dlc, dlt_data, sett10_model = np.load('/data/jhyl3/windy-dust/mcfost_files/dlc_data.npy')
br_data = np.load('/data/jhyl3/windy-dust/mcfost_files/br_data.npy')[1]

# --- #


# --- generate mass density, SED and temperature plots --- #

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3))

cax1 = ax1.contourf(R, z, mass_density[0].data, levels=10**np.linspace(-28,-12,100), cmap='viridis', norm=LogNorm())
ax1.plot(R, z_IF, color='white', linestyle='--', linewidth=0.8)
clb1 = fig.colorbar(cax1, ax=ax1, location="top", shrink=0.8)
clb1.set_ticks(10**np.linspace(-28, -12, 5))
clb1.set_label(r'$\rho_d$ / $\mathrm{g\,cm^{-3}}$')
ax1.set_ylim(0,7)
ax1.set_yticks(np.arange(0,8,1))
ax1.set_ylabel(r'$z$ / $H$')
ax1.set_xlabel(r'$R$ / au')
ax1.set_xticks(np.arange(0,401,100))
ax1.text(230, 6.2, r'mass density', fontsize=9, color='white')

cax2 = ax2.contourf(R, z, temp[0].data, levels=10**np.linspace(0,3,100), vmin=10, vmax=200, norm=LogNorm(), cmap='coolwarm')
ax2.plot(R, z_IF, color='white', linestyle='--', linewidth=0.8)
clb2 = fig.colorbar(cax2, ax=ax2, ticks=10**np.linspace(0,3,4), location="top", shrink=0.8)
clb2.set_label(r'$T$ / $\mathrm{K}$')
ax2.set_ylim(0,7)
ax2.set_yticks(np.arange(0,8,1))
ax2.set_ylabel(r'$z$ / $H$')
ax2.set_xlabel(r'$R$ / au')
ax2.set_xticks(np.arange(0,401,100))
ax2.text(240, 6.2, r'temperature', fontsize=9, color='black')

color = cm.Dark2(np.linspace(0,1,4))
wavelength = sed[1].data
ax3.plot(wavelength, sed[0].data[1,0,-1], lw=0.8, c=color[0])
ax3.plot(wavelength, sed[0].data[2,0,-1], lw=0.8, c='#DC267F', label='scattered stellar emission')
ax3.plot(wavelength, sed[0].data[3,0,-1], lw=0.8, c='#FFB000', label='dust thermal emission')
ax3.plot(wavelength, sed[0].data[4,0,-1], lw=0.8, c='#648FFF', label='scattered dust emission')
ax3.plot(wavelength, sed[0].data[0,0,-1], lw=1, c='black', label='total')
ax3.scatter(sed_wavelength, sed_data, c='C8', s=10, marker='o')
ax3.set_ylim(1e-17, 3e-12)
ax3.set_xlim(1e-1, 3e3)
ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.legend(loc='upper left', prop={'size':8}, reverse=True, labelspacing=0.3)
ax3.set_ylabel(r'$\lambda\,F_{\lambda}$ / $\mathrm{W\,m^{-2}}$', labelpad=-4)
ax3.set_xlabel(r'$\lambda$ / $\mathrm{\mu m}$')
ax3.text(6e2, 1.2e-12, r'SED', fontsize=9, color='black')

fig.subplots_adjust(wspace=0.4)
plt.savefig('model_structure.png', dpi=300, bbox_inches='tight')

# --- #


# --- POINT SPREAD FUNCTION GENERATION --- #

# PSF generation is slow, so please load pre-generated PSFs
# choose a directory.

psf_dir = '/data/jhyl3/windy-dust/mcfost_files/psf_fits'

wl_list = [float(s.split('_')[1]) for s in img_paths]

# load PSF fits files from `psf_dir'
psf_hdus = [fits.open(os.path.join(psf_dir, s)) for s in os.listdir(psf_dir)]

# need to sort by wavelength
sort_key = [int(s.split('F')[1].split('W')[0]) for s in os.listdir(psf_dir)]
psf_hdus = [s for _, s in sorted(zip(sort_key, psf_hdus))]

# extract kernels for each JWST filter. 3 uses the third ImageHDU, named `DET_DIST' which is
# their best guess for the PSF actually observed on a real detector, including real-world effects
# e.g. geometric distortion, detector charge transfer, interpixel capacitance. see STPSF docs
jwst_kernels = [s[3].data for s in psf_hdus]
# use a 2D Gaussian for HST
fwhm_list = [0.072] # FWHM for HST F814W in arcsec
# measure FWHM for each JWST PSF
for s in psf_hdus: fwhm_list.append(stpsf.measure_fwhm(s, ext=3))

# convolve the image with a 2D Gaussian kernel
def gaussian_conv(img, psf_FWHM, i):
    image = img.image[0,0,i]
    sigma = psf_FWHM / (img.pixelscale * 2 * np.sqrt(2*np.log(2)))
    psf = Gaussian2DKernel(sigma, x_size=int(15*sigma), y_size=int(15*sigma))
    conv_img = convolve_fft(image, psf, boundary='wrap')
    conv_img /= np.max(conv_img) # normalisation
    return conv_img

# if shallow copy then img_list gets updated ... 
conv_img_list = copy.deepcopy(img_list)
conv_img_list[0].image[0,0,-1] = gaussian_conv(img_list[0], fwhm_list[0], -1)
for i, s in enumerate(conv_img_list[1:]): s.image[0,0,-1] = convolve_fft(img_list[i+1].image[0,0,-1], jwst_kernels[i], boundary='wrap') # excluding Hubble

# --- SIX WAVELENGTH IMAGE PLOT --- #
# img_list_plot = [0.8, 2.0, 4.4, 7.7, 12.8, 21.0]
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2,3, figsize=(9,6), sharex=True, sharey=True)
axes = [ax1, ax2, ax3, ax4, ax5, ax6]

for i, s in enumerate(conv_img_list):
    s.plot(i=-1, vmax=1, scale='sqrt', 
           ax=axes[i], cmap='cividis', colorbar=0, norm=1,
           no_ylabel=1, no_xlabel=1)
    axes[i].text(2.7, 3, f'{wl_list[i]}'+r' $\mathrm{\mu m}$', fontsize=10, c='white')
    axes[i].add_patch(Circle(axes[i].transLimits.inverted().transform((0.125, 0.125)), radius=fwhm_list[i]/2, fill=1, color='gray'))

fig.subplots_adjust(hspace=0.04, wspace=0.04)
fig.savefig('model_img.png', dpi=300, bbox_inches='tight')

# --- generate brightness profiles ---
x = np.arange(lim_neg, lim_pos, pixelscale)
color = cm.viridis(np.linspace(0,1,6))
rcParams['figure.figsize'] = (4,2)
for i, s in enumerate(img_list):
    plt.plot(x, s.image[0,0,-1][:,ct_px[0]] / np.max(s.image[0,0,-1][:,ct_px[0]]), 
             label=f'{wl_list[i]}'+r' $\mathrm{\mu m}$', lw=0.9, c=color[i])
plt.legend(prop={'size':8}, loc='upper right')
plt.xlim(-2, 3)
# plt.savefig()

# outputs brightness ratio (br_ratio) and dark lane thickness (dlt)
# 'im' is the HDU file, 'fit_lim' forces the program to search for a local maxima close to the mp
# 'window' is for the Savitzky-Golay filter, 'img_param' contains parameters e.g. pixelscale, extent
def dlcalc(im, fit_lim, window, img_param, plot=False):
    img_array = im.image[0,0,-1] # obtains the array associated with the image
    x = np.arange(img_param['lim_neg'], img_param['lim_pos'], im.pixelscale) # pixel coordinates

    x1_data = x[:img_param['ct_px'][0]]
    x2_data = x[img_param['ct_px'][0]:]
    norm_data = img_array[:,img_param['ct_px'][0]]/ np.max(img_array[:,img_param['ct_px'][0]])

    y1_data = norm_data[:img_param['ct_px'][0]]
    y2_data = norm_data[img_param['ct_px'][0]:]
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
        print('No peak found for ' + f'{im.wl}' + ' um')
        plot = True
        return {'br_ratio': None, 'dlt': None}
    return {'br_ratio': y1_data_max/y2_data_max, 'dlt': np.abs(x2_data_max - x1_data_max)[0]}

br_list = [dlcalc(s, fit_lim=0.6, window=10, img_param=img_param)['br_ratio'] for s in conv_img_list]
dlt_list = [dlcalc(s, fit_lim=0.6, window=10, img_param=img_param)['dlt'] for s in conv_img_list]

# --- calculate model SED ---
sed_model_wl = sed[1].data
sed_list = np.interp(wl_list, sed_model_wl, sed[0].data[0,0,-1])

output = pd.DataFrame({'wl': wl_list, 'sed': sed_list, 'br_ratio': br_list, 'dlt': dlt_list})
output.to_pickle('img_output.pkl')

# --- plot SED, brightness ratio and dark lane thicknesses ---
fig, (ax3, ax1, ax2) = plt.subplots(1, 3, figsize=(9, 2))

ax3.plot(sed_model_wl, sed[0].data[0,0,-1], lw=1, c='C3', label=r'Wind' + f'{a_st}' + r'$\,\mathrm{\mu m}$ stalling')
ax3.scatter(sed_wavelength, sed_data, c='black', s=10, marker='o')

ax3.set_xscale('log')
ax3.set_yscale('log')
ax3.set_ylim(1e-17, 8e-13)
ax3.set_xlabel(r'$\lambda$ / $\mathrm{\mu m}$')
ax3.set_ylabel(r'$\lambda\,F_{\lambda}$ / $\mathrm{W\,m^{-2}}$')
ax3.text(1e-1, 3e-13, r'SED', fontsize=9, color='black')

ax1.scatter(wavelength_dlc, br_list, c='C3', s=10, marker='x', label=r'Wind ' + f'{a_st}' + r'$\,\mathrm{\mu m}$ stalling')
ax1.plot(wavelength_dlc, br_list, c='C3', lw=0.8, alpha=0.5)
ax1.scatter(wavelength_dlc, br_data, c='black', s=10, marker='o')
ax1.set_ylim(0,1)
ax1.set_xscale('log')
ax1.set_xlabel(r'$\lambda$ / $\mathrm{\mu m}$', labelpad=-2)
ax1.set_ylabel(r'Asymmetry factor')
ax1.legend(prop={'size': 8}, loc='upper right')

ax2.plot(wavelength_dlc, sett10_model, c='m', lw=0.8, alpha=1, ls='--', label=r'D24 $10\,\mathrm{\mu m}$ settling')
ax2.scatter(wavelength_dlc, dlt_list, c='C3', s=10, marker='x')
ax2.plot(wavelength_dlc, dlt_list, c='C3', lw=0.8, alpha=0.5)
ax2.scatter(wavelength_dlc, dlt_data, c='black', s=10, marker='o')
ax2.legend(prop={'size': 8}, loc='upper right', labelspacing=0.1)
ax2.set_xscale('log')
ax2.set_ylim(0.5, 1.7)
ax2.set_xlabel(r'$\lambda$ / $\mathrm{\mu m}$', labelpad=-2)
ax2.set_ylabel(r'$d_\mathrm{neb}$ / arcsec')
fig.subplots_adjust(wspace=0.5)
fig.savefig('model_darklane.png', dpi=300, bbox_inches='tight')