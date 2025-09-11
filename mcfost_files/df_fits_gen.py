from astropy.io import fits
import numpy as np
import os
import scipy.constants as sc
from scipy.optimize import bisect

# --- read the run parameter file (not mcfost parameter file) --- #
with open('_run_param.txt', 'r') as f: run_param_lines = f.readlines(); f.close()

path = [s for s in run_param_lines if 'pwd' in s][0].split()[1]
sdot_in = [s for s in run_param_lines if 'smlr' in s][0].split()
print(sdot_in)
sdot_ref = float(sdot_in[1]) # at 10 au in g/cm2/s
sdot_r_ref = float(sdot_in[2])
sdot_p = float(sdot_in[3])
a_st = float([s for s in run_param_lines if 'a_stl' in s][0].split()[1]) # in cm
print('a_st:', f'{(a_st*1e4):.1e}', 'um')
if_incl = float([s for s in run_param_lines if 'incl' in s][0].split()[1]) # in rad
print('IF inclination:', if_incl, 'rad')

# --- #

# --- mcfost parameter file --- #

# finds the parameter file in the current directory
para_path = [s for s in os.listdir(path) if '.para' in s][0] # provided there is only one parameter file. but there is a test before this in the bash script

# generate spatial and grain size grid from parameter file
grid = fits.open(path+'/data_disk_original/grid.fits.gz')
grain_sizes = fits.open(path+'/data_disk_original/grain_sizes.fits.gz')
grain_sizes_max = fits.open(path+'/data_disk_original/grain_sizes_max.fits.gz')
grain_sizes_min = fits.open(path+'/data_disk_original/grain_sizes_min.fits.gz')
grain_masses = fits.open(path+'/data_disk_original/grain_masses.fits.gz')
volume = fits.open(path+'/data_disk_original/volume.fits.gz')
mass_density = fits.open(path+'/data_disk_original/dust_mass_density.fits.gz')
number_density = fits.open(path+'/data_disk_original/dust_particle_density.fits.gz')
gas_density = fits.open(path+'/data_disk_original/gas_density.fits.gz')

R = grid[0].data[0,0,0] # radial grid
P = len(R) # number of radial cells
N = len(grid[0].data[1,0,:,0]) # number of vertical cells
H_max = 7 # maximum scale height THIS NEEDS FIXING

amin = grain_sizes_min[0].data[0] / 1e4 # min grain size, in cm
amax = grain_sizes_max[0].data[-1] / 1e4 # max grain size, in cm
a = grain_sizes[0].data / 1e4 # list of grain sizes in cm

# --- #

# --- quantities obtained directly from the parameter file --- #

with open(para_path, 'r') as f: para_lines = f.readlines(); f.close()

aexp = float([s for s in para_lines if 'amax' in s][0].split()[2])
dust_disc_mass, g2d = [float(a) for a in [s for s in para_lines if 'gas-to-dust' in s][0].split()[:2]]
H_100 = float([s for s in para_lines if 'scale height, r' in s][0].split()[0]) # scale height at 100 au in au
rhos = 3.5 # in g/cm3
v2m = 4*np.pi/3 * rhos # in g/cm3
p1, p2 = [int(a) for a in [s for s in para_lines if 'surface density exponent' in s][0].split()[:2]]
p = -p1
beta = float([s for s in para_lines if 'flaring exponent' in s][0].split()[0])
q = 3 - 2*beta
r_c = float([s for s in para_lines if 'Rc (AU)' in s][0].split()[3]) # in au
stellar_mass = float([s for s in para_lines if 'M (solar mass)' in s][0].split()[2]) # in solar masses

# --- #

# --- calculate other parameters and quantities --- #
Omega = (np.sqrt(sc.G * stellar_mass * 2e30) * 1.5e11**(-1.5)) * R**(-1.5) # Keplerian angular frequency in s^-1
H = H_100 * (R / 100)**beta # scale height in au
H_cm = H * 1.5e13 # in cm
H_R = H / R # aspect ratio
c_s = Omega * H * 1.5e8 # isothermal sound speed in km/s

# gas surface density
Sigma_data = 2*np.sum(gas_density[0].data * np.tile(H_cm, (N, 1))*H_max/N, axis=0)
Sigma_0 = Sigma_data[np.argmin(np.abs(R-1))] # sd at 1 au in g/cm2

# # exponentially-tapered surface density profile
# sd = Sigma_0 * R**p1 * np.exp(-(R/r_c)**(2+p2)) # in g/cm2

# power law surface density profile
sd = Sigma_0 * R**p1

# defining the z grid
z = (grid[0].data[1,0] / np.tile(H, (N,1))).T[0]

# surface mass loss rate function
sdot = sdot_ref * (R/sdot_r_ref)**(sdot_p) # in g/cm2/s
u0 = np.sqrt(2*np.pi) / Omega * sdot / sd
rho0 = sd / np.sqrt(2*np.pi) / (c_s * 1e5) * Omega
k = sdot / rhos / Omega # in cm

# --- calculate IF height --- #

# # self-consistent description following HJLS94 
# c_si = 12.9 # in km/s
# v_w = 0.5 * c_si
# rho_if = (c_si**2 + v_w**2)/(2*c_s**2) * sdot/(v_w*1e5) * ( 1 + np.sqrt( 1 - ( 2*c_s*v_w / (c_si**2 + v_w**2 ) )**2 ) )
# z_IF = np.sqrt(2 * np.log(sd / np.sqrt(2*np.pi) / H_cm / rho_if))

# constant inclination following CA16, HC21
if np.isnan(if_incl): z_IF = np.full((P,), 1e30) # effectively infinite height if no inclination is provided
else: z_IF = H_R**(-1) * np.tan(if_incl)

# --- #

# --- generate the density grid to feed into mcfost --- #

# if no global stalling grain size supplied, calculates the minimum stalling grain size at each R
if np.isnan(a_st):
    a_ms = np.sqrt(2) * k * H_R # minimum stalling grain size at each radius
    print(f'Global minimum stalling grain size: {a_ms[0]*1e4:.2f} um') # taken to be at R_in
# if a global stalling grain size is supplied, use this everywhere
else: a_ms = np.full((P,), a_st); print(f'Minimum stalling grain size everywhere: {a_st*1e4:.2f} um')
a_s = [a[a>=s] for s in a_ms] # list of grain sizes at each radius which can stall - not used

# setting up grid for calculation
a_s_grid = np.tile(a, (P,1)) # each column is grain size array

# each radius can have a different value of a_ms. at each radius, if a < a_ms, set to nan. these grains are all in the wind
for i, s in enumerate(a_ms): (a_s_grid[i])[a_s_grid[i]<s] = np.nan
a_s_grid = a_s_grid.T # now each row is a grain size array at a specific radius

# calculate the stalling heights at each radius for each grain size, using the approximate calculation h = k/a
height_grid = np.tile(k, (len(a),1)) / a_s_grid # grid of stalling heights. each column is a radius, each row is a grain size

# hg_unaltered_hdu = fits.ImageHDU(height_grid) # for debugging

# impose the ionisation front. the stalling surface is taken to be the minimum of IF height and max stalling height
# one further imposition is the grid. if neither the IF or stalling height is within the grid, then set to nan
ms_height = np.min((1 / np.sqrt(2) / H_R, z_IF, np.full((P,), H_max)), axis=0)
for i in range(P): height_grid[:,i][(height_grid[:,i] > ms_height[i])] = np.nan

# invert ms_height for list of stalling grain sizes
a_ms_new = k / ms_height

# set up the density grid
n_a = []

# index of the vertical cell which contains the stalled grain at each radius for each grain size
z_idx_list = []

# assign densities to each cell in the grid
z_int = np.append([0], z) # cell interfaces
for j, t in enumerate(a):
    # indices of the vertical cell which contains the stalled grain t, at each radius. so length P. if q is np.nan, then returns -1
    z_idx = [ (np.sum(z_int<q) - 1) for q in height_grid[j] ]
    n_zr = np.zeros((N, P)) # the spatial grid for each grain size
    for i, s in enumerate(z_idx):
        if (s == -1): n_zr[:, i] = np.copy(gas_density[0].data)[:,i] * t**-2.5 # when the grain doesn't stall. then scale by gas density (wind)
        else: n_zr[s, i] = (sd / H)[i] * t**-2.5 # when the grain stalls
    n_zr = np.reshape(n_zr, (1, N, P)); n_a.append(n_zr) # add to n_a
    z_idx_list.append(z_idx)

# --- #

# --- output to fits file --- #
density_grid = fits.PrimaryHDU(n_a)
gs_hdu = fits.ImageHDU(a*1e4)
ms_height_hdu = fits.ImageHDU(ms_height, name='MS height')
ms_grain_hdu = fits.ImageHDU(a_ms_new*1e4, name='MS grain sizes')
density_grid.header['read_n_a'] = 0
hdulist = fits.HDUList([density_grid, gs_hdu, ms_height_hdu, ms_grain_hdu])
hdulist.writeto('settled_density.fits', overwrite=True)

# # DEBUGGING
# z_idx_hdu = fits.ImageHDU(z_idx_list, name='z_idx')
# hdulist = fits.HDUList([density_grid, gs_hdu, ms_height_hdu, ms_grain_hdu, z_idx_hdu, hg_unaltered_hdu])
# hdulist.writeto('debug.fits', overwrite=True)

print('Wrote density fits file.')
# --- #