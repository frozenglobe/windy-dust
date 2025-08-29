from astropy.io import fits
import numpy as np
import os
import scipy.constants as sc
from scipy.optimize import bisect

with open('_run_param.txt', 'r') as f:
    run_param_lines = f.readlines()

path = [s for s in run_param_lines if 'pwd' in s][0].split()[1]
sdot_in = [s for s in run_param_lines if 'smlr' in s][0].split()
print(sdot_in)
sdot_ref = float(sdot_in[1]) # at 10 au in g/cm2/s
sdot_r_ref = float(sdot_in[2])
sdot_p = float(sdot_in[3])
a_sett = float([s for s in run_param_lines if 'a_stl' in s][0].split()[1]) # in cm
print('a_sett:', f'{(a_sett*1e4):.1e}', 'um')
if_incl = float([s for s in run_param_lines if 'incl' in s][0].split()[1]) # in rad
print('IF inclination:', if_incl, 'rad')

f.close()

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

# --- quantities from the parameter file, via disk_struct ---

R = grid[0].data[0,0,0] # radial grid
P = len(R) # number of radial cells
N = len(grid[0].data[1,0,:,0]) # number of vertical cells
H_max = 7 # maximum scale height THIS NEEDS FIXING

amin = grain_sizes_min[0].data[0] / 1e4 # min grain size, in cm
amax = grain_sizes_max[0].data[-1] / 1e4 # max grain size, in cm
a = grain_sizes[0].data / 1e4 # list of grain sizes in cm

# --- quantities obtained directly from the parameter file ---

with open(para_path, 'r') as f:
    para_lines = f.readlines()

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

f.close()

# --- calculate other parameters and quantities ---
Omega = (np.sqrt(sc.G * stellar_mass * 2e30) * 1.5e11**(-1.5)) * R**(-1.5) # Keplerian angular frequency
H = H_100 * (R / 100)**beta # in au
H_cm = H * 1.5e13 # in cm
H_R = H / R # aspect ratio
c_s = Omega * H * 1.5e8 # in km/s

# --- gas surface density ---
Sigma_data = 2*np.sum(gas_density[0].data * np.tile(H_cm, (N, 1))*H_max/N, axis=0)
Sigma_0 = Sigma_data[np.argmin(np.abs(R-1))] # sd at 1 au in g/cm2
# sd = Sigma_0 * R**p1 * np.exp(-(R/r_c)**(2+p2)) # in g/cm2
sd = Sigma_0 * R**p1

# --- 

z = (grid[0].data[1,0] / np.tile(H, (N,1))).T[0]

sdot = sdot_ref * (R/sdot_r_ref)**(sdot_p) # in g/cm2/s
u0 = np.sqrt(2*np.pi) / Omega * sdot / sd
rho0 = sd / np.sqrt(2*np.pi) / (c_s * 1e5) * Omega
k = sdot / rhos / Omega # in cm

# --- calculate IF height ---
# c_si = 12.9 # in km/s
# v_w = 0.5 * c_si
# rho_if = (c_si**2 + v_w**2)/(2*c_s**2) * sdot/(v_w*1e5) * ( 1 + np.sqrt( 1 - ( 2*c_s*v_w / (c_si**2 + v_w**2 ) )**2 ) )
# z_IF = np.sqrt(2 * np.log(sd / np.sqrt(2*np.pi) / H_cm / rho_if))

z_IF = H_R**(-1) * np.tan(if_incl)

# --- separating into stalled and wind reservoirs ---
a_s = a[a >= a_sett]
a_w = a[a < a_sett]

g = lambda a: 2 * 3**-1.5 / H_R - k[0] / a

root_list = []
lower_bounds = k[0]/a_s # lower bound for root finding

for i in range(len(a_s)):
    root_R = []
    lb_loc = lower_bounds[i]-0.1 # local lower bound at specific grain size
    solution_check = np.sum(g(a_s[i])>0) # length of grain sizes that have solutions
    if solution_check == 0:
        root_list.append(np.full((P,), np.nan))
        continue
    for j in range(solution_check):
        g_loc = lambda z: z / (1 + (H_R[j]*z)**2) ** (3/2) - k[0]/a_s[i]
        result = bisect(g_loc, lb_loc, 2**-0.5 / H_R[j], maxiter=2000)
        lb_loc = result-2
        root_R.append(result)
    root_R = np.array(root_R)
    root_full_R = np.full((P,), np.nan)
    root_full_R[:solution_check] = root_R
    root_list.append(root_full_R)

root_list = np.array(root_list)

# filter out values greater than max_settle_height
root_list_copy = np.copy(root_list)

for i in range(len(R)):
    max_settle_height_loc = z_IF[i]
    root_list_copy[:,i][(root_list_copy[:,i] > max_settle_height_loc)] = np.nan
settle_idx = np.sum(np.sum(~np.isnan(root_list_copy), axis=1)==0)
a_s2 = np.copy(a_s)[settle_idx:] # settled grain sizes after filtering
root_list_copy = root_list_copy[settle_idx:] # so it is the same size as a_s
a_w2 = sorted(set(a) - set(a_s2)) # sorted list of grain sizes that are not settled
scale_s = sd / H # for each radius
scale_w = np.copy(gas_density[0].data) # for each cell
n_a = []

for i in range(len(a_w)):
    n_ZR = scale_w * a_w[i]**-2.5
    # n_ZR = np.zeros((N,P)) # don't put any grains in the wind
    n_a_append = np.reshape(n_ZR, (1, N, P))
    n_a.append(n_a_append)

for i in range(len(a_s)):
    root_list_loc = root_list_copy[i]
    root_list_loc_s = root_list_loc[~np.isnan(root_list_loc)]
    n_ZR = np.zeros((N,P))
    # for j in range(len(root_list_loc_s), P):
        # n_ZR[:, j] = scale_w[:,j] * a_s[i]**-2.5
    z_loc_old = np.sum(root_list_loc_s[0] > z[:-1])
    for j in range(len(root_list_loc_s)):
        z_loc = np.sum(root_list_loc_s[j] > z[:-1])
        n_ZR[z_loc_old:z_loc+1, j] = scale_s[j] * a_s[i]**-2.5
        z_loc_old = z_loc
    n_a_append = np.reshape(n_ZR, (1, N, P))
    n_a.append(n_a_append)

n_a = np.array(n_a)

density_grid = fits.PrimaryHDU(n_a)
grain_size = fits.ImageHDU(a*1e4)
density_grid.header['read_n_a'] = 0
hdulist = fits.HDUList([density_grid, grain_size])
hdulist.writeto('settled_density.fits', overwrite=True)