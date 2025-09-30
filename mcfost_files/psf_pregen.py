import os
import numpy as np
import stpsf

# calculate pixelscale from parameter file
with open('_run_param.txt', 'r') as f: run_param_lines = f.readlines(); f.close()
path = [s for s in run_param_lines if 'pwd' in s][0].split()[1]

para_path = [s for s in os.listdir(path) if '.para' in s][0] # provided there is only one parameter file. but there is a test before this in the bash script

with open(para_path, 'r') as f: para_lines = f.readlines(); f.close()

grid_line = [s for s in para_lines if 'grid (nx,ny)' in s][0]

nx, ny, size_max = grid_line.split()[:3]
nx, ny, size_max = int(nx), int(ny), float(size_max)

distance = float([s for s in para_lines if 'distance (pc)' in s][0].split()[0]) # in pc

pixelscale = size_max / np.max([nx, ny]) / distance

# load stpsf and set pixelscale
nrc = stpsf.NIRCam()
nrc.pixelscale = pixelscale
miri = stpsf.MIRI()
miri.pixelscale = pixelscale

os.mkdir('psf_fits')

nrc_list = ['F200W', 'F444W']
miri_list = ['F770W', 'F1280W', 'F2100W']

for s in nrc_list:
    nrc.filter = s
    psf = nrc.calc_psf(oversample=4)
    psf.writeto(f'psf_fits/{s}_psf.fits', overwrite=True)
    print(f'Wrote psf_fits/{s}_psf.fits')

for s in miri_list:
    miri.filter = s
    psf = miri.calc_psf(oversample=4)
    psf.writeto(f'psf_fits/{s}_psf.fits', overwrite=True)
    print(f'Wrote psf_fits/{s}_psf.fits')