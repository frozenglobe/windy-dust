import numpy as np
import os


with open('_run_param.txt', 'r') as f: run_param_lines = f.readlines(); f.close()
path = [s for s in run_param_lines if 'pwd' in s][0].split()[1]

para_path = [s for s in os.listdir(path) if '.para' in s][0] # provided there is only one parameter file. but there is a test before this in the bash script

with open(para_path, 'r') as f: para_lines = f.readlines(); f.close()

grain_line_no = [i for i, s in enumerate(para_lines) if 'amax' in s][0] # to target grain size
grain_list = para_lines[grain_line_no].split()
amin, amax = grain_list[:2]
aexp = float(grain_list[2])

R_in = float([s for s in para_lines if 'Rin, edge' in s][0].split()[0]) # inner radius in au
r_c = float([s for s in para_lines if 'Rc (AU)' in s][0].split()[3]) # in au
N = int([s for s in para_lines if 'n_rad (log distribution)' in s][0].split()[1]) # number of vertical cells
H_100 = float([s for s in para_lines if 'scale height, r' in s][0].split()[0]) # scale height at 100 au in au
beta = float([s for s in para_lines if 'flaring exponent' in s][0].split()[0])
q = 3 - 2*beta

H_max = 7 # maximum scale height THIS NEEDS FIXING

# calculate necessary grain size resolution
H_R_in = H_100 * (R_in / 100)**beta / R_in # aspect ratio at R_in
M_exact = np.log10(float(amax)/float(amin)) * np.log(10) * N / np.sqrt(2) / H_R_in / H_max
M = int(round(M_exact, -2) + 3 * 10**( np.floor(np.log10(M_exact) - 1 ) ) ) # round to nearest hundred, then add 300 for safety

# replace the n_grain line in the parameter file with the new value
new_lines = para_lines.copy()
grain_list[3] = str(M)
new_lines[grain_line_no] = '  ' + ' '.join(grain_list) + '\n'

# write to new and old parameter file
with open('old_para.bak', 'w') as f: f.writelines(para_lines); f.close() # backup of old parameter file
with open(para_path, 'w') as f: f.writelines(new_lines); f.close()