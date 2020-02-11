import numpy as np
import os
import matplotlib.pyplot as plt

directory = 'D:/Research/Bin Optimization/'
file = 'spec_120kV_0p5mmCu.npy'

spectrum_total = np.load(directory+file)

# Take the spectrum and get every third file
spectrum = spectrum_total[2:-1:3]
print(spectrum)
np.save(directory + file, spectrum)

#np.savetxt(directory+'lower_energies.txt', spectrum[0:100, 0], fmt='%.6f')
#np.savetxt(directory+'upper_energies.txt', spectrum[100:200, 0], fmt='%.6f')

#%%
import numpy as np
directory = 'D:/Research/Bin Optimization/Txt Attenuation/'
save_directory = 'D:/Research/Bin Optimization/'
file = 'Au4P.txt'

att = np.loadtxt(directory+file)
file = file.replace('txt', 'npy')
np.save(save_directory+file, att[:, 1])

#%%
directory = 'D:/Research/Bin Optimization/Npy Attenuation/'
file = 'I.npy'

density = 4.93

att = np.load(directory + file)
print(att[0:10])
att = np.multiply(att, density)
print(att[0:10])
np.save(directory + file, att)

