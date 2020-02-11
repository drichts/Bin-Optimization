import numpy as np
from scipy.special import factorial
import matplotlib.pyplot as plt


def optimize(k_edge, material, concentration, filter_type, filter_thickness, directory='D:/Research/Bin Optimization/'):
    """
    Calculate the SNR for a number of bin widths
    :param k_edge: The k-edge of the material in keV
    :param material: The contrast material with concentration in wt % (e.g. 'Au')
    :param concentration: The concentration of the material, i.e. 5, 3, 1, 0.5 (in % weight)
    :param filter_type: The beam filter material (e.g. 'Al')
    :param filter_thickness: The filter thickness (in mm)
    :param directory: The directory in which all the beam spectrum and attenuation info is
    :return: The SNR values, the width values, and the energy width for 1 'width'
    """
    SDD = 30  # Source to detector distance
    N = 1  # The average number of photons in the air scan

    # Load the energy spectrum
    spectrum = np.load(directory + '/Beam Spectrum/corrected-spectrum_120kV.npy')  # Load the energy spectrum
    energies = spectrum[:, 0]
    energy_width = energies[1] - energies[0]
    spectrum_weights = spectrum[:, 1]  # Relative weights of all energies

    # Filter the spectrum with the desired filtering
    spectrum_filt = filter_spectrum(filter_type, filter_thickness, spectrum_weights)

    # Filter the spectrum due to air from source to phantom
    spectrum_filt = filter_air(SDD-3, spectrum_filt)

    # Multiply the spectrum by the number of photons N
    spectrum_filt = np.multiply(spectrum_filt, N)

    # Find the closest energy to k-edge energy in the energy matrix to find the correct index
    near_energy, idx, above = find_nearest(energies, k_edge/1000)

    # Find largest bin width possible with our spectrum data (50 points with energies, or less)
    width = bin_width(idx, len(energies))

    # Keep track of the SNR for each bin width from 1 to width
    snr = np.zeros(width)

    # Calculate the SNR for each bin width (1 to width)
    for w in np.arange(width):
        # Number of photons in left and right bin
        lb, rb = norm_mean_photons(spectrum_filt, idx, w + 1, material, concentration, above)
        #print(lb, rb)
        poisson_values = calc_4_poisson(lb, rb)
        snr[w] = SNR(poisson_values)

    width_values = np.multiply(np.arange(1, width+1), energy_width*1000)
    return snr, width_values


def filter_phantom(material, concentration, spectrum, folder='D:/Research/Bin Optimization/Npy Attenuation/'):
    """
    This function filters the beam through 1D length of 3 cm with differing material compositions
    2.4 cm of phantom (PLA), 0.1 cm of PCR tube, and 0.5 cm of the contrast material
    :param material: The contrast material element, e.g. 'Au', 'Gd', 'Dy', 'I'
    :param concentration: The concentration of the material, i.e. 5, 3, 1, 0.5
    :param spectrum: The spectrum of the beam you'd like to filter
    :param folder: The folder containing the linear attenuation correction files
    :return:
    """
    # Load the attenuation coefficients for PLA, PCR tube, and the material.

    pla = np.load(folder + 'PLA.npy')
    mat = np.load(folder + material + '.npy')
    water = np.load(folder + 'H2O.npy')
    poly = np.load(folder + 'Polypropylene.npy')

    # Get the attenuation of the mixture of water and contrast material
    concentration = concentration/100
    mat = np.multiply(mat, concentration)
    water = np.multiply(water, (1 - concentration))

    mat = np.add(mat, water)

    pla_corr = np.multiply(pla, -2.4)  # Corrects for 2.4 cm of phantom material
    poly_corr = np.multiply(poly, -0.1)  # Corrects for 0.1 cm of PCR tube
    mat_corr = np.multiply(mat, -0.5)  # Corrects for the material attentuation
    #pla_corr = np.multiply(pla, -2)
    #mat_corr = np.multiply(mat, -1)

    attenuation_corr = np.add(pla_corr, mat_corr)
    attenuation_corr = np.add(attenuation_corr, poly_corr)

    # Take the exp of the attenuation correction
    attenuation_corr = np.exp(attenuation_corr)

    # Filter the spectrum
    new_spectrum = np.multiply(spectrum, attenuation_corr)

    return new_spectrum


def filter_spectrum(filter_type, filter_thickness, spectrum, directory='D:/Research/Bin Optimization/Npy Attenuation/'):
    """
    This function takes the initial beam spectrum and attenuates it according to the thickness of filter placed at the
    origin of the beam
    :param filter_type: Type of beam filtration used, e.g. 'Al', 'Cu'
    :param filter_thickness: The thickness of your filter in mm
    :param spectrum: The beam spectrum to be filtered
    :param directory: The folder with the linear attenuation files
    :return: The filtered spectrum as an array
    """

    # Load the filter mass attenuation coefficients
    filter = np.load(directory + filter_type + '.npy')

    # Calculate the spectrum correction according to the exponential term of Beer's Law
    filter = np.multiply(filter, filter_thickness/10)  # Multiply the attenuation by the thickness of the filter
    filter_corr = np.exp(-filter)

    spectrum_filtered = np.multiply(spectrum, filter_corr)

    return spectrum_filtered


def filter_air(distance, spectrum, directory='D:/Research/Bin Optimization/Npy Attenuation/'):
    """
    This function filters the beam through a distance of air
    :param distance: The distance the beam travels in air (in cm)
    :param spectrum: The beam spectrum to be filtered
    :param directory: The folder with the linear attenuation files
    :return: The filtered spectrum as an array
    :return:
    """
    air = np.load(directory + 'Air.npy')  # Load the air attenuation coefficients at all energies

    # Calculate the spectrum air correction according to the exponential term of Beer's Law
    air = np.multiply(air, distance)
    air_corr = np.exp(-air)

    spectrum_filtered = np.multiply(spectrum, air_corr)

    return spectrum_filtered


def norm_mean_photons(spectrum, idx, bw, contrast_mat, concentration, above):
    """
    This function calculates the average normalized number of photons that hit the detector for both the left and right
    bin
    :param spectrum: The beam spectrum hitting the phantom
    :param idx: The index of the k-edge
    :param bw: the bin width (is in terms of number of indices in the energy matrix)
    :param contrast_mat: the contrast material (e.g. 'Au', 'Lu')
    :param concentration: The concentration of the material, i.e. 5, 3, 1, 0.5
    :param above: True if the energy in the energy spectrum is above the k-edge
    :return: The number of photons in the left bin and the right bin
    """

    spectrum = np.asarray(spectrum)

    # Filter airscan beam through the 3 cm of air that would be there in place of the phantom
    spectrum_air = filter_air(3, spectrum)

    # Filter phantom beam through the phantom with the contrast material
    spectrum_phantom = filter_phantom(contrast_mat, concentration, spectrum)

    # If the energy is above, include that energy in the bin above the k-edge
    if above:
        left_index = np.arange(idx - bw, idx)
        right_index = np.arange(idx, idx + bw)
    # If the energy is below, include that energy in the bin below the k-edge
    else:
        left_index = np.arange(idx + 1 - bw, idx + 1)
        right_index = np.arange(idx + 1, idx + 1 + bw)

    # Partition the spectrum to just the spectrum corresponding to the left bin energies
    left_spectrum = spectrum_phantom[left_index]

    # Partition the spectrum and phantom to just the spectrum and phantom corresponding to the right bin energies
    right_spectrum = spectrum_phantom[right_index]

    # Calculate the total number of photons in each bin
    left_bin = np.sum(left_spectrum)
    right_bin = np.sum(right_spectrum)

    # Get the sum of the photons in the airscan for each bin
    left_bin_I0 = np.sum(spectrum_air[left_index])
    right_bin_I0 = np.sum(spectrum_air[right_index])

    # Normalize to the air spectrum
    left_bin = left_bin/left_bin_I0
    right_bin = right_bin/right_bin_I0

    return left_bin, right_bin


def calc_4_poisson(left_bin, right_bin, L=3):
    """

    :param left_bin: The sum of the photons in the left bin
    :param right_bin: The sum of the phtons in the right bin
    :param L: The total thickness of the phantom
    :return: The four Poisson distribution array as an array (size 4 x k)
    """

    # Get the range to calculate the 4 Poisson distributions
    k = np.arange(1, L+1)

    # Calculate the components of the Poisson distributions (see paper, Energy window optimization for X-Ray K-edge
    # tomographic imaging, Meng et al)
    k_fact = factorial(k)
    k_log = np.log(k)
    k_term = np.divide(k_log, k_fact)
    left_power = np.power(left_bin, k)
    right_power = np.power(right_bin, k)

    left_exp = np.exp(-left_bin)
    right_exp = np.exp(-right_bin)

    left_poisson1 = np.multiply(k_term, left_power) * left_exp
    left_poisson2 = np.multiply(left_poisson1, k_log)
    right_poisson1 = np.multiply(k_term, right_power) * right_exp
    right_poisson2 = np.multiply(right_poisson1, k_log)

    return np.array([left_poisson1,  left_poisson2, right_poisson1, right_poisson2])


def SNR(poisson_values):
    """
    Calculates the SNR based on the equation in Energy window optimization for X-Ray K-edge tomographic imaging, by
    Meng et al
    :param poisson_values: Takes the poisson array from the 'calc_4_poisson' function
    :return: The SNR
    """

    # Grab the four Poisson value arrays
    left_poisson1 = np.asarray(poisson_values[0])
    left_poisson2 = np.asarray(poisson_values[1])
    right_poisson1 = np.asarray(poisson_values[2])
    right_poisson2 = np.asarray(poisson_values[3])

    # Calculate the numerator and denominator for the SNR equation
    numerator = np.sum(np.subtract(left_poisson1, right_poisson1))
    denom1 = np.sum(left_poisson2)
    denom2 = (np.sum(left_poisson1))**2
    denom3 = np.sum(right_poisson2)
    denom4 = (np.sum(right_poisson1))**2

    denominator = np.sqrt(denom1 - denom2 + denom3 - denom4)

    return numerator/denominator


def bin_width(idx, num_energies):
    """
    Get the maximum bin width possible based on the index of the k-edge in our beam spectrum and the number of energies
    in the spectrum. Sets the maximum bin width as 1/5 of the number of beam energies
    :param idx: index of the energy closest to our k-edge
    :param num_energies: the length of the energy spectrum array
    :return: The bin width in terms of number of index positions
    """
    if idx < num_energies/5:
        width = idx
    elif idx > 4/5 * num_energies:
        width = num_energies - idx - 1
    else:
        width = int(np.round(num_energies/4))

    return width


def find_nearest(array, value):
    """
    Find the nearest value to 'value' in the given array and output the values and the index of the value
    :param array: The array to search
    :param value: The value to look for
    :return: The closest number, the index of the number, and True if the number is greater than the value
    """
    array = np.asarray(array)
    difference = np.abs(array - value)  # Get difference of each value to find the smallest difference
    idx = difference.argmin()  # Find index of the smallest distance
    e_width = array[1] - array[0]  # Get the bin width of the energy spectrum

    # Make sure the energy of the k-edge bin is included on the correct side
    if difference[idx] < e_width/2:
        above = True
    else:
        above = False

    return array[idx], idx, above

#%% Directory

directory = 'D:/Research/Bin Optimization/'



#%% 2.0 mm Al spectrum
import matplotlib.lines as mlines

directory1 = directory + '/Npy Attenuation/'

spectrum = np.load(directory + 'Beam Spectrum/corrected-spectrum_120kV.npy')

energies = 1000*spectrum[:, 0]
spectrum = spectrum[:, 1]
fig = plt.figure(figsize=(8, 8))

Al_spectrum = filter_spectrum('Al', 2, spectrum, directory=directory1)
Cu05_spectrum = filter_spectrum('Cu', 0.5, spectrum, directory=directory1)
Cu1_spectrum = filter_spectrum('Cu', 1.0, spectrum, directory=directory1)

#Cu05_spectrum = np.multiply(Cu05_spectrum)
#Cu1_spectrum = np.multiply(Cu1_spectrum, 4.75)

Al_sum = np.sum(Al_spectrum)
Cu05_sum = np.sum(Cu05_spectrum)
Cu1_sum = np.sum(Cu1_spectrum)
print(Al_sum, Cu05_sum, Cu1_sum)

np.save(directory + 'Al2.0_spectrum.npy', Al_spectrum)
np.save(directory + 'Cu0.5_spectrum.npy', Cu05_spectrum)
np.save(directory + 'Cu1.0_spectrum.npy', Cu1_spectrum)

plt.plot(energies, Al_spectrum, ls='--', color='black')
plt.plot(energies, Cu05_spectrum, ls='-', color='black')
plt.plot(energies, Cu1_spectrum, ls=':', color='black')

linepatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='-', label='0.5 mm Cu')
dashpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle='--', label='2.0 mm Al')
dotpatch = mlines.Line2D([0], [0], color='black', lw=2, linestyle=':', label='1.0 mm Cu')

plt.legend(handles=[dashpatch, linepatch, dotpatch], fancybox=True, shadow=False, fontsize=15)

plt.xlabel('Energy (keV)', fontsize=18)
plt.ylabel('Weight', fontsize=18)
plt.xlim([0, 120])
plt.ylim([0, 8.2E-7])
plt.tick_params(labelsize=15)
plt.show()

