import numpy as np
from scipy.ndimage import rotate
from scipy.misc import factorial
import time
import matplotlib.pyplot as plt


def optimize(k_edge, filter_type, filter_thickness, directory='D:/Research/Bin Optimization/Beam Spectrum/'):

    SDD = 30  # Source to object distance
    N = 5500000  # Number of photons to scale the spectrum by

    phantom = np.load('D:/Research/Bin Optimization/Phantoms/Lan_phantom_3P.npy')  # Create the desired phantom
    spectrum = np.load(directory + 'corrected-spectrum_120kV.npy')  # Load the energy spectrum
    energies = spectrum[:, 0]
    spectrum_weights = spectrum[:, 1]  # Relative weights of all energies

    # Filter the spectrum with the desired filtering
    spectrum_filt = filter_spectrum(filter_type, filter_thickness, spectrum_weights)

    # Filter the spectrum due to air from source to phantom
    spectrum_filt = filter_air(SDD, spectrum_filt)

    # Num of photons of each energy that will hit the detector in the absence of the phantom
    spectrum_I0 = np.multiply(spectrum_filt, N)

    # Find the closest energy to k-edge energy in the energy matrix to find the correct index
    near_energy, idx = find_nearest(energies, k_edge)

    # Find largest bin width possible with our spectrum data (50 points with energies, or less)
    width = bin_width(idx, len(energies))

    left_bin_sum = np.zeros(width)  # Keep track of number of photons in the left bin for each bin width
    right_bin_sum = np.zeros(width)  # Keep track of number of photons in the left bin for each bin width
    poisson_values = np.zeros([width, 4])
    snr = np.zeros(width)

    for w in np.arange(width):
        lb, rb = norm_mean_photons(spectrum_I0, idx, w+1, phantom)
        left_bin_sum[w] = lb
        right_bin_sum[w] = rb
        poisson_values[w] = calc_4_poisson(lb, rb, 3)
        snr[w] = SNR(poisson_values[w], 3)

    return snr, width, energies[1]-energies[0]


def initialize_phantom(energies=200):

    # Create an empty phantom within which to put the phantom
    size = 1000
    # Possible center coordinates of the vials, both row and column
    pcc = [int(size/2), int(np.round(0.2*size)), int(np.round(0.35*size)), int(np.round(0.24*size)),
           int(np.round(0.65*size)), int(np.round(0.8*size)), int(np.round(0.76*size))]
    phantom = np.ones([energies, size, size])
    center = (pcc[0], pcc[0])  # Center of the matrix and phantom (the indices)
    phantom_radius = pcc[0]
    hole_radius = int(np.round(phantom_radius/30 * 6))  # Radius of the holes in the phantom
    vial_radius = int(np.round(phantom_radius/30 * 5))  # Radius of the holes minus the thickness of the vial
    vial_centers = [(pcc[1], pcc[0]), (pcc[2], pcc[3]), (pcc[4], pcc[3]), (pcc[5], pcc[0]), (pcc[4], pcc[6]),
                    (pcc[2], pcc[6]), center]

    # Create the masks for the vials contents
    vial_masks = np.zeros([7, size, size], dtype='bool')
    for i in np.arange(7):
        vial_masks[i] = circular_mask(vial_centers[i], vial_radius, [size, size])

    # Get the water vials
    water_masks = np.add(vial_masks[0], vial_masks[6])

    # Create the ring of PLA for the PCR tubes
    tube_mask = np.zeros([size, size], dtype='bool')
    for j in np.arange(7):
        temp = ring_mask(vial_centers[j], vial_radius, hole_radius, [size, size])
        tube_mask = np.add(tube_mask, temp)

    # Create the phantom mask
    phantom_mask = circular_mask(center, phantom_radius, [size, size])

    # Load the attenuation coefficients for air, water, PLA, etc.
    folder = 'D:/Research/Bin Optimization/Npy Attenuation/'
    air = np.load(folder + 'Air.npy')
    water = np.load(folder + 'H2O.npy')
    pla = np.load(folder + 'PLA.npy')
    poly = np.load(folder + 'Polypropylene.npy')

    # Place all the initial attenuation values in the phantom matrix
    for k in np.arange(energies):
        phantom[k] = np.multiply(air[k], phantom[k])  # Fill with appropriate air value for the energy
        np.place(phantom[k], phantom_mask, pla[k])  # Place phantom values at the appropriate locations
        np.place(phantom[k], tube_mask, poly[k])  # Place tube plastic values
        np.place(phantom[k], water_masks, water[k])  # Place water values in the top tube and center tube

    # Fill the remaining vials with the appropraite solutions
    fill_vial(phantom, vial_masks[1], 'Au', 3)
    fill_vial(phantom, vial_masks[2], 'Dy', 3)
    fill_vial(phantom, vial_masks[3], 'Lu', 3)
    fill_vial(phantom, vial_masks[4], 'Gd', 3)
    fill_vial(phantom, vial_masks[5], 'I', 3)

    # Multiple entire phantom by the distance of a single pixel in cm to get the product of linear attenuation and distance
    phantom = np.multiply(phantom, 3/size)

    np.save('D:/Research/Bin Optimization/Phantoms/Lan_phantom_3P_small.npy', phantom)

    return phantom


def circular_mask(center, radius, img_dim):
    """

    :param center:
    :param radius:
    :param img_dim:
    :return:
    """
    # Create meshgrid of values from 0 to img_dim in both dimension
    xx, yy, = np.mgrid[:img_dim[0], :img_dim[1]]

    # Define the equation of the circle that we would like to create
    circle = (xx - center[0])**2 + (yy - center[1])**2

    # Create the mask of the circle
    arr = np.ones(img_dim)
    mask = np.ma.masked_where(circle < radius**2, arr)

    return mask.mask


def ring_mask(center, inner_radius, outer_radius, img_dim):
    """

    :param center:
    :param inner_radius:
    :param outer_radius:
    :param img_dim:
    :return:
    """
    # Create meshgrid of values from 0 to img_dim in both dimension
    xx, yy, = np.mgrid[:img_dim[0], :img_dim[1]]

    # Define the equation of the circle that we would like to create
    circle = (xx - center[0]) ** 2 + (yy - center[1]) ** 2

    # Create the mask of the ring with the inner and outer radii
    ring = np.logical_and(circle < outer_radius**2, circle > inner_radius**2, dtype='bool')

    return ring


def fill_vial(phantom, mask, material, concentration, directory='D:/Research/Bin Optimization/Npy Attenuation/'):
    """

    :param phantom:
    :param mask:
    :param material:
    :param concentration:
    :param directory:
    :return:
    """
    # Load the desired material attenuation file
    file = material + str(concentration) + 'P.npy'
    fill_mat = np.load(directory + file)

    # Fill the vial of each energy matrix with the appropriate attenuation coefficient
    for i in np.arange(len(phantom)):
        np.place(phantom[i], mask, fill_mat[i])

    return phantom


def filter_spectrum(filter_type, filter_thickness, spectrum, directory='D:/Research/Bin Optimization/Npy Attenuation/'):
    """

    :param filter_type:
    :param filter_thickness:
    :param spectrum:
    :param directory:
    :return:
    """

    # Load the filter mass attenuation coefficients
    filter = np.load(directory + filter_type + '.npy')

    # Calculate the spectrum correction according to the exponential term of Beer's Law
    filter = np.multiply(filter, filter_thickness)  # Multiply the attenuation by the thickness of the filter
    filter_corr = np.exp(-filter)

    spectrum_filtered = np.multiply(spectrum, filter_corr)

    return spectrum_filtered


def filter_air(distance, spectrum, directory='D:/Research/Bin Optimization/Npy Attenuation/'):

    air = np.load(directory + 'Air.npy')  # Load the air attenuation coefficients at all energies

    # Calculate the spectrum air correction according to the exponential term of Beer's Law
    air = np.multiply(air, distance)
    air_corr = np.exp(-air)

    spectrum_filtered = np.multiply(spectrum, air_corr)

    return spectrum_filtered


def norm_mean_photons(spectrum, idx, bw, phantom):
    """

    :param spectrum:
    :param idx:
    :param bw: is in terms of number of indices in the energy matrix
    :param phantom:
    :return:
    """
    start = time.time()
    spectrum = np.asarray(spectrum)
    phantom = np.asarray(phantom)

    # Partition the spectrum and phantom to just the spectrum and phantom corresponding to the left bin energies
    left_spectrum = spectrum[idx - bw:idx]
    left_phantom = phantom[idx - bw:idx]

    # Partition the spectrum and phantom to just the spectrum and phantom corresponding to the right bin energies
    right_spectrum = spectrum[idx+1:idx + bw]
    right_phantom = phantom[idx+1:idx + bw]

    # Calculate the total number of photons in each bin
    left_bin = sum_photons_all_proj(left_spectrum, left_phantom)
    right_bin = sum_photons_all_proj(right_spectrum, right_phantom)

    # Normalize to the air spectrum
    left_bin_I0 = np.sum(left_spectrum)
    right_bin_I0 = np.sum(right_spectrum)

    left_bin = left_bin/left_bin_I0
    right_bin = right_bin/right_bin_I0
    end = time.time()
    print(end-start)
    return left_bin, right_bin


def sum_photons_single_proj(spectrum_weights, phantom):
    """
    This function takes a single phantom energy slice (or multiple energies) and calculates the sum of the photons of
    that energy that would make it to the detector for a single projection and outputs the single sum of the number of
    photons (all energies) that hit the detector
    :param spectrum_weights: the spectrum weights of the specific beam energies to look at
    :param phantom: the phantoms corresponding to those energies
    :return:
    """
    spectrum_weights = np.asarray(spectrum_weights, dtype=np.float64)
    phantom = np.asarray(phantom, dtype=np.float64)

    # Check the dimensions of the phantom to see if we're working with 1 or multiple energies
    dimensions = np.array(np.shape(phantom))
    if len(dimensions) == 2:
        attenuation_corr = np.sum(phantom, axis=1)  # Calculate the product of the exponentials in the matrix
        attenuation_corr = np.exp(-attenuation_corr)  # Take the negative exponential of the sum

        photon_sum_each_line = np.multiply(attenuation_corr, spectrum_weights)  # Update the spectrum_weights with their att. correction
        photon_sum = np.sum(photon_sum_each_line)  # Add all the photons from all energies
    else:
        attenuation_corr = np.sum(phantom, axis=2)  # Calculate the product of the exps in multiple energy case
        attenuation_corr = np.exp(-attenuation_corr)  # Take the negative exponential of the sum

        photon_sum_each_line = np.zeros((dimensions[0], dimensions[1]))
        for j in np.arange(dimensions[0]):
            photon_sum_each_line[j] = np.multiply(attenuation_corr[j],
                                                  spectrum_weights[j])  # Update the spectrum_weights with their att. correction

        photon_sum = np.sum(photon_sum_each_line, axis=1)  # Add all the photons from all energies
        photon_sum = np.sum(photon_sum)  # Add all photons (from all energies)

    return int(np.round(photon_sum))


def sum_photons_all_proj(spectrum_weights, phantom):
    """
    This function takes a single phantom energy slice (or multiple energies) and calculates the sum of the photons of
    that energy that would make it to the detector for 180 projections and outputs the single sum of all photon energies
    :param spectrum_weights: the spectrum weights of the specific beam energies to look at
    :param phantom: the phantoms corresponding to those energies
    :return:
    """

    spectrum_weights = np.asarray(spectrum_weights)
    photon_sum = 0

    # Calculate the number of photons that would hit the detector every 2 degrees of rotation
    for ang in np.arange(0, 360, 2):
        curr_phantom = np.array(rotate(phantom, ang, axes=(1, 2), reshape=False))  # Rotate the phantom to the current angle
        curr_photon_sum = sum_photons_single_proj(spectrum_weights, curr_phantom)  # Sum all photons in current projection for all energies
        photon_sum += curr_photon_sum  # Add the current sum to the total sum

    return photon_sum


def calc_4_poisson(left_bin, right_bin, L):
    """

    :param left_bin:
    :param right_bin:
    :param L:
    :return:
    """
    left_bin = np.asarray(left_bin)
    right_bin = np.asarray(right_bin)
    k = np.arange(1, L+1)

    k_fact = factorial(k)
    k_log = np.log(k)
    k_term = np.divide(k_log, k_fact)
    left_power = np.power(left_bin, k)
    right_power = np.power(right_bin, k)

    left_exp = np.exp(-left_bin)
    right_exp = np.exp(-right_bin)

    left_poisson1 = np.multiply(k_term, left_power) * left_exp
    left_poisson2 = np.mutliply(left_poisson1, k_log)
    right_poisson1 = np.multiply(k_term, right_power) * right_exp
    right_poisson2 = np.multiply(right_poisson1, k_log)

    return np.array([left_poisson1,  left_poisson2, right_poisson1, right_poisson2])


def SNR(poisson_values):
    """

    :param poisson_values:
    :return:
    """
    left_poisson1 = np.asarray(poisson_values[0])
    left_poisson2 = np.asarray(poisson_values[1])
    right_poisson1 = np.asarray(poisson_values[2])
    right_poisson2 = np.asarray(poisson_values[3])

    numerator = np.sum(np.subtract(left_poisson1, right_poisson1))
    denom1 = np.sum(left_poisson2)
    denom2 = (np.sum(left_poisson1))**2
    denom3 = np.sum(right_poisson2)
    denom4 = (np.sum(right_poisson1))**2

    denominator = np.sqrt(denom1 - denom2 + denom3 - denom4)

    return numerator/denominator


def bin_width(idx, num_energies):
    """

    :param idx: index of the energy closest to our k-edge
    :param num_energies: the length of the energy spectrum array
    :return:
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

    :param array:
    :param value:
    :return:
    """
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx