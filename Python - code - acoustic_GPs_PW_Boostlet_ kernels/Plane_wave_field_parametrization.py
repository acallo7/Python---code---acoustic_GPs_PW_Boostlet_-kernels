# -*- coding: utf-8 -*-
"""
SD2101 Individual Project Work on Sound and Vibrationer (2024- September-December) 
 / MWL internship (2025- June-August)
Title : "Derivation and implementation of spatio-temporal plane waves 
and boostlet based kernels for sound field reconstruction with Gaussian processes"
@author : Awen Callo

Based on Diego Cavedies-Nozal's code 
in "Gaussian processes for sound field reconstruction" 
published in 'The Journal of Acoustical Society of America', February 11, 2021.

Module : Plane_wave_field_parametrization
Functions: plane_wave_field, extract_datah5, random_locations, random_locations2,
get_sparse_microphones
"""

import numpy as np
import h5py

def plane_wave_field(tx, txs, n_reps, snr, n_waves, f, freq_samp, c, data):
    """
    Generate a plane wave field with noise, true sound field, and measured field.
    
    Parameters:
    - tx: Array of transmitter locations
    - txs: Array of receiver locations
    - n_reps: Number of repetitions for noise
    - snr: Signal-to-noise ratio (in dB)
    - n_waves: Number of waves
    - f: Frequency of the waves
    - freq_samp: sampling frequency
    - c: Speed of sound
    - data: measured data
    
    Returns:
    - p_clean: Clean sound field (without noise)
    - p: Sound field with noise
    - setup: Dictionary containing setup informations
    """
    
    ome_max = np.pi * freq_samp

    ome = []
    k = []
    for i in range(len(f)):
        omei = 2*np.pi*f[i]
        ome.append(omei)
        ki = 2 * np.pi * f[i] / c
        k.append(ki)   
    
    xi = np.zeros((len(ome), 2))
    theta_boostlet = np.zeros((len(k), len(ome)))
    a_dilation = np.zeros((len(k), len(ome)))

    cn = np.zeros((len(ome),len(k)))
    wave_directions = np.zeros((len(data[:,0]), len(data[0,:]),len(ome), len(k)))
    for i in range(len(data[:,0])):
        for j in range(len(data[0,:])):
            for n in range(len(ome)): 
                for m in range(len(k)): 
                    if np.abs(ome[n]) < np.abs(k[m]): 
                        thetaij = np.sinh((-ome_max)/(k[m]))**(-1)
                        theta_boostlet[m,n] = np.arctanh(ome[n] / k[m]) 
                        a_dilation[m,n] = np.sqrt(k[m]**2 - ome[n]**2)
                    elif np.abs(ome[n]) > np.abs(k[m]): 
                        thetaij = np.cosh((ome_max)/(ome[n]))**(-1)
                        theta_boostlet[m,n] = np.arctanh(k[m]/ ome[n]) 
                        a_dilation[m,n] = np.sqrt(ome[n]**2- k[m]**2)
                    elif np.abs(ome[n]) == np.abs(k[m]):
                        thetaij = 0
                        theta_boostlet[m,n] = 0
                        a_dilation[m,n] = 0
                    xi[m, 0] = k[m]
                    xi[n, 1] = ome[n]
                    cn_ijl = c * np.tanh(thetaij)
                    cn[n,m] = cn_ijl
                    wave_directions[i,j,n,m] = np.arctanh(cn_ijl / c)  
                    
    # a_i = 2**(1 / np.linspace(1, len(a_dilation[:, 0]), len(a_dilation[:, 0])))
    # sigma_ai = np.abs(a_i[:-1] - a_i[1:])
    # mu_ai = (a_i[:-1] + a_i[1:]) / 2
    
    # sigma_ai = np.append(sigma_ai, 2*sigma_ai[-1] - sigma_ai[-2])
    # mu_ai = np.append(mu_ai, 2*mu_ai[-1] - mu_ai[-2])
    
    # theta_i = np.linspace(-2, 2, len(theta_boostlet[:,0]))
    # sigma_thetai = np.abs(theta_i[:-1] - theta_i[1:])
    # mu_thetai = (theta_i[:-1] + theta_i[1:]) / 2
    
    # sigma_thetai = np.append(sigma_thetai, 2*sigma_thetai[-1] - sigma_thetai[-2])
    # mu_thetai = np.append(mu_thetai, 2*mu_thetai[-1] - mu_thetai[-2])
    
    a_i = 2**(-np.linspace(0,len(a_dilation[:, 0]),len(a_dilation[:, 0])))
    mu_ai = a_i
    sigma_ai = np.abs(a_i[:-1] - a_i[1:])
    sigma_ai = np.append(sigma_ai, 2*sigma_ai[-1] - sigma_ai[-2])

    theta_i = np.zeros((len(a_i),len(theta_boostlet[:,0])))
    sigma_thetai = np.zeros((len(a_i),len(theta_boostlet[:,0])))
    mu_thetai = np.zeros((len(a_i),len(theta_boostlet[:,0])))
    for i in range(len(a_i)):
        theta_ij = np.linspace(-(ome_max/a_i[i]), (ome_max/a_i[i]),len(theta_boostlet[:,0]))
        theta_i[i,:] = theta_ij
        sigma_thetaij = np.abs(theta_ij[:-1] - theta_ij[1:])
        sigma_thetaij = np.append(sigma_thetaij, 2*sigma_thetaij[-1] - sigma_thetaij[-2])
        sigma_thetai[i,:] = sigma_thetaij
        mu_thetaij = (theta_ij[:-1] + theta_ij[1:]) / 2
        mu_thetaij = np.append(mu_thetaij, 2*mu_thetaij[-1] - mu_thetaij[-2])
        mu_thetai[i,:] = mu_thetaij
    
    setup = {
        'xi': xi,
        'ome' : ome,
        'k' : k,
        'cn' : cn,
        'wave_directions': wave_directions,
        'theta_boostlet': theta_boostlet,
        'a_dilation': a_dilation,
        'a_i':a_i,
        'sigma_ai':sigma_ai,
        'mu_ai':mu_ai,
        'theta_i':theta_i,
        'sigma_thetai':sigma_thetai,
        'mu_thetai':mu_thetai
    }
    
    p_clean = data
    norm = np.mean(np.abs(p_clean), axis=1)
    p_clean2 = p_clean / norm[:, None]  # Broadcasting for normalization
    p_mean = np.mean(np.abs(p_clean2), axis=1)
    noise_std = p_mean / (20**(snr/20))
    p = p_clean
    
    # Setup dictionary
    setup['noise_std'] = noise_std
    setup['norm'] = norm
    setup['tx'] = tx
    setup['txs'] = txs
    setup['c'] = c
    
    return p_clean, p, setup

def extract_datah5():

    with h5py.File('rir_NBI_line.h5', 'r') as file:
        print("Keys: %s" % list(file.keys()))
        
        line = str(input("Choose the file from 'Keys': "))
        
        dataset = file['{}'.format(line)]
        data = dataset['/{}'.format(line)]
        print("Data: %s" % data)
        
        # Access impulse_response
        member = data['impulse_response']
        if isinstance(member, h5py.Dataset):
        # Extract the data
            impulse_response = member[()]
            print('Impulse_response: {}'.format(impulse_response.shape))
            
        
        # Access noise
        member = data['noise']
        if isinstance(member, h5py.Dataset):
        # Extract the data
            noise = member[()]
            print('Noise: {}'.format(noise.shape))
            
        # Access posNoise
        member = data['posNoise']
        if isinstance(member, h5py.Dataset):
        # Extract the data
            posNoise = member[()]
            print('posNoise: {}'.format(posNoise.shape))
            
        # Access posRIR
        member = data['posRIR']
        if isinstance(member, h5py.Dataset):
        # Extract the data
            posRIR = member[()]
            print('posRIR: {}'.format(posRIR.shape))
            
        return impulse_response, noise, posNoise, posRIR

def random_locations(n_mics_zone, n_rows):
    # Determine random locations
    x_first = 0
    x_last = n_rows
    i_mics = get_sparse_microphones(n_mics_zone, n_rows, x_first, x_last)
    return sorted(i_mics)

def random_locations2(n_mics_zone, n_rows):
    # Determine random locations
    x_first = 0
    x_last = n_rows
    i_mics = np.array([], dtype=int)
    grid = x_last - x_first
    j = 1
    
    for i in range(grid):
        if i % 2 == 0 and j < n_mics_zone:
            # Calculate the value to be added
            value = np.floor(i) + x_first
            # Append the value to the i_mics array
            i_mics = np.append(i_mics, value.astype(int))
            j += 1

    # Ensure we have exactly n_mics_zone elements
    i_mics = i_mics[:n_mics_zone]
    
    # Randomly permute the elements
    i_mics = np.random.permutation(i_mics)
    
    return sorted(i_mics)

def get_sparse_microphones(n_mics, n_rows, x_first, x_last):
    """
    Use Latin Hypercube to randomize microphone positions.

    Parameters
    ----------
    n_mics : int
        Number of microphones.
    n_rows : int
        Number of rows in the grid.
    x_first : int
        First index of the grid.
    x_last : int
        Last index of the grid.

    Returns
    -------
    i_mics : numpy.ndarray
        Array of randomized microphone positions.
    """
    i_mics = np.array([], dtype=int)
    grid = x_last - x_first

    if n_mics > 0:
        while len(np.unique(i_mics)) < n_mics:
            i_mics = np.floor(np.random.rand(n_mics) * grid) + x_first
            i_mics = i_mics.astype(int)

        i_mics = np.random.permutation(i_mics)[:n_mics]

    return i_mics

def I_mics(n_mics, data, field_grid_rows, random='no'):
    
    if random =='yes':
        i_mics = random_locations(n_mics, field_grid_rows)
    elif random =='no':
        if n_mics <= 100:
            step = len(data[:,0]) // (n_mics-1)
            i_mics = [i * step for i in range(n_mics)] 
        elif n_mics ==134:   # 134 locations (2/3)
            i_mics = [0, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18, 20, 21, 23, 24, 26, 27, 29,
            30, 32, 33, 35, 36, 38, 39, 41, 42, 44, 45, 47, 48, 50, 51, 53, 54, 56,
            57, 59, 60, 62, 63, 65, 66, 68, 69, 71, 72, 74, 75, 77, 78, 80, 81, 83,
            84, 86, 87, 89, 90, 92, 93, 95, 96, 98, 99, 101, 102, 104, 105, 107, 108,
            110, 111, 113, 114, 116, 117, 119, 120, 122, 123, 125, 126, 128, 129, 131,
            132, 134, 135, 137, 138, 140, 141, 143, 144, 146, 147, 149, 150, 152, 153,
            155, 156, 158, 159, 161, 162, 164, 165, 167, 168, 170, 171, 173, 174, 176,
            177, 179, 180, 182, 183, 185, 186, 188, 189, 191, 192, 194, 195, 197, 198, 200]
        elif n_mics ==150:      #150 locations (3/4)
            i_mics = [0, 2, 4, 6, 8, 10, 13, 15, 17, 19, 21, 23, 25, 27, 30, 32, 34, 36, 38, 40,
            43, 45, 47, 49, 51, 53, 55, 58, 60, 62, 64, 66, 68, 70, 73, 75, 77, 79, 81, 83,
            85, 87, 89, 91, 94, 96, 98, 100, 102, 104, 107, 109, 111, 113, 115, 117, 119, 122,
            124, 126, 128, 130, 132, 134, 137, 139, 141, 143, 145, 147, 149, 151, 153, 155, 158,
            160, 162, 164, 166, 168, 170, 172, 175, 177, 179, 181, 183, 185, 187, 189, 191, 193,
            195, 197, 199, 200, 3, 7, 11, 14, 18, 22, 26, 29, 33, 35, 39, 42, 46, 50, 54, 57, 61,
            65, 69, 72, 76, 80, 84, 88, 92, 95, 99, 103, 106, 110, 114, 118, 121, 125, 129, 133,
            138, 142, 146, 150, 154, 159, 163, 167, 171, 176, 180, 184, 188, 192, 196, 198]     
        elif n_mics ==181:       #181 locations (9/10)
            i_mics = [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 66, 67, 69, 70, 71, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
            97, 98, 99, 100, 101, 102, 104, 105, 106, 107, 109, 110, 111, 112,
            113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
            128, 130, 131, 132, 133, 134, 135, 137, 138, 139, 140, 141, 142,
            143, 144, 145, 146, 147, 149, 150, 151, 152, 153, 154, 155, 156, 157,
            158, 159, 160, 161, 162, 163, 165, 166, 168, 170, 171, 172,
            173, 174, 175, 176, 177, 178, 179, 180, 182, 183, 184, 185, 186, 187,
            188, 189, 191, 192, 194, 195, 196, 198, 199, 200] 
        elif n_mics ==201:      # 201 locations (all)
            i_mics = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
            78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
            97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112,
            113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
            128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
            143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157,
            158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172,
            173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200] 
            
    return i_mics