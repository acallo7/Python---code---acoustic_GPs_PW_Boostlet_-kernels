# -*- coding: utf-8 -*-
"""
SD2101 Individual Project Work on Sound and Vibrationer (2024- October-December) 
 / MWL internship (2025- June-August)
Title : "Derivation and implementation of spatio-temporal plane waves 
and boostlet based kernels for sound field reconstruction with Gaussian processes"
@author : Awen Callo

Based on Diego Cavedies-Nozal's code 
in "Gaussian processes for sound field reconstruction" 
published in 'The Journal of Acoustical Society of America', February 11, 2021.

Module : example
Function: example
"""

import numpy as np
import Plane_wave_field_parametrization as pwfp
import model_sampling as ms
import GPs as gps
import Reconstruction as r
#import scipy as sp

def example():
    print('Progress: Start program')
    
    print('Progress: Start extract data')
    impulse_response, noise, posNoise, posRIR = pwfp.extract_datah5()
    print('Progress: End extract data')
    
    rep = 1 # number of repetition of the spatial scale (1 to have a square matrix)
    start = int(input('Start time discretisation : '))
    end = start + impulse_response.shape[0]*rep
    
    impulse_response = impulse_response[:, start:end]   # reshape data to have square matrix

    montecarlo_chains = 2
    samples_per_chain = 200
    warmup_samples_per_chain = 100
    L = 64  # number of basis kernels (for anisotropic kernels)
    N = int(input('Number of observation points (max. 201) : '))  # number of observations
    freq_step = int(input('Frequency step (Hz) : ')) # for range [100, 20000] : .., 100, 199, 398, 796, 1000, 2000, 4000, ..
    f = np.arange(100, 20000, freq_step) # freq_min=100Hz to have a convenient minimum frequency to use hyperbolic functions
    freq_samp = 48000
    rt = 1  # time resolution
    c = 343  # Sound speed in the fluid
    number_of_waves = 1  # number of plane waves that conform the field
    number_meas_reps = 1  # number of measurement repetitions
    field_grid_rows = (impulse_response.shape[0])  # number of measurement on x-axis
    field_grid_time = int((impulse_response.shape[1]) / rt)  # observation time divided by time resolution
    
    tmin = (1/freq_samp) * start
    tmax = tmin + (1/freq_samp) * impulse_response.shape[1]

    dt = np.linspace(tmin, tmax, field_grid_time)
    dx = np.linspace(0, 5.04, field_grid_rows)
    all_timelocations = np.zeros((len(dt),2))
    for i in range(len(dx)):
        for j in range(len(dt)):
            all_timelocations[i,0] = dx[i]
            all_timelocations[j,1] = dt[j]
              
    print('Progress: Start random_locations')
    
    random = input('Random location: (yes, no)')
    i_measured = pwfp.I_mics(N, impulse_response, field_grid_rows, random)
    
    print(i_measured)
    
    print('Progress: End random_locations')
    
    measured_locations = np.zeros_like(all_timelocations)
    measured_locations_stan = np.zeros_like(all_timelocations[0:N,:])

    for i in range(field_grid_time):
        for j in range(len(i_measured)):
            measured_locations_stan[j,0] = all_timelocations[i_measured[j],0]
            measured_locations_stan[j,1] = all_timelocations[j,1]
            
    for j in range(len(i_measured)):
        measured_locations[i_measured[j],0] = all_timelocations[i_measured[j], 0]
        
    measured_locations[:,1] = all_timelocations[:,1]
            
    # Generate plane wave field
    print('Progress: Start plane_wave_field')
    true_sound_field, noisy_sound_field, sound_field_parameters = pwfp.plane_wave_field(
        all_timelocations, measured_locations, number_meas_reps, 20, number_of_waves, f, freq_samp, c, impulse_response)
    print('Progress: End plane_wave_field')

    measured_sound_field = np.zeros((len(noisy_sound_field[:,0]), len(noisy_sound_field[0,:])) )
    measured_sound_field_stan = np.zeros((len(i_measured), len(i_measured)))

    for i in range(len(i_measured)):
        for j in range(len(i_measured)):
            measured_sound_field_stan[j, i] = noisy_sound_field[i_measured[j], i]
    for j in range(len(i_measured)):
        measured_sound_field[i_measured[j], :] = noisy_sound_field[i_measured[j], :]

    print('Progress: Start compile_model')
    # Choose kernel
    Kernel = str(input('Kernel to use (0 : plane_wave_tx, 1 : boostlet_kernel) : '))
    if Kernel == '0':
        kernel = 'plane_wave_tx'
    elif Kernel == '1':
        kernel = 'boostlet_kernel'
    print('Kernel used: {}'.format(kernel))

    # Compile Stan model if needed
    compile = True
    if compile:
        ms.compile_model(kernel)
    print('Progress: End compile_model')
    
    direction = sound_field_parameters['wave_directions']
    directions_stan = np.zeros((measured_locations_stan.shape[0],measured_locations_stan.shape[0],len(sound_field_parameters['ome']),len(sound_field_parameters['k'])))
    
    for i in range(measured_locations_stan.shape[0]):
        for j in range(measured_locations_stan.shape[0]):
            for n in range(len(sound_field_parameters['ome'])):
                for m in range(len(sound_field_parameters['k'])):
                    directions_stan[i,j,n,m] = direction[i,j,n,m]

    data = dict(
        y = measured_locations_stan[:,0].reshape(-1,1),
        t = measured_locations_stan[:,1].reshape(-1,1),
        N_meas = measured_locations_stan.shape[0],
        N_time = measured_locations_stan.shape[0],
        N_reps = number_meas_reps,
        N_f = len(sound_field_parameters['xi'][:,1]),
        h_stan = np.concatenate(
                        (measured_sound_field_stan.real,
                         measured_sound_field_stan.imag),
            axis=-1),
        h = np.vstack((np.real(measured_sound_field), np.imag(measured_sound_field))),
        cn = sound_field_parameters['cn'],
        k = sound_field_parameters['k'],
        ome = sound_field_parameters['ome'],
        directions = sound_field_parameters['wave_directions'],
        delta = 1e-10,
        Sigma_stan = (sound_field_parameters['noise_std'][0] ** 2) / 2 * np.eye(2 * measured_locations_stan.shape[0]),
        Sigma = (sound_field_parameters['noise_std'][0] ** 2) / 2 * np.eye(2 * measured_locations.shape[0]),
        N_a = len(sound_field_parameters['a_dilation'][:, 0]),
        mu_a = np.zeros((len(sound_field_parameters['a_dilation'][:, 0]))),
        mu_theta = np.zeros((len(sound_field_parameters['a_dilation'][:, 0]))),
        sigma_a = np.zeros((len(sound_field_parameters['a_dilation'][:, 0]))),
        sigma_theta = np.zeros((len(sound_field_parameters['a_dilation'][:, 0])))
        )
    
    # Hyperparameters from prior distributions
    print('Progress: Start init_model')
    bivariate_kernels, kernel_param_names, prior_params, stan_pars = ms.init_model(kernel, L)
    data.update(prior_params)
    print('Progress: End init_model')

    # Monte Carlo sampling
    print('Progress: Start mc_sampling')
    # posterior_samples, posterior_summary = ms.mc_sampling(
        # data, kernel, stan_pars, samples_per_chain, montecarlo_chains, warmup_samples_per_chain)
    print('Progress: End mc_sampling')

    # Reconstruction: For this example, the median of the inferences is used
    kernel_params = {}
    kernel_params['N_samples'] = 1
    # for i in range(len(kernel_param_names)):
        # kernel_params[kernel_param_names[i]] = np.median(posterior_samples[kernel_param_names[i]], axis=0)
    kernel_params['xi'] = sound_field_parameters['xi']
    kernel_params['theta_boostlet'] = sound_field_parameters['theta_boostlet']
    kernel_params['a_dilation'] = sound_field_parameters['a_dilation']
    
    # Specific parameters for the boostlet kernel
    kernel_params['sigma_w'] = 0.02 # to use when the extraction from the data is not done, the value is in the order for 100 samp pts.
    if kernel=='plane_wave_tx':
        print('sigma_w : {}'.format(kernel_params['sigma_w']))
    
    # Specific parameters for the boostlet kernel
    # the parameters defined below are totally random and will be further determined previously from the data (Stan code)
    kernel_params['a_i'] = sound_field_parameters['a_i']
    kernel_params['sigma_ai'] = sound_field_parameters['sigma_ai']
    kernel_params['mu_ai'] = sound_field_parameters['mu_ai']
    kernel_params['theta_i'] = sound_field_parameters['theta_i']
    kernel_params['sigma_thetai'] = sound_field_parameters['sigma_thetai']
    kernel_params['mu_thetai'] = sound_field_parameters['mu_thetai']
    if kernel=='boostlet_kernel':
        print('theta_i : {}'.format(kernel_params['theta_i']))
        print('mu_thetai : {}'.format(kernel_params['mu_thetai']))
        print('sigma_thetai : {}'.format(kernel_params['sigma_thetai']))
        print('a_i : {}'.format(kernel_params['a_i']))
        print('mu_ai : {}'.format(kernel_params['mu_ai']))
        print('sigma_ai : {}'.format(kernel_params['sigma_ai']))
    
    # Predict
    print('Progress: Start predict')
    bivariate_predicted_mean, bivariate_predicted_covariance, _ = gps.predict(
        data['h'], measured_locations, all_timelocations, bivariate_kernels,
        data['Sigma'], -1, 1e-8, kernel_params)
    print('Progress: End predict')
    
    # Get the number of time locations
    num_timelocations = all_timelocations.shape[0]

    # Extract the real and imaginary parts
    real_part = bivariate_predicted_mean[:num_timelocations, :]
    imaginary_part = bivariate_predicted_mean[num_timelocations:, :]

    # Combine them to form the complex predicted_mean
    predicted_mean = real_part + 1j * imaginary_part

    Krr, Kri, Kir, Kii = gps.split_covariance_in_blocks(bivariate_predicted_covariance)
    predicted_covariance = Krr + Kii + 1j * (Kir - Kri)
    predicted_variance = np.abs(np.diag(predicted_covariance[0, :]))
    predicted_sound_field = np.real(predicted_mean)

    # Show reconstruction
    print('Progress: Start plot_reconstruction')
    r.plot_reconstruction(all_timelocations, measured_locations, 
                        all_timelocations, measured_locations,
                        true_sound_field,
                        measured_sound_field,
                        predicted_sound_field,
                        np.abs(predicted_variance),
                        field_grid_rows,
                        field_grid_time,
                        i_measured)
    print('Progress: End plot_reconstruction')

    print('Progress: End program')
    
if __name__ == '__main__':
    example()