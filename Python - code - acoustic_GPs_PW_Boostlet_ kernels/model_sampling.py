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

Module : model_sampling
Functions: compile_model, init_model, mc_sampling
"""

import numpy as np
import cmdstanpy
import pickle
import os

BASE_DIRS = os.path.dirname(__file__)
STAN_MODELS = os.path.join(BASE_DIRS + os.sep, 'stan_models')
COMPILED_STAN_MODELS = os.path.join(STAN_MODELS + os.sep, 'compiled')

def compile_model(model_name, model_path=STAN_MODELS, compiled_save_path=COMPILED_STAN_MODELS):
    # Compile Stan model
    stan_file = os.path.join(model_path, f"{model_name}.stan")

    # Stan compilation
    model = cmdstanpy.CmdStanModel(stan_file=stan_file)

    # Save model
    import pickle
    with open(os.path.join(compiled_save_path, f"{model_name}.pkl"), 'wb') as file:
        pickle.dump(model, file)
        

def init_model(kernel, n_basis_functions):
    """
    Initialize model parameters.

    Parameters
    ----------
    t : array_like
        Time points.
    x : array_like
        Spatial points.
    kernel : str
        Type of kernel to use.
    n_basis_functions : int or array_like
        Number of basis functions.

    Returns
    -------
    kernel_names : list
        List of kernel names.
    kernel_params : list
        List of kernel parameters.
    prior_params : dict
        Dictionary of prior parameters.
    stan_params : list
        List of Stan parameters.
    """
    stan_params = []
    kernel_names = []
    kernel_params = []
    prior_params = {}

    if kernel == 'plane_wave_tx':
        stan_params = ["sigma_w", "b_log"]
        kernel_names = ["cosine", "cosine", "sine", "sine_neg"]
        kernel_params = ["sigma_w"]

        # PLANE WAVE
        angle_range = 2 * np.pi
        
        delta_angle = angle_range / n_basis_functions
        #angles = np.linspace(0, angle_range - delta_angle, n_basis_functions)
        angles = np.arange(0, angle_range-delta_angle, n_basis_functions)
        possible_directions = np.concatenate((np.cos(angles[:, None]), np.sin(angles[:, None])), axis=-1)

        prior_params = {
            'D': possible_directions.shape[0],
            'wave_directions': possible_directions,
            'a': 1,
            'b_log_mean': 2,
            'b_log_std': 1
        }

    elif kernel == 'boostlet_kernel':
        stan_params = ["sigma_a", "sigma_theta", "mu_a", "mu_theta", "b_log"]
        kernel_names = ["cosine_boostlet", "cosine_boostlet", "sine_boostlet", "sine_neg_boostlet"]
        kernel_params = ["sigma_a", "sigma_theta", "mu_a", "mu_theta"]

        # PLANE WAVE
        angle_range = 2 * np.pi
        
        delta_angle = angle_range / n_basis_functions
        #angles = np.linspace(0, angle_range - delta_angle, n_basis_functions)
        angles = np.arange(0, angle_range-delta_angle, n_basis_functions)
        possible_directions = np.concatenate((np.cos(angles[:, None]), np.sin(angles[:, None])), axis=-1)

        prior_params = {
            'D': possible_directions.shape[0],
            'wave_directions': possible_directions,
            'a': 1,
            'b_log_mean': 2,
            'b_log_std': 1
        }

    return kernel_names, kernel_params, prior_params, stan_params
        
BASE_DIRS = os.path.dirname(__file__)
STAN_MODELS = os.path.join(BASE_DIRS + os.sep, 'stan_models')
COMPILED_STAN_MODELS = os.path.join(STAN_MODELS + os.sep, 'compiled')

def mc_sampling(
        data,
        kernel=['plane_wave_tx'],
        model_path=COMPILED_STAN_MODELS,
        n_samples=300,
        chains=3,
        warmup_samples=150,
        pars=['alpha']
):
    compiled_model_pkl = os.path.join(COMPILED_STAN_MODELS + os.sep, kernel + '.pkl')
    try:
        _file = open(compiled_model_pkl, "rb")
    except FileNotFoundError:
        print('Please ensure that the Stan model was compiled and the Pickle file exists.')
        return
    else:
        model = pickle.load(_file)

    posterior_ = model.sample(
        data=data,
        iter_sampling=n_samples,
        iter_warmup=warmup_samples,
        #iter=n_samples,
        #warmup=warmup_samples,
        chains=chains,
        show_console=True
        #variables=pars
        #pars=pars
    )
    #posterior_samples = posterior_.draws_pd()
    #posterior_summary = posterior_.summary()
    #posterior_samples = posterior_.extract(pars=pars, permuted=True)
    #posterior_summary = posterior_.summary(pars=pars)
    
    pars = ['sigma_w', 'b_log']

    # Extract each variable individually
    posterior_samples = {par: posterior_.stan_variable(par) for par in pars}
    
    # Combine the extracted variables into a DataFrame
    #posterior_samples_df = pd.DataFrame(posterior_samples)
    
    # Generate posterior summary
    posterior_summary = posterior_.summary()
    
    return posterior_samples, posterior_summary
        
    