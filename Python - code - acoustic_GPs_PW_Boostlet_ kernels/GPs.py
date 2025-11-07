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

Module : GPs
Functions: predict, split_covariance_in_blocks, stack_block_covariance, 
complex_covariance_from_real
"""

import numpy as np
import Kernels as kernel

def predict(y, tx, txs, kernel_names, Sigma, axis, delta, *args, sample=False):
    """
    Prediction function for sound field reconstruction.

    Parameters
    ----------
    y : numpy.ndarray
        Observed data.
    tx : numpy.ndarray
        Input data.
    txs : numpy.ndarray
        Input data for prediction.
    kernel_names : list
        Names of the kernels to use.
    Sigma : numpy.ndarray
        Noise covariance matrix.
    axis : int
        Axis for computation.
    delta : float
        Small value for numerical stability.
    *args : tuple
        Additional arguments for the kernel functions.
    sample : bool, optional
        Flag to indicate if sampling is required (default is False).

    Returns
    -------
    y_mean : numpy.ndarray
        Mean prediction.
    y_cov : numpy.ndarray
        Covariance of the prediction.
    y_samples : numpy.ndarray
        Samples if sampling is required, otherwise empty.
    """
    
    print('Progress in predict: Start bivariates')
       
    k_uu = getattr(kernel, kernel_names[0])
    k_vv = getattr(kernel, kernel_names[1])
    k_uv = getattr(kernel, kernel_names[2])
    k_vu = getattr(kernel, kernel_names[3])

    print('Progress in predict: Bivariates: K_zz')
    # tx-tx
    K_xx_uu = k_uu(tx, tx, *args)
    K_xx_vv = k_vv(tx, tx, *args)
    K_xx_uv = k_uv(tx, tx, *args)
    K_xx_vu = k_vu(tx, tx, *args)

    # Full bivariate
    K_zz = stack_block_covariance(K_xx_uu, K_xx_uv, K_xx_vu, K_xx_vv)

    print('Progress in predict: Bivariates: K_zsz')
    # txs-tx
    K_xsx_uu = k_uu(txs, tx, *args)
    K_xsx_vv = k_vv(txs, tx, *args)
    K_xsx_uv = k_uv(txs, tx, *args)
    K_xsx_vu = k_vu(txs, tx, *args)

    # Full bivariate
    K_zsz = stack_block_covariance(K_xsx_uu, K_xsx_uv, K_xsx_vu, K_xsx_vv)

    print('Progress in predict: Bivariates: K_zszs')
    # txs-txs
    K_xsxs_uu = k_uu(txs, txs, *args)
    K_xsxs_vv = k_vv(txs, txs, *args)
    K_xsxs_uv = k_uv(txs, txs, *args)
    K_xsxs_vu = k_vu(txs, txs, *args)

    # Full bivariate
    K_zszs = stack_block_covariance(K_xsxs_uu, K_xsxs_uv, K_xsxs_vu, K_xsxs_vv)

    print('Progress in predict: End bivariates')
    
    print('Progress in predict: Start bivariate_mean')

    inv_Kzz_Sigma = np.linalg.inv(K_zz + Sigma)  # Shape: (j, j)
    temp = np.matmul(inv_Kzz_Sigma, y) 
    if temp.ndim == 1:
        temp = temp[:, np.newaxis]  
    temp2 = np.matmul(K_zsz, temp)  
    y_mean=temp2
    
    print('Progress in predict: End bivariate_mean')
    
    print('Progress in predict: Start bivariate_cov')
    
    X = np.linalg.solve(K_zz + Sigma, K_zsz)
    X_T = X.T
    y_cov = K_zszs - np.matmul(K_zsz, X_T)
    
    print('Progress in predict: End bivariate_cov')

    if sample:
        delta_ = delta
        cholesky = np.zeros_like(y_cov)
        for i in range(y_mean.shape[0]):
            while True:
                try:
                    cholesky[i, :, :] = np.linalg.cholesky(y_cov[i, :, :])
                    break
                except np.linalg.LinAlgError:
                    y_cov[i, :, :] = y_cov[i, :, :] + delta_ * np.eye(y_mean.shape[1])
                    delta_ = delta_ * 10
            delta_ = delta
        y_samples = y_mean + np.einsum('ijk,kl->ijl', cholesky, np.random.randn(*y_mean.shape))
    else:
        y_samples = np.array([])

    return y_mean, y_cov, y_samples


def split_covariance_in_blocks(K):
    """
    Split covariance matrix into four blocks.
    
    Parameters:
    - K: a 2D NumPy array (assumed square or even-sized in both dimensions)
    
    Returns:
    - K_top_left, K_top_right, K_bottom_left, K_bottom_right: submatrices
    """
    N = K.shape[0] // 2
    M = K.shape[1] // 2

    K_top_left = K[:N, :M]
    K_top_right = K[:N, M:]
    K_bottom_left = K[N:, :M]
    K_bottom_right = K[N:, M:]

    return K_top_left, K_top_right, K_bottom_left, K_bottom_right


def stack_block_covariance(Krr, Kri, Kir, Kii):
    """
    Stack block covariance matrices into a single matrix.
    
    Parameters:
    - Krr, Kri, Kir, Kii: 2D NumPy arrays
    
    Returns:
    - K: stacked covariance matrix
    """
    # Concatenate Krr and Kri along the second dimension (columns)
    top_half = np.concatenate((Krr, Kri), axis=1)
    
    # Concatenate Kir and Kii along the second dimension (columns)
    bottom_half = np.concatenate((Kir, Kii), axis=1)
    
    # Concatenate the top_half and bottom_half along the first dimension (rows)
    K = np.concatenate((top_half, bottom_half), axis=0)
    
    return K

def complex_covariance_from_real(Krr, Kii, Kri):
    """
    Compute complex covariance from real parts.

    Parameters:
    Krr (numpy.ndarray): Real part of the covariance matrix.
    Kii (numpy.ndarray): Imaginary part of the covariance matrix.
    Kri (numpy.ndarray): Cross part of the covariance matrix.

    Returns:
    tuple: A tuple containing the complex covariance matrices (K, Kp).
    """
    # Compute complex covariance matrices
    K = Krr + Kii + 1j * (Kri.T - Kri)
    Kp = Krr - Kii + 1j * (Kri.T + Kri)

    return K, Kp
