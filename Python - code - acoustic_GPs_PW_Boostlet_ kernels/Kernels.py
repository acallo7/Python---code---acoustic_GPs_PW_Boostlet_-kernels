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

Module : Kernels
Functions: cosine, sine, sine_neg, cosine_boostlet, sine_boostlet, sine_neg_boostlet
"""

import numpy as np

"""
Space-time plane waves kernel
@author: Awen Callo
"""

def cosine(y1, y2, params):
    """
    Definition plane waves kernel in summation of cosine.

    Parameters
    ----------
    y1 : numpy.ndarray
        Array of shape (N, 2) representing [T_temps1, P_positionx1].
    y2 : numpy.ndarray
        Array of shape (M, 2) representing [T_temps2, P_positionx2].
    params : dict
        Dictionary containing parameters:
        - xi: numpy.ndarray
            Array of shape (L, 2) representing [k, omega]
            Wavenumber-frequency vector.
        - theta: numpy.ndarray
            Array of shape (M, N)
            Hyperbolic rotation.
        - a: numpy.ndarray
            Array of shape (M, N)
            Dilation parameter.
        - sigma_w: float
            Variance.

    Returns
    -------
    K : numpy.ndarray
        Resulting kernel matrix.
    """
    
    xi = params['xi'] 
    theta = params['theta_boostlet']
    a = params['a_dilation'] 
    sigma_w = params['sigma_w']

    # Compute X and T matrices
    X = np.subtract.outer(y1[:, 0], y2[:, 0])
    T = np.subtract.outer(y1[:, 1], y2[:, 1])

    # Initialize kernel matrix K
    K = np.zeros((len(y1[:,0]), len(y1[:,1])))

    # Compute the kernel matrix
    for i in range(len(y1[:,0])):
        for j in range(len(y1[:,1])):
            Kk = 0
            for n in range(len(xi[:,1])):
                for m in range(len(xi[:,0])):
                    if np.abs(xi[n,1]) < np.abs(xi[m,0]): 
                        D = (np.sinh(theta[m,n]) * T[i, j]) - np.cosh(theta[m,n]) * X[i, j]
                        Kk += a[m,n] * (sigma_w) * np.cos(a[m,n] * D)
                    elif np.abs(xi[n,1]) > np.abs(xi[m,0]): 
                        D = (np.cosh(theta[m,n]) * T[i, j]) - np.sinh(theta[m,n]) * X[i, j]
                        Kk += a[m,n] * (sigma_w) * np.cos(a[m,n] * D)
                    elif np.abs(xi[n,1]) == np.abs(xi[m,0]):
                        D = 0
                        Kk += a[m,n] * (sigma_w) * np.cos(a[m,n] * D)

            K[i, j] = Kk

    return K


def sine(y1, y2, params):
    """
    Definition plane waves kernel in summation of sine.

    Parameters
    ----------
    y1 : numpy.ndarray
        Array of shape (N, 2) representing [T_temps1, P_positionx1].
    y2 : numpy.ndarray
        Array of shape (M, 2) representing [T_temps2, P_positionx2].
    params : dict
        Dictionary containing parameters:
        - xi: numpy.ndarray
            Array of shape (L, 2) representing [k, omega]
            Wavenumber-frequency vector.
        - theta: numpy.ndarray
            Array of shape (M, N)
            Hyperbolic rotation.
        - a: numpy.ndarray
            Array of shape (M, N)
            Dilation parameter.
        - sigma_w: float
            Variance.

    Returns
    -------
    K : numpy.ndarray
        Resulting kernel matrix.
    """
    
    xi = params['xi'] 
    theta = params['theta_boostlet']
    a = params['a_dilation'] 
    sigma_w = params['sigma_w']

    # Compute X and T matrices
    X = np.subtract.outer(y1[:, 0], y2[:, 0])
    T = np.subtract.outer(y1[:, 1], y2[:, 1])

    # Initialize kernel matrix K
    K = np.zeros((len(y1[:,0]), len(y1[:,1])))

    # Compute the kernel matrix
    for i in range(len(y1[:,0])):
        for j in range(len(y1[:,1])):
            Kk = 0
            for n in range(len(xi[:,1])):
                for m in range(len(xi[:,0])):
                    if np.abs(xi[n,1]) < np.abs(xi[m,0]): 
                        D = (np.sinh(theta[m,n]) * T[i, j]) - np.cosh(theta[m,n]) * X[i, j]
                        Kk += a[m,n] * (sigma_w) * np.sin(a[m,n] * D)
                    elif np.abs(xi[n,1]) > np.abs(xi[m,0]): 
                        D = (np.cosh(theta[m,n]) * T[i, j]) - np.sinh(theta[m,n]) * X[i, j]
                        Kk += a[m,n] * (sigma_w) * np.sin(a[m,n] * D)
                    elif np.abs(xi[n,1]) == np.abs(xi[m,0]):
                        D = np.pi/2
                        Kk += a[m,n] * (sigma_w) * np.sin(a[m,n] * D)

            K[i, j] = Kk

    return K


def sine_neg(y1, y2, params):
    """
    Definition plane waves kernel in summation of negative sine.

    Parameters
    ----------
    y1 : numpy.ndarray
        Array of shape (N, 2) representing [T_temps1, P_positionx1].
    y2 : numpy.ndarray
        Array of shape (M, 2) representing [T_temps2, P_positionx2].
    params : dict
        Dictionary containing parameters:
        - xi: numpy.ndarray
            Array of shape (L, 2) representing [k, omega]
            Wavenumber-frequency vector.
        - theta: numpy.ndarray
            Array of shape (M, N)
            Hyperbolic rotation.
        - a: numpy.ndarray
            Array of shape (M, N)
            Dilation parameter.
        - sigma_w: float
            Variance.

    Returns
    -------
    K : numpy.ndarray
        Resulting kernel matrix.
    """
    
    xi = params['xi'] 
    theta = params['theta_boostlet']
    a = params['a_dilation'] 
    sigma_w = params['sigma_w']

    # Compute X and T matrices
    X = np.subtract.outer(y1[:, 0], y2[:, 0])
    T = np.subtract.outer(y1[:, 1], y2[:, 1])

    # Initialize kernel matrix K
    K = np.zeros((len(y1[:,0]), len(y1[:,1])))

    # Compute the kernel matrix
    for i in range(len(y1[:,0])):
        for j in range(len(y1[:,1])):
            Kk = 0
            for n in range(len(xi[:,1])):
                for m in range(len(xi[:,0])):
                    if np.abs(xi[n,1]) < np.abs(xi[m,0]): 
                        D = (np.sinh(theta[m,n]) * T[i, j]) - np.cosh(theta[m,n]) * X[i, j]
                        Kk -= a[m,n] * (sigma_w) * np.sin(a[m,n] * D)
                    elif np.abs(xi[n,1]) > np.abs(xi[m,0]): 
                        D = (np.cosh(theta[m,n]) * T[i, j]) - np.sinh(theta[m,n]) * X[i, j]
                        Kk -= a[m,n] * (sigma_w) * np.sin(a[m,n] * D)
                    elif np.abs(xi[n,1]) == np.abs(xi[m,0]):
                        D = np.pi/2
                        Kk -= a[m,n] * (sigma_w) * np.sin(a[m,n] * D)

            K[i, j] = Kk

    return K


"""
Boostlet kernel
@author: Awen Callo
"""


def cosine_boostlet(y1, y2, params):
    """
    Definition boostlet kernel in summation of cosine sub-kernel.

    Parameters
    ----------
    y1 : numpy.ndarray
        Array of shape (N, 2) representing [T_temps1, P_positionx1].
    y2 : numpy.ndarray
        Array of shape (M, 2) representing [T_temps2, P_positionx2].
    params : dict
        Dictionary containing parameters:
        - xi: numpy.ndarray
            Array of shape (L, 2) representing [k, omega]
            Wavenumber-frequency vector.
        - Boost parameters:
            - theta: numpy.ndarray
                Array of shape (M, N)
                Hyperbolic rotation.
            - theta_i: array_like (I,1)
                Hyperbolic rotation i^th.
            - mu_thetai: array_like (I,1)
                Mean of the boost.
            - sigma_thetai: array_like (I,1)
                Standard deviation of the boost.
        - Dilation parameters:
            - a: numpy.ndarray
                Array of shape (M, N) 
                Dilation parameter.
            - a_i: array_like (I,1)
                Dilation parameter i^th.
            - mu_ai: array_like (I,1)
                Mean of the dilation.
            - sigma_ai: array_like (I,1)
                Standard deviation of the dilation.
                
    Returns
    -------
    K : numpy.ndarray
        Resulting kernel matrix.
    """
    
    xi = params['xi']  

    # Boost parameters
    theta = params['theta_boostlet']
    theta_i = params['theta_i']
    mu_thetai = params['mu_thetai']
    sigma_thetai = params['sigma_thetai']

    # Dilation parameters
    a = params['a_dilation'] 
    a_i = params['a_i']
    mu_ai = params['mu_ai']
    sigma_ai = params['sigma_ai']

    # Compute X and T matrices
    X = np.subtract.outer(y1[:, 0], y2[:, 0])
    T = np.subtract.outer(y1[:, 1], y2[:, 1])

    # Initialize kernel matrix K
    K = np.zeros((len(y1[:,0]), len(y1[:,1])))

    # Compute the kernel matrix
    
    for x in range(len(y1[:,0])):
        for t in range(len(y1[:,1])):
            Ki = 0
            wi = 1
            for i in range(len(a_i)):
                for j in range(len(theta_i[0,:])):
                    Kk = 0
                    for n in range(len(xi[:,1])):
                        for m in range(len(xi[:,0])):
                            if np.abs(xi[n,1]) < np.abs(xi[m,0]): 
                                #D = np.sinh(theta[m,n]) * T[x, t] - np.cosh(theta[m,n]) * X[x, t] # Without dilation-boost matrix M_{a,\theta}
                                D = -X[x, t]*a[m,n] # With dilation-boost matrix M_{a,\theta}
                                #Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.cos(a[m,n] * D) * np.exp((-1/2)*(((a[m,n]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta[m,n]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                                Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.cos(a[m,n] * D) * np.exp((-1/2)*(((a_i[i]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta_i[i,j]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                            elif np.abs(xi[n,1]) > np.abs(xi[m,0]): 
                                #D = np.cosh(theta[m,n]) * T[x, t] - np.sinh(theta[m,n]) * X[x, t] # Without dilation-boost matrix M_{a,\theta}
                                D = T[x, t]*a[m,n] # With dilation-boost matrix M_{a,\theta}
                                #Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.cos(a[m,n] * D) * np.exp((-1/2)*(((a[m,n]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta[m,n]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                                Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.cos(a[m,n] * D) * np.exp((-1/2)*(((a_i[i]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta_i[i,j]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                            elif np.abs(xi[n,1]) == np.abs(xi[m,0]):
                                D = 0
                                #Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.cos(D) * np.exp((-1/2)*(((a[m,n]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta[m,n]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                                Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.cos(D) * np.exp((-1/2)*(((a_i[i]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta_i[i,j]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                    Ki += Kk
                K[x,t] = wi * Ki

    return K


def sine_boostlet(y1, y2, params):
    """
    Definition boostlet kernel in summation of sine sub-kernel.

    Parameters
    ----------
    y1 : numpy.ndarray
        Array of shape (N, 2) representing [T_temps1, P_positionx1].
    y2 : numpy.ndarray
        Array of shape (M, 2) representing [T_temps2, P_positionx2].
    params : dict
        Dictionary containing parameters:
        - xi: numpy.ndarray
            Array of shape (L, 2) representing [k, omega]
            Wavenumber-frequency vector.
        - Boost parameters:
            - theta: numpy.ndarray
                Array of shape (M, N)
                Hyperbolic rotation.
            - theta_i: array_like (I,1)
                Hyperbolic rotation i^th.
            - mu_thetai: array_like (I,1)
                Mean of the boost.
            - sigma_thetai: array_like (I,1)
                Standard deviation of the boost.
        - Dilation parameters:
            - a: numpy.ndarray
                Array of shape (M, N) 
                Dilation parameter.
            - a_i: array_like (I,1)
                Dilation parameter i^th.
            - mu_ai: array_like (I,1)
                Mean of the dilation.
            - sigma_ai: array_like (I,1)
                Standard deviation of the dilation.
                
    Returns
    -------
    K : numpy.ndarray
        Resulting kernel matrix.
    """
    
    xi = params['xi']  

    # Boost parameters
    theta = params['theta_boostlet']
    theta_i = params['theta_i']
    mu_thetai = params['mu_thetai']
    sigma_thetai = params['sigma_thetai']

    # Dilation parameters
    a = params['a_dilation'] 
    a_i = params['a_i']
    mu_ai = params['mu_ai']
    sigma_ai = params['sigma_ai']

    # Compute X and T matrices
    X = np.subtract.outer(y1[:, 0], y2[:, 0])
    T = np.subtract.outer(y1[:, 1], y2[:, 1])

    # Initialize kernel matrix K
    K = np.zeros((len(y1[:,0]), len(y1[:,1])))

    # Compute the kernel matrix
    
    for x in range(len(y1[:,0])):
        for t in range(len(y1[:,1])):
            Ki = 0
            wi = 1
            for i in range(len(a_i)):
                for j in range(len(theta_i[0,:])):
                    Kk = 0
                    for n in range(len(xi[:,1])):
                        for m in range(len(xi[:,0])):
                            if np.abs(xi[n,1]) < np.abs(xi[m,0]): 
                                #D = np.sinh(theta[m,n]) * T[x, t] - np.cosh(theta[m,n]) * X[x, t] # Without dilation-boost matrix M_{a,\theta}
                                D = -X[x, t]*a[m,n] # With dilation-boost matrix M_{a,\theta}
                                #Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(a[m,n] * D) * np.exp((-1/2)*(((a[m,n]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta[m,n]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                                Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(a[m,n] * D) * np.exp((-1/2)*(((a_i[i]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta_i[i,j]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                            elif np.abs(xi[n,1]) > np.abs(xi[m,0]): 
                                #D = np.cosh(theta[m,n]) * T[x, t] - np.sinh(theta[m,n]) * X[x, t] # Without dilation-boost matrix M_{a,\theta}
                                D = T[x, t]*a[m,n] # With dilation-boost matrix M_{a,\theta}
                                #Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(a[m,n] * D) * np.exp((-1/2)*(((a[m,n]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta[m,n]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                                Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(a[m,n] * D) * np.exp((-1/2)*(((a_i[i]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta_i[i,j]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                            elif np.abs(xi[n,1]) == np.abs(xi[m,0]):
                                D = np.pi/2
                                #Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(D) * np.exp((-1/2)*(((a[m,n]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta[m,n]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                                Kk += (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(D) * np.exp((-1/2)*(((a_i[i]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta_i[i,j]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                    Ki += Kk
                K[x,t] = wi * Ki
    
    return K


def sine_neg_boostlet(y1, y2, params):
    """
    Definition boostlet kernel in summation of negative sine sub-kernel.

    Parameters
    ----------
    y1 : numpy.ndarray
        Array of shape (N, 2) representing [T_temps1, P_positionx1].
    y2 : numpy.ndarray
        Array of shape (M, 2) representing [T_temps2, P_positionx2].
    params : dict
        Dictionary containing parameters:
        - xi: numpy.ndarray
            Array of shape (L, 2) representing [k, omega]
            Wavenumber-frequency vector.
        - Boost parameters:
            - theta: numpy.ndarray
                Array of shape (M, N)
                Hyperbolic rotation.
            - theta_i: array_like (I,1)
                Hyperbolic rotation i^th.
            - mu_thetai: array_like (I,1)
                Mean of the boost.
            - sigma_thetai: array_like (I,1)
                Standard deviation of the boost.
        - Dilation parameters:
            - a: numpy.ndarray
                Array of shape (M, N) 
                Dilation parameter.
            - a_i: array_like (I,1)
                Dilation parameter i^th.
            - mu_ai: array_like (I,1)
                Mean of the dilation.
            - sigma_ai: array_like (I,1)
                Standard deviation of the dilation.
                
    Returns
    -------
    K : numpy.ndarray
        Resulting kernel matrix.
    """
    
    xi = params['xi']  

    # Boost parameters
    theta = params['theta_boostlet']
    theta_i = params['theta_i']
    mu_thetai = params['mu_thetai']
    sigma_thetai = params['sigma_thetai']

    # Dilation parameters
    a = params['a_dilation'] 
    a_i = params['a_i']
    mu_ai = params['mu_ai']
    sigma_ai = params['sigma_ai']

    # Compute X and T matrices
    X = np.subtract.outer(y1[:, 0], y2[:, 0])
    T = np.subtract.outer(y1[:, 1], y2[:, 1])

    # Initialize kernel matrix K
    K = np.zeros((len(y1[:,0]), len(y1[:,1])))

    # Compute the kernel matrix
    
    for x in range(len(y1[:,0])):
        for t in range(len(y1[:,1])):
            Ki = 0
            wi = 1
            for i in range(len(a_i)):
                for j in range(len(theta_i[0,:])):
                    Kk = 0
                    for n in range(len(xi[:,1])):
                        for m in range(len(xi[:,0])):
                            if np.abs(xi[n,1]) < np.abs(xi[m,0]): 
                                #D = np.sinh(theta[m,n]) * T[x, t] - np.cosh(theta[m,n]) * X[x, t] # Without dilation-boost matrix M_{a,\theta}
                                D = -X[x, t]*a[m,n] # With dilation-boost matrix M_{a,\theta}
                                #Kk -= (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(a[m,n] * D) * np.exp((-1/2)*(((a[m,n]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta[m,n]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                                Kk -= (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(a[m,n] * D) * np.exp((-1/2)*(((a_i[i]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta_i[i,j]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                            elif np.abs(xi[n,1]) > np.abs(xi[m,0]): 
                                #D = np.cosh(theta[m,n]) * T[x, t] - np.sinh(theta[m,n]) * X[x, t] # Without dilation-boost matrix M_{a,\theta}
                                D = T[x, t]*a[m,n] # With dilation-boost matrix M_{a,\theta}
                                #Kk -= (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(a[m,n] * D) * np.exp((-1/2)*(((a[m,n]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta[m,n]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                                Kk -= (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(a[m,n] * D) * np.exp((-1/2)*(((a_i[i]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta_i[i,j]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                            elif np.abs(xi[n,1]) == np.abs(xi[m,0]):
                                D = np.pi/2
                                #Kk -= (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(D) * np.exp((-1/2)*(((a[m,n]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta[m,n]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                                Kk -= (a[m,n]/(2*np.pi*sigma_ai[i]*sigma_thetai[i,j])) * np.sin(D) * np.exp((-1/2)*(((a_i[i]-mu_ai[i])**2)/(sigma_ai[i]**2) + ((theta_i[i,j]-mu_thetai[i,j])**2)/(sigma_thetai[i,j]**2)))
                    Ki += Kk
                K[x,t] = wi * Ki

    return K
