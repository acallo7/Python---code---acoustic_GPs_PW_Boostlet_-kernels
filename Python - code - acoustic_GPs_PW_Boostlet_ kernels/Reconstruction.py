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

Module : Reconstruction
Functions: plot_reconstruction, show_soundfield_true, show_soundfield_predict,
show_soundfield_uncertainty
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_reconstruction(tx, txs, tx_predict, txs_predict, p_true, p_measured, p_predict, uncertainty, density_x, density_t, i_measured):
    """
    Plot the reconstruction of the sound field.

    Parameters
    ----------
    tx : numpy.ndarray
        Array of shape (N, 2) representing time and space points.
    txs : numpy.ndarray
        Array of shape (M, 2) representing measured time and space points.
    p_true : numpy.ndarray
        True sound field data.
    p_predict : numpy.ndarray
        Predicted sound field data.
    uncertainty : numpy.ndarray
        Uncertainty data.
    density_x : int
        Density of points in the x dimension.
    density_t : int
        Density of points in the t dimension.
    i_measured : list
        List of measured indices.
    """
    
    # Define the scale factor
    scale = 0.5

    # Create a figure with a specific size
    fig, axs = plt.subplots(2, 2, figsize=(17 * scale, 5 * scale))

    # Calculate the color limits
    axs[0,0].labelsize = 20
    axs[0,1].labelsize = 20
    axs[1,0].labelsize = 20
    axs[1,1].labelsize = 20
    axs[0,0].tick_params(labelsize = 18)
    axs[0,1].tick_params(labelsize = 18)
    axs[1,0].tick_params(labelsize = 18)
    axs[1,1].tick_params(labelsize = 18)
    # Store the values in a list
    #lim = (min(p_true.min(), p_predict.min()), max(p_true.max(), p_predict.max()))
    #lim = (-max(np.abs(p_true).max(), np.abs(p_predict).max()), max(np.abs(p_true).max(), np.abs(p_predict).max()))
    lim_true = [np.min(p_true), np.max(p_true)]
    lim_predict = [np.min(p_predict), np.max(p_predict)]
    #lim_true = [-1, 1]
    #lim_predict = [-1, 1]
    #lim = [-1, 1]

    lim_uncertainty = [np.min(uncertainty), np.max(uncertainty)]

    # Plot the true sound field
    show_soundfield_true(axs[0,0], tx[:,0], tx[:,1], p_true, lim_true, 'none')
    #plt.colorbar(im_true, cax=axs[0])
    #for j in range(len(tx[:,1])):
        #for i in range(len(i_measured)):  
            #axs[0].plot(tx[i_measured[i],0], tx[2,1], 's', markeredgecolor='k', markerfacecolor=(0.6, 0.8, 0.2))
    axs[0,0].set_xlabel('x [m]')
    axs[0,0].set_ylabel('t [s]')
    axs[0,0].set_title('True Sound Field')
    
    # Plot the measured sound field sampled
    show_soundfield_true(axs[0,1], tx[:,0], tx[:,1], p_measured, lim_true, 'none')
    #plt.colorbar(im_true, cax=axs[0])
    #for j in range(len(tx[:,1])):
        #for i in range(len(i_measured)):  
            #axs[1].plot(tx[i_measured[i],0], tx[2,1], 's', markeredgecolor='k', markerfacecolor=(0.6, 0.8, 0.2))
    axs[0,1].set_xlabel('x [m]')
    axs[0,1].set_ylabel('t [s]')
    axs[0,1].set_title('Measured Sound Field Sampled')

    # Plot the predicted sound field
    show_soundfield_predict(axs[1,0], tx_predict[:,0], tx_predict[:,1], p_predict, lim_predict, 'none')
    #plt.colorbar(im_predict, cax=axs[1])
    #for j in range(len(tx[:,1])):
        #for i in range(len(i_measured)):  
            #axs[2].plot(tx[i_measured[i],0], tx[2,1], 's', markeredgecolor='k', markerfacecolor=(0.6, 0.8, 0.2))
    axs[1,0].set_xlabel('x [m]')
    axs[1,0].set_ylabel('t [s]')
    axs[1,0].set_title('Mean Prediction')

    # Plot the uncertainty
    show_soundfield_uncertainty(axs[1,1], tx_predict[:,0], tx_predict[:,1], uncertainty, lim_uncertainty, 'none')
    #for j in range(len(tx)):
    #    for i in i_measured:
    #        axs[3].plot(tx[i, 1], txs[j, 0], 's', markeredgecolor='k', markerfacecolor=[0.6, 0.8, 0.2])
    axs[1,1].set_xlabel('x [m]')
    axs[1,1].set_ylabel('t [s]')
    axs[1,1].set_title('Uncertainty')

    plt.tight_layout()
    plt.show()

def show_soundfield_true(ax, r_x, r_t, p, lim, what, **kwargs):
    """
    Show sound field on the given axis.
    
    Parameters:
    - ax: matplotlib Axes object
    - r_tx: 2xN array, first row time, second row space
    - p: complex or real pressure values
    - lim: limits for color scale (can be None)
    - what: 'phase', 'spl', or any other value for raw
    - density_x: unused here, kept for compatibility
    - density_t: unused here, kept for compatibility
    - *args: additional optional arguments (unused)
    """
    
    if what == 'phase':
        z = np.angle(p)
    elif what == 'spl':
        z = 20 * np.log10(np.abs(p) / 2e-5)
    else:
        z = p

    z = z.T

    tmin = np.min(r_t)
    xmin = np.min(r_x)
    tmax = np.max(r_t)
    xmax = np.max(r_x)

    xl = p.shape[0]
    tl = p.shape[1]
    xg = np.linspace(xmin, xmax, xl)
    tg = np.linspace(tmin, tmax, tl)
    Xg, Tg = np.meshgrid(xg, tg)

    if lim is None:
        lim = [np.min(z), np.max(z)]

    max_abs_z = np.max(np.abs(z))

    # Zg=z/max_abs_z
    Zg=z
    
    cs = ax.pcolormesh(Xg, Tg, Zg, vmin=lim[0], vmax=lim[1], cmap='RdBu', **kwargs) #'jet', 'RdBu'

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([tmin, tmax])

    cbar = plt.colorbar(cs, ax=ax, norm=lim)
    # cbar.set_label('Normalized h(t), max. value : {:.2f}'.format(max_abs_z), rotation=90, labelpad=15)
    
    return cs, max_abs_z

def show_soundfield_predict(ax, r_x, r_t, p, lim, what, **kwargs):
    """
    Show sound field on the given axis.
    
    Parameters:
    - ax: matplotlib Axes object
    - r_tx: 2xN array, first row time, second row space
    - p: complex or real pressure values
    - lim: limits for color scale (can be None)
    - what: 'phase', 'spl', or any other value for raw
    - density_x: unused here, kept for compatibility
    - density_t: unused here, kept for compatibility
    - *args: additional optional arguments (unused)
    """
    
    if what == 'phase':
        z = np.angle(p)
    elif what == 'spl':
        z = 20 * np.log10(np.abs(p) / 2e-5)
    else:
        z = p

    z = z.T  

    tmin = np.min(r_t)
    xmin = np.min(r_x)
    tmax = np.max(r_t)
    xmax = np.max(r_x)

    xl=p.shape[0]
    tl=p.shape[1]
    xg = np.linspace(xmin, xmax, xl)
    tg = np.linspace(tmin, tmax, tl)
    Xg, Tg = np.meshgrid(xg, tg)

    if lim is None:
        lim = [np.min(z), np.max(z)]

    # max_abs_z = np.max(np.abs(z))
    # Zg=z/max_abs_z
    Zg =z

    # cs = ax.pcolormesh(Xg, Tg, 100*Zg, shading='auto', cmap='jet')
    # cs.set_edgecolor('none')
    
    cs = ax.pcolormesh(Xg, Tg, Zg, vmin=lim[0], vmax=lim[1], cmap='RdBu', **kwargs)
    # ax.set_aspect("equal")

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([tmin, tmax])

    cbar = plt.colorbar(cs, ax=ax, norm=lim)
    # cbar.set_label('Normalized h(t), max. value : {:.2f}'.format(max_abs_z), rotation=90, labelpad=15)
    
    return cs

def show_soundfield_uncertainty(ax, r_x, r_t, p, lim, what, **kwargs):
    """
    Show sound field on the given axis.
    
    Parameters:
    - ax: matplotlib Axes object
    - r_tx: 2xN array, first row time, second row space
    - p: complex or real pressure values
    - lim: limits for color scale (can be None)
    - what: 'phase', 'spl', or any other value for raw
    - density_x: unused here, kept for compatibility
    - density_t: unused here, kept for compatibility
    - *args: additional optional arguments (unused)
    """
    
    if what == 'phase':
        z = np.angle(p)
    elif what == 'spl':
        z = 20 * np.log10(np.abs(p) / 2e-5)
    else:
        z = p

    z = z.T  

    tmin = np.min(r_t)
    xmin = np.min(r_x)
    tmax = np.max(r_t)
    xmax = np.max(r_x)

    xl=p.shape[0]
    tl=p.shape[1]
    xg = np.linspace(xmin, xmax, xl)
    tg = np.linspace(tmin, tmax, tl)
    Xg, Tg = np.meshgrid(xg, tg)

    if lim is None:
        lim = [np.min(z), np.max(z)]

    # max_abs_z = np.max(np.abs(z))
    # Zg=z/max_abs_z
    Zg=z
    
    cs = ax.pcolormesh(Xg, Tg, Zg, vmin=lim[0], vmax=lim[1], cmap='RdBu', **kwargs) #'jet'

    ax.set_xlim([xmin, xmax])
    ax.set_ylim([tmin, tmax])

    cbar = plt.colorbar(cs, ax=ax, norm=lim)
    # cbar.set_label('Normalized h(t), max. value : {:.2f}'.format(max_abs_z), rotation=90, labelpad=15)
    
    return cs
