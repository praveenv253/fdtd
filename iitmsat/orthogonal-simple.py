#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
2D simulation of a gaussian source using orthogonal dipole coordinates.
"""

import scipy as sp
import matplotlib.pyplot as pl

# Number of cells in each coordinate
SIZE_NU = 100
SIZE_MU = 100
# Maximum number of iterations
MAXTIME = 1000
PPW = 20        # Points per wave of the ricker source

## Field matrix definitions ##

# E-field matrices are defined mainly at "integral points".
# More precisely, the i-component of E, E_i, is defined at half-integral points
# of i, but at integral points of the other two axes.
# If (x, y, z) is an integral point in the coordinate space, then a typical
# E_y point is (x, y+0.5, z).
# As a result of this discretization, the dimension of the E_i matrix will
# become SIZE_I-1 in the dimension corresponding to coordinate i.
E_nu = sp.zeros((SIZE_NU-1, SIZE_MU))
E_phi = sp.zeros((SIZE_NU, SIZE_MU))
# E_mu is zero in this simulation

# H-field matrices are defined mainly at "half-integral points".
# More precisely, the i-th component H_i is defined at integral points of i,
# but at half-integral points of the other two axes.
# If (x, y, z) is an integral point in the coordinate space, then a typical
# H_y point is (x+0.5, y, z+0.5).
# As a result of this discretization, the dimension of the H_i matrix will
# become SIZE_J-1 and SIZE_K-1 in the dimensions j and k, corresponding to the
# two coordinates other than i.
H_nu = sp.zeros((SIZE_NU, SIZE_MU-1))
H_phi = sp.zeros((SIZE_NU-1, SIZE_MU-1))
H_mu = sp.zeros((SIZE_NU-1, SIZE_MU))

## Courant stability factor and characteristic impedance ##

Sc_nu = 0.2 #1 / sp.sqrt(2.0)
Sc_mu = 0.2 #1 / sp.sqrt(2.0)
imp0 = 377.0

## Constants for the PML absorbing boundary condition ##

# Positions of PML - not used right now
#PML_WIDTH = 30
#top = SIZE_MU - PML_WIDTH
#bottom = PML_WIDTH
#left = PML_WIDTH
#right = SIZE_NU - PML_WIDTH

## Medium ##

# Permittivity and permeability of PML - also not used right now
MU_R = 1
EPSILON_R = 1
#muR = sp.ones((SIZE_NU, SIZE_MU))
#muR[:, bottom:top] *= MU_R
#epsR = sp.ones((SIZE_NU, SIZE_MU))
#epsR[:, bottom:top] *= EPSILON_R

## Losses, including PML ##

# Electric and magnetic losses - currently not used
#LOSS_NU = 0 * (sp.mgrid[0:PML_WIDTH, 0:SIZE_MU][0] ** 2)
#LOSS_MU = 0 * (sp.mgrid[0:SIZE_NU, 0:PML_WIDTH][1] ** 2)
#loss_nu = sp.zeros((SIZE_NU, SIZE_MU))
#loss_mu = sp.zeros((SIZE_NU, SIZE_MU))
#loss_nu[right:, :] += LOSS_NU
#loss_nu[:left, :] += LOSS_NU[::-1, :]
#loss_mu[:, top:] += LOSS_MU
#loss_mu[:, :bottom] += LOSS_MU[:, ::-1]

## Grid for plotting ##

grid = sp.mgrid[0.01:1.01:1j*SIZE_NU, 0.01:1.01:1j*SIZE_MU]

# Re-represent the grid values in cartesian coordinates
nu = grid[0]
mu = grid[1]
alpha = (256.0 * mu * mu) / (27 * (nu ** 4))
beta = (1 + sp.sqrt(1 + alpha)) ** (2.0/3)
gamma = alpha ** (1.0/3)
zeta = ( ((beta**2 + beta*gamma + gamma**2) / beta) ** (1.5) ) / 2
r = 4 * zeta / (nu * (1 + zeta) * (1 + sp.sqrt(2*zeta - 1)))
sin_theta = sp.sqrt(r * nu)
delta = sp.sqrt(4 - 3 * sin_theta ** 2)
gridx = r * sp.sqrt(1 - sin_theta**2)
gridy = r * sin_theta

## Coordinate-dependent scale factors ##

# We define the scale factors at *integral points*. Thus, the position of the
# scale factors does not coincide with *any* field point, because the fields
# are defined at points which are half-integral in at least one coordinate
RI = 1
h_nu = (r ** 2) / (RI * sin_theta * delta)
h_phi = r * sin_theta
h_mu = (r ** 3) / (RI * RI * delta)

## Coefficients for update equations ##

# In this section, the notation cA_i_B_j is used to mean the coefficient of the
# j-component of B in the update equation for the i-component of A.
# The notation is reduced to cA_i_B where there is only one component of B in
# A_i's update equation.

# Coefficients for the H_nu update
cH_nu_H = sp.ones(H_nu.shape)
# H_nu is defined at points like (nu, phi+0.5, mu+0.5), but h_mu and h_phi are
# defined at points like (nu, phi, mu). To evaluate h_mu and h_phi at
# (nu, phi+0.5, mu+0.5), we perform linear interpolation.
# For now, we assume symmetry in the phi-direction so that interpolation is not
# required for the phi dimension.
H_nu_h_mu = (h_mu[:, 1:] + h_mu[:, :-1]) / 2
H_nu_h_phi = (h_phi[:, 1:] + h_phi[:, :-1]) / 2
cH_nu_E = Sc_mu / (imp0 * MU_R * H_nu_h_phi * H_nu_h_mu)

# Coefficients for the H_phi update
cH_phi_H = sp.ones(H_phi.shape)
# H_phi is defined at points like (nu+0.5, phi, mu+0.5), but h_nu and h_mu are
# defined at points like (nu, phi, mu). To evaluate h_nu and h_mu at
# (nu+0.5, phi, mu+0.5), we perform linear interpolation.
H_phi_h_nu = (  h_nu[1:, 1:] + h_nu[1:, :-1]
              + h_nu[:-1, 1:] + h_nu[:-1, :-1] ) / 4
H_phi_h_mu = (  h_mu[1:, 1:] + h_mu[1:, :-1]
              + h_mu[:-1, 1:] + h_mu[:-1, :-1] ) / 4
H_phi_h_nu_fwd_avg = (h_nu[1:, 1:] + h_nu[:-1, 1:]) / 2
H_phi_h_nu_bwd_avg = (h_nu[1:, :-1] + h_nu[:-1, :-1]) / 2
cH_phi_E = - Sc_mu / (imp0 * MU_R * H_phi_h_nu * H_phi_h_mu)

# Coefficients for the H_mu update
cH_mu_H = sp.ones(H_mu.shape)
# Linear interpolation
H_mu_h_nu = (h_nu[1:, :] + h_nu[:-1, :]) / 2
H_mu_h_phi = (h_phi[1:, :] + h_phi[:-1, :]) / 2
cH_mu_E_phi = - Sc_nu / (imp0 * MU_R * H_mu_h_nu * H_mu_h_phi)

# Coefficients for the E_nu update
cE_nu_E = sp.ones((E_nu[:, 1:-1]).shape)
# Linear interpolation
E_nu_h_phi = (h_phi[1:, 1:-1] + h_phi[:-1, 1:-1]) / 2
E_nu_h_mu = (h_mu[1:, 1:-1] + h_mu[:-1, 1:-1]) / 2
# Here, we need to do linear interpolation for the h_phi that multiplies H_phi
# in the update equation for E_nu.
E_nu_h_phi_fwd_avg = (  h_phi[1:, 2:] + h_phi[1:, 1:-1]
                      + h_phi[:-1, 2:] + h_phi[:-1, 1:-1] ) / 4
E_nu_h_phi_bwd_avg = (  h_phi[1:, 1:-1] + h_phi[1:, :-2]
                      + h_phi[:-1, 1:-1] + h_phi[:-1, :-2] ) / 4
cE_nu_H_phi = - Sc_mu * imp0 / (EPSILON_R * E_nu_h_phi * E_nu_h_mu)

# Coefficients for the E_phi update
cE_phi_E = sp.ones((E_phi[1:-1, 1:-1]).shape)
# No linear interpolation here because of assumed symmetry in the phi-direction
E_phi_h_nu = h_nu[1:-1, 1:-1]
E_phi_h_mu = h_mu[1:-1, 1:-1]
# Here, we do linear interpolation for the h_nu that multiplies H_nu and for
# the h_mu that multiplies H_mu in the update equation for E_phi.
E_phi_h_nu_fwd_avg = (h_nu[1:-1, 2:] + h_nu[1:-1, 1:-1]) / 2
E_phi_h_nu_bwd_avg = (h_nu[1:-1, 1:-1] + h_nu[1:-1, :-2]) / 2
E_phi_h_mu_fwd_avg = (h_mu[2:, 1:-1] + h_mu[1:-1, 1:-1]) / 2
E_phi_h_mu_bwd_avg = (h_mu[1:-1, 1:-1] + h_mu[:-2, 1:-1]) / 2
cE_phi_H_nu = Sc_mu * imp0 / (EPSILON_R * E_phi_h_nu * E_phi_h_mu)
cE_phi_H_mu = - Sc_nu * imp0 / (EPSILON_R * E_phi_h_nu * E_phi_h_mu)

# Turn on interactive mode for animation
pl.ion()

for t in range(MAXTIME):

    ## Update equations ##
    
    # Update H-field, nu-component
    H_nu = cH_nu_H * H_nu + cH_nu_E * (  h_phi[:, 1:] * E_phi[:, 1:]
                                       - h_phi[:, :-1] * E_phi[:, :-1] )
    
    # Update H-field, phi-componenet
    H_phi = (  cH_phi_H * H_phi
             + cH_phi_E * (  H_phi_h_nu_fwd_avg * E_nu[:, 1:]
                           - H_phi_h_nu_bwd_avg * E_nu[:, :-1] ) )
    
    # Update H-field, mu-component (del/delphi = 0)
    H_mu = cH_mu_H * H_mu + cH_mu_E_phi * (  h_phi[1:, :] * E_phi[1:, :]
                                           - h_phi[:-1, :] * E_phi[:-1, :] )
    
    # Update E-field, nu-component (del/delphi = 0)
    E_nu[:, 1:-1] = (  cE_nu_E * E_nu[:, 1:-1]
                     + cE_nu_H_phi * (  E_nu_h_phi_fwd_avg * H_phi[:, 1:] 
                                      - E_nu_h_phi_bwd_avg * H_phi[:, :-1] ) )
    
    # Update E-field, phi-component
    E_phi[1:-1, 1:-1] = (  cE_phi_E * E_phi[1:-1, 1:-1]
                         + cE_phi_H_nu * ( E_phi_h_nu_fwd_avg * H_nu[1:-1, 1:]
                                          -E_phi_h_nu_bwd_avg * H_nu[1:-1, :-1])
                         + cE_phi_H_mu * ( E_phi_h_mu_fwd_avg * H_mu[1:, 1:-1]
                                          -E_phi_h_mu_bwd_avg * H_mu[:-1, 1:-1])
                        )
    ## Hard source ##
    
    # Ricker wavelet
    #arg = (sp.pi * (Sc_nu * t / PPW - 1.0)) ** 2
    #E_phi[SIZE_NU/2, SIZE_MU/2] = (1 - 2*arg) * sp.exp(-arg)
    
    ## Gaussian
    E_phi[50, 50] = sp.exp(-(t-30) * (t-30) / 100.0)
    
    # Sine wave
    #omega = 0.0001
    #E_phi[SIZE_NU/2 , SIZE_MU/2] = sp.cos(omega * t)
    
    ## Plotting ##
    
    if t % 5 == 0:
        pl.contour(gridy, gridx, E_phi, 100)
        #pl.pcolor(E_phi)
        pl.draw()
        pl.clf()
