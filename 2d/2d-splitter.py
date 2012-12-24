#!/usr/bin/env python

import scipy as sp
import matplotlib.pylab as pl

SIZE_X = 151
SIZE_Y = 101
MAXTIME = 2000
PPW = 20        # Points per wave of the ricker source

ez = sp.zeros((SIZE_X, SIZE_Y))
hx = sp.zeros((SIZE_X, SIZE_Y-1))
hy = sp.zeros((SIZE_X-1, SIZE_Y))
imp0 = 377.0
Sc = 1 / sp.sqrt(2.0)

# Position of waveguides (Assume waveguides are parallel to x-axis)
top_center = SIZE_Y / 2 + 10
bottom_center = SIZE_Y / 2 - 10
top_top = SIZE_Y / 2 + 21
bottom_top = SIZE_Y / 2 + 11
top_bottom = SIZE_Y / 2 - 11
bottom_bottom = SIZE_Y / 2 - 21

# Permittivity and permeability of waveguides
MU_R = 1
EPSILON_R = 9
muR = sp.ones((SIZE_X, SIZE_Y))
epsR = sp.ones((SIZE_X, SIZE_Y))
# Center waveguide
muR[:, bottom_center:top_center] *= MU_R
epsR[:, bottom_center:top_center] *= EPSILON_R
# Top waveguide
muR[:, bottom_top:top_top] *= MU_R
epsR[:, bottom_top:top_top] *= EPSILON_R
# Bottom waveguide
muR[:, bottom_bottom:top_bottom] *= MU_R
epsR[:, bottom_bottom:top_bottom] *= EPSILON_R


pl.ion()

for t in range(MAXTIME):
    # Update H-field, x-componenet
    hx = hx - (ez[:, 1:] - ez[:, :-1]) * Sc / imp0 / ((muR[:,1:] + 
                                                               muR[:,:-1]) / 2)
    
    # Update H-field, y-component
    hy = hy + (ez[1:, :] - ez[:-1, :]) * Sc / imp0 / ((muR[1:,:] + 
                                                               muR[:-1,:]) / 2)
    
    # Simple ABC for E-field
    ez[0, :] = ez[1, :]
    ez[:, 0] = ez[:, 1]
    ez[-1, :] = ez[-2, :]
    ez[:, -1] = ez[:, -2]
    
    # Update E-field, z-component
    ez[1:-1, 1:-1] = (   ez[1:-1, 1:-1] 
                       + (hy[1:, 1:-1] - hy[:-1, 1:-1]) * Sc * imp0 / epsR[1:-1, 1:-1]
                       - (hx[1:-1, 1:] - hx[1:-1, :-1]) * Sc * imp0 / epsR[1:-1, 1:-1])
    
    # Hard source(s):
    #arg = (sp.pi * (Sc * t / PPW - 1.0)) ** 2         # Uncomment for a Ricker
    #ez[0, SIZE_Y/2] = (1 - 2*arg) * sp.exp(-arg)      # wavelet
    #ez[0, SIZE_Y/2] = sp.exp(-(t-30) * (t-30) / 100.0) # Gaussian pulse
    ez[0, bottom_center:top_center] = sp.cos(0.05 * t)
    
    if t % 10 == 0:
        pl.contour(ez.T, 50)
        pl.draw()
        pl.clf()
