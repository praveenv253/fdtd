#!/usr/bin/env python

import scipy as sp
import matplotlib.pylab as pl

SIZE_X = 101
SIZE_Y = 101
MAXTIME = 1000
PPW = 20        # Points per wave of the ricker source

ez = sp.zeros((SIZE_X, SIZE_Y))
hx = sp.zeros((SIZE_X, SIZE_Y-1))
hy = sp.zeros((SIZE_X-1, SIZE_Y))
imp0 = 377.0
Sc = 1 / sp.sqrt(2.0)

# Position of new medium (Assume medium interface is parallel to y-axis)
# Position is therefore x-position
pos = SIZE_X / 2

# Permittivity and permeability of new medium
MU_R = 4
EPSILON_R = 9
muR = sp.ones((SIZE_X, SIZE_Y))
muR[pos:, :] *= MU_R
epsR = sp.ones((SIZE_X, SIZE_Y))
epsR[pos:, :] *= EPSILON_R

pl.ion()

for t in range(MAXTIME):
    # Update H-field, x-componenet
    hx = hx - (ez[:, 1:] - ez[:, :-1]) * Sc * 2 / (imp0*(muR[:,1:]+muR[:,:-1]))
    
    # Update H-field, y-component
    hy = hy + (ez[1:, :] - ez[:-1, :]) * Sc * 2 / (imp0*(muR[1:,:]+muR[:-1,:]))
    
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
    ez[0, SIZE_Y/2] = sp.exp(-(t-30) * (t-30) / 100.0) # Gaussian pulse
    
    if t % 5 == 0:
        pl.contour(ez.T, 200)
        pl.draw()
        pl.clf()
