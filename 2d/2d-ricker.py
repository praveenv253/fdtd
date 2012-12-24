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
Sc = 0.707

pl.ion()

for t in range(MAXTIME):
    # Update H-field, x-componenet
    hx = hx - (ez[:, 1:] - ez[:, :-1]) * Sc / imp0
    
    # Update H-field, y-component
    hy = hy + (ez[1:, :] - ez[:-1, :]) * Sc / imp0
    
    # Update E-field, z-component
    ez[1:-1, 1:-1] = (   ez[1:-1, 1:-1] 
                       + (hy[1:, 1:-1] - hy[:-1, 1:-1]) * Sc * imp0
                       - (hx[1:-1, 1:] - hx[1:-1, :-1]) * Sc * imp0 )
    
    # Hard source
    arg = (sp.pi * (Sc * t / PPW - 1.0)) ** 2
    ez[SIZE_X/2, SIZE_Y/2] = (1 - 2*arg) * sp.exp(-arg)
    #sp.exp(-(t-30) * (t-30) / 100.0)
    
    if t % 5 == 0:
        pl.contour(ez, 200)
        pl.draw()
        pl.clf()
