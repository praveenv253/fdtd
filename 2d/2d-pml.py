#!/usr/bin/env python

import sys

import scipy as sp
import matplotlib.pylab as pl

SIZE_X = 201
SIZE_Y = 201
MAXTIME = 1000
PPW = 20        # Points per wave of the ricker source

ez = sp.zeros((SIZE_X, SIZE_Y))
ezx = sp.zeros((SIZE_X, SIZE_Y))
ezy = sp.zeros((SIZE_X, SIZE_Y))
hx = sp.zeros((SIZE_X, SIZE_Y-1))
hy = sp.zeros((SIZE_X-1, SIZE_Y))
Sc_x = 1/sp.sqrt(2.0)
Sc_y = 1/sp.sqrt(2.0)
imp0 = 377.0

# Positions of PML
PML_WIDTH = 30
top = SIZE_Y - PML_WIDTH
bottom = PML_WIDTH
left = PML_WIDTH
right = SIZE_X - PML_WIDTH

# Permittivity and permeability of PML
MU_R = 1
EPSILON_R = 1
muR = sp.ones((SIZE_X, SIZE_Y))
muR[:, bottom:top] *= MU_R
epsR = sp.ones((SIZE_X, SIZE_Y))
epsR[:, bottom:top] *= EPSILON_R

# Electric and magnetic losses
LOSS_X = 0.00025 * (sp.mgrid[0:PML_WIDTH, 0:SIZE_Y][0] ** 2)
LOSS_Y = 0.00025 * (sp.mgrid[0:SIZE_X, 0:PML_WIDTH][1] ** 2)
sp.set_printoptions(precision = 2, threshold = sp.nan, linewidth = 135, suppress = True)
print LOSS_X
loss_x = sp.zeros((SIZE_X, SIZE_Y))
loss_y = sp.zeros((SIZE_X, SIZE_Y))
loss_x[right:, :] += LOSS_X
loss_x[:left, :] += LOSS_X[::-1, :]
loss_y[:, top:] += LOSS_Y
loss_y[:, :bottom] += LOSS_Y[:, ::-1]

# Coefficients multiplying Hx and Ez for the Hx update
chx_loss_y = (loss_y[:, 1:] + loss_y[:, :-1]) / 2
chx_muR = (muR[:, 1:] + muR[:, :-1]) / 2
chxh = (1 - chx_loss_y) / (1 + chx_loss_y)
chxe = - Sc_y / (chx_muR * imp0 * (1+chx_loss_y))
# Coefficients multiplying Hy and Ez for the Hy update
chy_loss_x = (loss_x[1:, :] + loss_x[:-1, :]) / 2
chy_muR = (muR[1:, :] + muR[:-1, :]) / 2
chyh = (1 - chy_loss_x) / (1 + chy_loss_x)
chye = + Sc_x / (chy_muR * imp0 * (1+chy_loss_x))
# Coefficients multiplying Ezx and Hy for the Ezx update
cez_epsR = epsR[1:-1, 1:-1]
cezx_loss_x = loss_x[1:-1, 1:-1]
cezxe = (1 - cezx_loss_x) / (1 + cezx_loss_x)
cezxh = + (Sc_x * imp0) / (cez_epsR * (1+cezx_loss_x))
# Coefficients multiplying Ezy and Hx for the Ezy update
cezy_loss_y = loss_y[1:-1, 1:-1]
cezye = (1 - cezy_loss_y) / (1 + cezy_loss_y)
cezyh = - (Sc_y * imp0) / (cez_epsR * (1+cezy_loss_y))

# Turn on interactive mode for animation
pl.ion()

for t in range(MAXTIME):
    # Update H-field, x-componenet
    hx = chxh * hx + chxe * (ez[:, 1:] - ez[:, :-1])
    
    # Update H-field, y-component
    hy = chyh * hy + chye * (ez[1:, :] - ez[:-1, :])
    
    # Update E-field
    ez[0, :] = ez[1, :]
    ez[-1, :] = ez[-2, :]
    ez[:, 0] = ez[:, 1]
    ez[:, -1] = ez[:, -2]
    ezx[1:-1, 1:-1] = (  cezxe * ezx[1:-1, 1:-1] 
                       + cezxh * (hy[1:, 1:-1] - hy[:-1, 1:-1]) )
    ezy[1:-1, 1:-1] = (  cezye * ezy[1:-1, 1:-1]
                       + cezyh * (hx[1:-1, 1:] - hx[1:-1, :-1]) )
    ez = ezx + ezy
    
    # Additive source
    arg = (sp.pi * (Sc_x * t / PPW - 1.0)) ** 2
    ez[SIZE_X/2, SIZE_Y/2] = (1 - 2*arg) * sp.exp(-arg)
    #ez[SIZE_X/2, SIZE_Y/2] += sp.exp(-(t-30) * (t-30) / 100.0)
    
    if t % 5 == 0:
        pl.contour(ez, 200)
        #pl.pcolor(ez)
        pl.draw()
        pl.clf()
