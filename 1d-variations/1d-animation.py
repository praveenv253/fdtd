#!/usr/bin/env python

import scipy as sp
import matplotlib.pylab as pl

SIZE = 200
MAXTIME = 500

ez = sp.zeros(SIZE)
hy = sp.zeros(SIZE)
imp0 = 377.0
Sc = 0.5

x = sp.arange(SIZE)

# Switch on interactive mode for drawing continuously
pl.ion()

for t in range(MAXTIME):
    hy[:-1] = hy[:-1] + (ez[1:] - ez[:-1]) * Sc / imp0
    
    ez[1:] = ez[1:] + (hy[1:] - hy[:-1]) * Sc * imp0
    
    ez[0] = sp.exp(-(t-30) * (t-30) / 100.0)
    
    pl.plot(x, ez, 'b-')
    pl.draw()
    pl.clf()
