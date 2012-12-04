#!/usr/bin/env python

import scipy as sp
import matplotlib.pylab as pl

SIZE = 200
MAXTIME = 500

ez = sp.zeros(SIZE)
hy = sp.zeros(SIZE)
imp0 = 377.0
snapshots = []

for t in range(MAXTIME):
    hy[:-1] = hy[:-1] + (ez[1:] - ez[:-1]) / imp0
    
    ez[1:] = ez[1:] + (hy[1:] - hy[:-1]) * imp0
    
    ez[0] = sp.exp(-(t-30) * (t-30) / 100.0)
    
    if t % 10 == 0:
        snapshots.append(ez.copy() + (t / 10)*sp.ones(SIZE))

print 'Done with time stepping'

pl.figure(0)
for out in snapshots:
    pl.plot(range(len(out)), out, 'b-')
pl.show()
