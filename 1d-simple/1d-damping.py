#!/usr/bin/env python

import scipy as sp
import matplotlib.pylab as pl

SIZE = 200
MAXTIME = 300
TFSF_POS = 50 # Index of electric field (included) from which total field starts
INTERFACE = 100 # E-field index (inclusive) from where the new medium starts
EPSILON_R = 9
MU_R = 1
LOSS = 0.01

ez = sp.zeros(SIZE)
hy = sp.zeros(SIZE)
imp0 = 377.0
snapshots = []
epsR = sp.ones(SIZE)
epsR[INTERFACE:] *= EPSILON_R
muR = sp.ones(SIZE)
muR[INTERFACE:] *= MU_R
ceze = sp.ones(SIZE)
ceze[INTERFACE:] *= (1 - LOSS) / (1 + LOSS)
cezh = sp.ones(SIZE)
cezh[INTERFACE:] *= 1 / (1 + LOSS)

for t in range(MAXTIME):
    # TODO: Find out why exactly there is a subtle difference in the incremental
    # electric and magnetic fields. Solution-wise, there is no noticable
    # difference.
    ezinc = sp.exp(-(t+0.5-(-0.5)-30) * (t+0.5-(-0.5)-30) / 100.0)
    hyinc = sp.exp(-(t-30) * (t-30) / 100.0) / (imp0 * muR)
    
    # TODO: Find out why the ABCs must be given *before* the corresponding 
    # update equation. I'd have thought that it should be done *after*.
    hy[-1] = hy[-2]
    
    hy[:-1] = hy[:-1] + (ez[1:] - ez[:-1]) / (imp0 * muR[:-1])
    
    hy[TFSF_POS-1] -= hyinc[TFSF_POS-1]
    
    ez[0] = ez[1]
    
    ez[1:] = ceze[1:] * ez[1:] + cezh[1:] * (hy[1:] - hy[:-1]) * imp0 / epsR[1:]
    
    ez[TFSF_POS] += ezinc
    
    if t % 10 == 0:
        snapshots.append(ez.copy() + (t / 10)*sp.ones(SIZE))

print 'Done with time stepping'

pl.figure(0)
for out in snapshots:
    pl.plot(range(len(out)), out, 'b-')
pl.show()
