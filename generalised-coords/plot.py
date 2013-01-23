#! /usr/bin/env python

import sys
import scipy as sp
import matplotlib.pyplot as pl

print sys.argv

if len(sys.argv) != 4 and len(sys.argv) != 5:
    print 'Insufficient number of command line arguments'
    print 'usage: %s <filename_prefix> <start_number> <step> [<extension>]' % sys.argv[0]
    sys.exit(1)

prefix = sys.argv[1]
start = int(sys.argv[2])
step = int(sys.argv[3])
if len(sys.argv) == 5:
    extension = sys.argv[4]
else:
    extension = '.txt'

pl.ion()

i = start
while(1):
    filename = '%s%d%s' % (prefix, i, extension)
    try:
        f = open(filename)
    except IOError:
        print 'No more files'
    print filename
    line_number = 1
    m = None
    n = None
    a = None
    for line in f:
        if line_number == 16387:
            continue
        if line_number == 1:
            m = float(line.replace('\n', ''))
            line_number += 1
        elif line_number == 2:
            n = float(line.replace('\n', ''))
            a = sp.zeros((m, n))
            line_number += 1
        else:
            index = line_number - 3
            a[int(index / m)][index % m] = float(line.replace('\n', ''))
            line_number += 1
    
    grid = sp.mgrid[0.01:1.01:1j*n, 0.01:1.01:1j*m]

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
    gridx = r * sin_theta
    gridy = r * sp.sqrt(1 - sin_theta**2)

    # Ensure that not all a's are zero before plotting
    if a.any():
        pl.clf()
        pl.contour(gridx, gridy, a, 100)
        pl.draw()
    else:
        print '    All elements are zero'

    i += step

print 'Done'
