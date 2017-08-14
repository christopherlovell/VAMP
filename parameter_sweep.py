import numpy as np
import pandas as pd

%matplotlib inline
import matplotlib.pyplot as plt

import vpfits

from scipy.ndimage.filters import gaussian_filter
from scipy.signal import argrelextrema

import pymc as mc
from pymc.Matplot import plot
from scipy import stats
import time

cont = np.loadtxt('data/q1422.cont')
vpfit = vpfits.VPfit()
onesigmaerror = 0.02

min_region_width = 2

regions = vpfits.compute_detection_regions(cont[:,0], cont[:,2], cont[:,3], 
                                          min_region_width=min_region_width)

region_arrays = []
region_pixels = []
for region in regions:
    start = np.where(cont[:,0] == region[0])[0][0]
    end = np.where(cont[:,0] == region[1])[0][0]
    region_pixels.append([start, end])
    region_arrays.append([cont[:,0][start:end], cont[:,2][start:end], cont[:,3][start:end]])
    
bpic = []
chi2 = []
count = 0
i=2
wavelengths = region_arrays[i][0]
fluxes_orig = region_arrays[i][1]
fluxes = region_arrays[i][1]
noise = region_arrays[i][2]

r = 0
n = argrelextrema(gaussian_filter(fluxes_orig, 3), np.less)[0].shape[0]
if n < 4:
    n = 1

iterations = 5000
n_runs = 4
n_max = 3
bpic = []
chi2 = []
bpic_list = []
parameter_mesh = np.array(np.meshgrid([1e-3, 1e-5, 1e-7],
                                      [2, 6 ,10],
                                      [1e-3, 1e-5, 1e-7],
                                      [2, 6, 10],
                                      [1e-3, 1e-5, 1e-7])).T.reshape(3**5, 5)
labels = ["f{}, m{}, f{}, m{}, f{}".format(item[0], item[1], item[2], item[3], item[4]) for item in parameter_mesh]

t = time.time()
for j, i in enumerate(parameter_mesh):
    j *= 1.
    bpic = []
    chi2 = []
    n = 1
    for asda in range(n, n + n_max):
        n += 1
        for run in range(n_runs):
            #print "n: {}, run: {}".format(n, run)
            vpfit_2 = vpfits.VPfit()
            vpfit_2.initialise_model(wavelengths, fluxes, n)
            vpfit_2.map = mc.MAP(vpfit_2.model)
            vpfit_2.mcmc = mc.MCMC(vpfit_2.model)

            vpfit_2.map.fit(iterlim=iterations, tol=i[0])
            vpfit_2.mcmc.sample(iter=i[1]*1e3, burn=1000, thin=15, progress_bar=False)
            #print("\n")
            vpfit_2.map.fit(iterlim=iterations, tol=i[2])
            vpfit_2.mcmc.sample(iter=i[3]*1e3, burn=1000, thin=15, progress_bar=False)
            #print("\n")
            vpfit_2.map.fit(iterlim=iterations, tol=i[4])
            #print "new: {}".format(vpfit_2.map.BIC)
            bpic.append((n, vpfit_2.map.BIC))
            #chi2.append((n, vpfits.VPfit.ReducedChisquared(fluxes, vpfit_2.total.value, noise, n*4+1)))
            del vpfit_2
    bpic_list.append(bpic)
    print "Time taken: {}".format(time.time() - t)
    print "Time to finish: {}".format((time.time() - t)/((j+1)/3**5))

plt.xlabel("run number")
plt.ylabel("BIC")
xmax = 5
plt.xticks(range(xmax))
plt.xlim((1, xmax))
#plt.ylim((-280, -240))
plt.grid(True)
for i, bpic in enumerate(bpic_list):
    bpic = np.array(bpic)
    plt.plot(range(2, xmax+2), stats.binned_statistic(bpic[:, 0], bpic[:, 1], bins=np.arange(1.5, xmax+.5+2, 1), statistic="median")[0])
    plt.scatter(bpic[:, 0], bpic[:, 1], label=labels[i])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig("plot.png")