import numpy as np
from matplotlib import pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve

import pygeode as pyg

import column

# import adiabat
from rrtm import astr

import xarray as xr

def test_strat_mechanism(Nz=200):
# {{{
   c = column.Configuration('strat_rad.json', 'configs/')
   c.radiation['active'] = True
   c.chemistry['active'] = True
   c.photolysis['active'] = True
   c.grid['Nz'] = Nz
   print(c.grid)

   #c.radiation['zenith'] = 'fixed_specified'
   #c.radiation['solar_zenith_angle'] = 54.

   col = column.Column(c)

   # mixing ratio or mol/m-3?
   col.O2[:] = 0.21
   col.N2[:] = 0.78
   col.O1D[:] = 0
   col.O[:] = 0
   col.O3[:] = 0   # np.interp(col.zfull[::-1], -col.cfg.H * np.log(dsr.pfull[::-1] / col.cfg.p0), dsr.o3[::-1])[::-1] 
   col.H2O[:] = 0.02 / 0.622 * np.exp(-col.zfull[:]/2000) # approximate profile with 2 km scale height
   col.H2O[col.H2O < 4e-6] = 4e-6

   col.N2O[:] = 3.e-7
   col.w[:] = 0.
   col.w[5:] = 0.0001
   col.wp[:] =  0.
   col.M[:] = 1.

   #col.TSfc = 300.

   # dst = pyg.open('/local1/storage1/alm334/tuv-x/sample_photolysis_rate_constants.nc')(time=0)

   #dss = []
   #for c in [400e-6]:
   col.CO2[:] = 400e-6
   #return col
   #ts, o0 = col.solve(26*6, 600)
   ts, o0 = col.solve(144 * 100, 600)

   ds = column.to_pyg(col, ts, o0)
   #dss.append(ds)

   #pyg.showvar(ds.T, fig=3)
   return col, ds
# }}}

def plot_noy(ds):
# {{{
   pfull = pyg.Pres(1e3*pyg.exp(-ds.zfull / 7e3)[:])

   ds = ds.replace_axes(zfull = pfull)(pres = (300, 1))

   NOx = ds.NO + ds.NO2
   NOy = ds.NO + ds.NO2 + ds.HNO3 + 2*ds.N2O5

   plt.ioff()

   axn = pyg.plot.AxesWrapper()
   ax  = pyg.plot.AxesWrapper()

   pyg.vplot(1e9*ds.N2O, 'o',  axes = axn, mew = 1., ms = 5., mec = '0.4', mfc = 'w', label = 'N2O')

   axn.setp(xscale = 'log', xlim = (4, 4e2))
   axn.setp_xaxis(major_formatter = plt.LogFormatter(), \
                  major_locator = plt.LogLocator(subs = [1.]))
   axn.legend(loc = 'lower center', frameon = False, ncols = 2)
                                         
   pyg.vplot(1e9*NOx,    '--', axes = ax, lw = 2., c = '0.6', label = 'NOx')
   pyg.vplot(1e9*NOy,    '-',  axes = ax, lw = 2., c = '0.8', label = 'NOy')
                                         
   pyg.vplot(1e9*ds.NO,   'o-', axes = ax, c = 'r',  ms = 5., label = 'NO')
   pyg.vplot(1e9*ds.NO2,  'd-', axes = ax, c = 'g',  ms = 5., label = 'NO2')
   pyg.vplot(1e9*ds.HNO3, 's-', axes = ax, c = 'b',  ms = 5., label = 'HNO3')
   pyg.vplot(2e9*ds.N2O5, 'p-', axes = ax, c = 'C1', ms = 5., label = '2xN2O5')

   ax.setp(xscale = 'log', xlim = (1e-4, 3e1))
   ax.setp_xaxis(major_formatter = plt.LogFormatter(), \
                 major_locator = plt.LogLocator(subs = [1.]))

   ax.legend(loc = 'lower center', frameon = False, ncols = 2)

   ax = pyg.plot.grid([[axn, ax]])

   plt.ion()

   ax.render(4)

# }}}
