import numpy as np
from matplotlib import pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve

import pygeode as pyg

import modpac

# import adiabat
#from rrtm import astr

import xarray as xr

def test_strat_mechanism(ndays=200, Nz=200, kappa = 0.):
# {{{
   c = modpac.Configuration('strat_rad.json', '..')
   c.radiation['active'] = True
   c.chemistry['active'] = True
   c.photolysis['active'] = True
   c.grid['Nz'] = Nz

   c.dynamics['kappa_zz'] = kappa#1e-2

   print(c.dynamics['kappa_zz'])

   #c.radiation['zenith'] = 'fixed_specified'
   #c.radiation['solar_zenith_angle'] = 54.

   col = modpac.ModPAC(c)

   pfull = 1000. * np.exp(-col.zfull / col.cfg.H)
   phalf = 1000. * np.exp(-col.zhalf / col.cfg.H)
   pf = pyg.Pres(pfull)
   ph = pyg.Pres(phalf)

   ref = pyg.open('/data/QOSM/basic_state_from_rce.nc')
   prk = pyg.open('/data/QOSM/park_noy_plot.nc')

   # mixing ratio 
   col.M[:] = 1.
   col.O2[:] = 0.21
   col.N2[:] = 0.78
   col.O1D[:] = 1e-15
   col.O[:] = 1e-15
   col.O3[:] = 0.1e-6   # np.interp(col.zfull[::-1], -col.cfg.H * np.log(dsr.pfull[::-1] / col.cfg.p0), dsr.o3[::-1])[::-1] 
   col.H2O[:] = 0.02 / 0.622 * np.exp(-col.zfull[:]/2000) # approximate profile with 2 km scale height
   col.H2O[col.H2O < 6e-6] = 6e-6
   col.HO2[:] = 1e-10
   col.H[:] = 1e-10
   col.H2[:] = 1e-9
   col.H2O2[:] = 1e-11
   col.HO2NO2[:] = 1e-14

   def to_col(var, pr):
      vi = var.interpolate('pres', pr, inx = pyg.log(var.pres), outx = pyg.log(pr), d_above = 0., d_below = 0.)
      return vi[:]

   # Better idea is to solve analytically for N2O given w, a lower boundary value, and a calculation of jn2o
   col.w[:] = to_col(ref.w, ph) * 1e-3
   col.w[col.w < 0] = 6e-5

   col.w[:] = col.w[:] * (0.5 + 0.5*np.tanh((col.zhalf - 7000) / 1500))
   col.w[1] = 0.
   col.w[-1] = 0.

   #col.w[5:-5] = 0.0003
   col.wp[:] =  0. + 0j

   col.N2O[:] = to_col(prk.N2O, pf) *1e-9
   col.variables['N2O'].fixed = True

   col.O3[:] = to_col(ref.O3, pf)
   col.variables['O3'].fixed = True

   #col.N2O[:] = solve_steady_n2o(col)

   #col.w[5:-5] = 0.

   #col.TSfc = 300.

   # dst = pyg.open('/local1/storage1/alm334/tuv-x/sample_photolysis_rate_constants.nc')(time=0)

   #dss = []
   #for c in [400e-6]:
   col.CO2[:] = 400e-6
   #return col
   #ts, o0 = col.solve(26*6, 600)
   ts, o0 = col.solve(24 * ndays, 6*600, 1)
   #ts, o0 = col.solve(112, 3*600, 1)

   ds = modpac.to_pyg(col, ts, o0)
   #dss.append(ds)

   #pyg.showvar(ds.T, fig=3)
   return col, ds
# }}}

def plot_noy(ds, fig=3):
# {{{
   pfull = pyg.Pres(1e3*pyg.exp(-ds.zfull / 7e3)[:])

   ds = ds.replace_axes(zfull = pfull)(pres = (300, 1))

   prk = pyg.open('/data/QOSM/park_noy_plot.nc')

   NOx = ds.NO + ds.NO2
   NOy = ds.NO + ds.NO2 + ds.HNO3 + 2*ds.N2O5

   plt.ioff()

   axn = pyg.plot.AxesWrapper()
   ax  = pyg.plot.AxesWrapper()

   pyg.vplot(1e9*ds.N2O,  'o',  axes = axn, mew = 1., ms = 5., mec = '0.4', mfc = 'w', label = 'N2O')
   pyg.vplot(prk.N2O, 'o',  axes = axn, mew = 1., ms = 5., mec = '0.4', mfc = '0.4', label = 'N2O (Park et al.)')

   axn.setp(xscale = 'log', xlim = (4, 4e2))
   axn.setp_xaxis(major_formatter = plt.LogFormatter(), \
                  major_locator = plt.LogLocator(subs = [1.]))
   axn.legend(loc = 'lower center', frameon = False, ncols = 2)
                                         
   pyg.vplot(1e9*NOx,    '--', axes = ax, lw = 2., c = '0.6', label = 'NOx')
   pyg.vplot(1e9*NOy,    '-',  axes = ax, lw = 2., c = '0.8', label = 'NOy')
                                         
   pyg.vplot(1e9*ds.NO,   'o-', axes = ax, c = 'r',  ms = 4., label = 'NO')
   pyg.vplot(1e9*ds.NO2,  'd-', axes = ax, c = 'g',  ms = 4., label = 'NO2')
   pyg.vplot(1e9*ds.HNO3, 's-', axes = ax, c = 'b',  ms = 4., label = 'HNO3')
   pyg.vplot(2e9*ds.N2O5, 'p-', axes = ax, c = 'C1', ms = 4., label = '2xN2O5')

   pyg.vplot(prk.NOx, '--', axes = ax, lw = 1., c = '0.2')
   pyg.vplot(prk.NOy,  '-', axes = ax, lw = 1., c = '0.4')
                                     
   pyg.vplot(prk.NO,   'o', axes = ax, c = 'r',  ms = 6.)
   pyg.vplot(prk.NO2,  'd', axes = ax, c = 'g',  ms = 6.)
   pyg.vplot(prk.HNO3, 's', axes = ax, c = 'b',  ms = 6.)
   pyg.vplot(prk.N2O5, 'p', axes = ax, c = 'C1', ms = 6.)

   ax.setp(xscale = 'log', xlim = (1e-4, 3e1))
   ax.setp_xaxis(major_formatter = plt.LogFormatter(), \
                 major_locator = plt.LogLocator(subs = [1.]))

   ax.legend(loc = 'lower center', frameon = False, ncols = 2)

   ax = pyg.plot.grid([[axn, ax]])

   plt.ion()

   ax.render(fig)

# }}}

def test_n2o_column(Nz=200, kappa = 0.):
# {{{
   c = modpac.Configuration('n2o_rad.json', '..')
   c.radiation['active'] = True
   c.chemistry['active'] = True
   c.photolysis['active'] = True
   c.convection['active'] = True
   c.humidity['active'] = True
   c.grid['Nz'] = Nz

   c.dynamics['kappa_zz'] = kappa#1e-2

   print(c.dynamics['kappa_zz'])

   ref = pyg.open('/data/QOSM/basic_state_from_rce.nc')
   prk = pyg.open('/data/QOSM/park_noy_plot.nc')

   col = modpac.ModPAC(c)

   pfull = 1000. * np.exp(-col.zfull / col.cfg.H)
   phalf = 1000. * np.exp(-col.zhalf / col.cfg.H)
   pf = pyg.Pres(pfull)
   ph = pyg.Pres(phalf)

   def to_col(var, pr):
      vi = var.interpolate('pres', pr, inx = pyg.log(var.pres), outx = pyg.log(pr), d_above = 0., d_below = 0.)
      return vi[:]

   # mixing ratio
   col.M[:] = 1.
   col.O2[:] = 0.21
   col.N2[:] = 0.78
   col.O1D[:] = 0
   col.CO2[:] = 400e-6

   col.O3[:] = to_col(ref.O3, pf)

   col.N2O[:] = to_col(prk.N2O*1e-9, pf)

   col.H2O[:] = 0.02 / 0.622 * np.exp(-col.zfull[:]/2000) # approximate profile with 2 km scale height
   col.H2O[col.H2O < 4e-6] = 4e-6

   # Better idea is to solve analytically for N2O given w, a lower boundary value, and a calculation of jn2o
   col.w[:] = to_col(ref.w, ph) * 1e-3
   col.w[col.w < 0] = 1e-5
   col.w[0] = 0.
   col.w[-1] = 0.
   col.wp[:] =  0.

   col.T[:] = to_col(ref.T, pf)
   col.Tsfc = 300.

   print(col.advected)
   for name, var in col.variables.items():
      if hasattr(var, 'fixed') and var.fixed: print(name)

   ts, o0 = col.solve(8 * 1000, 3*3600)

   ds = modpac.to_pyg(col, ts, o0)

   return col, ds
# }}}

def solve_steady_n2o(col, n2o_0 = 3.26e-7, jn2o = None):
# {{{
   if jn2o is None:
      s0 = col.get_internal_state(n = 1)
      o0 = col.create_output_state(1)
      jn2o = o0.columns['jn2o'][0, :]

   #p_org = col.cfg.p0 * np.exp(-col.z_full / col.cfg.H)
   #nafull = p_org * 100 / (col.cfg.R * state.T[0, :])

   col.compute_photolysis(s0, o0, col.zfull, 0, 0)

   wfull = np.interp(col.zfull[::-1], col.zhalf[::-1], col.w[::-1])[::-1]

   dz = np.diff(col.zhalf)

   tau = np.cumsum(jn2o / np.sqrt(wfull**2 + 1e-12) * dz)
   tau -= tau[-1]

   return n2o_0 * np.exp(-tau)
# }}}

def infer_jn2o(col):
# {{{
   dz = np.diff(col.zhalf)

   ip = np.arange(col.Nz) + 1
   im = np.arange(col.Nz) - 1

   ip[-1] = col.Nz - 1
   im[0] = 0

   dz = col.zfull[ip] - col.zfull[im]
   dn2o = col.N2O[ip] - col.N2O[im]

   dn2o = (np.log(col.N2O[1:]) - np.log(col.N2O[:-1])) / dz

   return -col.w[1:-1] * dn2o
# }}}

def test_h2o_column(ndays = 200, Nz=200, kappa = 0.):
# {{{
   c = modpac.Configuration('h2o_rad.json', '..')
   c.radiation['active'] = True
   c.grid['Nz'] = Nz

   c.dynamics['kappa_zz'] = kappa#1e-2

   print(f"kappa: {c.dynamics['kappa_zz']}")

   ref = pyg.open('/data/QOSM/basic_state_from_rce.nc')
   prk = pyg.open('/data/QOSM/park_noy_plot.nc')

   col = modpac.ModPAC(c)

   pfull = 1000. * np.exp(-col.zfull / col.cfg.H)
   phalf = 1000. * np.exp(-col.zhalf / col.cfg.H)
   pf = pyg.Pres(pfull)
   ph = pyg.Pres(phalf)

   def to_col(var, pr):
      vi = var.interpolate('pres', pr, inx = pyg.log(var.pres), outx = pyg.log(pr), d_above = 0., d_below = 0.)
      return vi[:]

   # mixing ratio
   col.M[:] = 1.
   col.O2[:] = 0.21
   col.N2[:] = 0.78
   col.CO2[:] = 400e-6

   col.O3[:] = to_col(ref.O3, pf)

   col.H2O[:] = 0.02 / 0.622 * np.exp(-col.zfull[:]/2000) # approximate profile with 2 km scale height
   col.H2O[col.H2O < 4e-6] = 4e-6

   # Better idea is to solve analytically for N2O given w, a lower boundary value, and a calculation of jn2o
   col.w[:] = to_col(ref.w, ph) * 1e-3
   col.w[col.w < 0] = 1e-5
   col.w[0] = 0.
   col.w[-1] = 0.
   col.wp[:] =  0.

   col.T[:] = to_col(ref.T, pf)
   col.Tsfc = 300.

   print(col.advected)
   for name, var in col.variables.items():
      if hasattr(var, 'fixed') and var.fixed: print(name)

   ts, o0 = col.solve(8 * ndays, 3*3600)

   ds = modpac.to_pyg(col, ts, o0)

   return col, ds
# }}}

def profile_run():
   import cProfile
   cProfile.run('test_strat_mechanism()', 'strat_timings')


