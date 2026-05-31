import numpy as np
import pygeode as pyg
from scipy.integrate import solve_ivp

ga = 9.81
cp = 1003.5
T0Cel = 273.15

# Triple point of water
T0 = 273.16                      # K, temperature at the triple point
es0 = 611.655                    # Pa, saturation vapour pressure at the triple point
Lv0 = 2.501e6                    # J / kg, latent heat of vaporization of water at the triple point
dcp = 2180.

Mdry  = 0.0289644                # kg/mol, Molecular mass of dry air
Mh2o  = 0.0180153                # kg/mol, Molecular mass of water
Mo3   = 0.048                    # kg/mol, Molecular mass of water
Runiv = 8.31446262               # J/(mol K), universal gas constant
Rd    = Runiv / Mdry             # J/(K kg), gas constant for dry air
Rv    = Runiv / Mh2o             # J/(K kg), gas constant for water vapour

eps   = Mh2o / Mdry              # dimensionless, ratio of molecular mass of H2O and dry air

def dry_lapse_rate():
   return ga/cp

def moist_lapse_rate(T, r):
   return ga * (Rd * T**2 + Lv0 * r * T) / (cp * Rd * T**2 + Lv0**2 * r * eps)

def saturated_vapor_pressure(T):
# {{{
   '''Returns saturated vapor pressure in hPa at temperature T (in Kelvin).
   Uses (11) from Bolton MWR 1980. '''

   Tc = T - T0Cel
   return 611.2 * np.exp(17.67 * Tc / (Tc + 243.5))
# }}}

def saturated_vapor_pressure_a(T):
# {{{
   '''Returns saturated vapor pressure in Pa at temperature T (in Kelvin).
   Uses (13) from Ambaum QJRMS (2020). '''

   L = Lv0 - dcp * (T - T0)
   return es0 * (T0 / T)**(dcp / Rv) * np.exp(Lv0 / (Rv * T0) - L / (Rv * T))
# }}}

def volume_mixing_ratio_from_RH(T, rh):
# {{{
    Tc = T - 273.15
    e_sat =  6.112 * np.exp(17.67 * Tc / (Tc + 243.5))
    e = e_sat * (rh / 100)

    # Mass mixing ratio
    #r = eps * (e / (rh.lev - e))

    # Volume mixing ratio
    r = (e / (rh.lev - e))
    r = r.rename('H2O')
    r.atts['units'] = 'vmr'
    return r
# }}}

def T_lcl(T, r):
# {{{
    e = T.lev * r / (eps + r)
    Tl = 55. + 2840. / (3.5 * np.log(T) - np.log(e) - 4.805)
    return Tl
# }}}

def d_saturated_vapor_pressure_dT_b(T):
# {{{
   '''Returns derivative of saturated vapor pressure with respect to temperature in Pa/K at temperature T (in Kelvin).
   Uses (11) from Bolton MWR 1980. '''

   Tc = T - T0Cel
   a = 17.67
   b = 243.5

   return 611.2 * b*a / (Tc + b)**2 * np.exp(a * Tc / (Tc + b))
# }}}

def d_saturated_vapor_pressure_dT_CC(T):
# {{{
   '''Returns derivative of saturated vapor pressure with respect to temperature in Pa/K at temperature T (in Kelvin).
   Uses Clausius-Clapeyron equation assuming constant latent heat of vaporization. '''

   return Lv0 * saturated_vapor_pressure(T) / (Rv * T**2)
# }}}

def get_dry_profile(T0, p0, zs=None, on_height = True):
# {{{
   if zs is None:
      zs = np.linspace(0, 20e3, 201)

   def dlr(z, y):
      T, p = y
      dens = p / (Rd * T)
      dT = -dry_lapse_rate()
      dp = -dens * ga
      return dT, dp

   dry   = solve_ivp(dlr, [t0, p0*100.], zs)
   T = dry[:, 0]
   p = dry[:, 1] / 100.

   if on_height:
      z = pyg.Height(zs)
      T = pyg.Var((z,), values=T,  name = 'T')
      p = pyg.Var((z,), values=p,  name = 'pres')
      return pyg.asdataset([T, p])
   else:
      pres = pyg.Pres(p)
      T = pyg.Var((pres,), values=T,  name = 'T')
      z = pyg.Var((pres,), values=zs, name = 'z')
      return pyg.asdataset([T, z])
# }}}

def get_moist_profile(T0, p0, r0, zs=None, on_height = True):
# {{{
   def mlr_subsaturated(z, y):
   # {{{
      T, p, r = y
      R = Rd * (1 + r/eps)
      dens = p / (R * T)

      dp = -dens * ga
      dT = -dry_lapse_rate()
      dr = 0.

      return dT, dp, dr
   # }}}

   def mlr_saturated(z, y):
   # {{{
      T, p, r = y
      R = Rd * (1 + r/eps)
      dens = p / (R * T)

      es = saturated_vapor_pressure(T)
      rs = eps * es / p

      dp = -dens * ga
      dT = -moist_lapse_rate(T, rs)
      dsat = d_saturated_vapor_pressure_dT_b(T)
      dr = r * (dsat / es * dT + ga / (R * T))

      return dT, dp, dr
   # }}}

   # Event function for detecting lifted condensation level 
   def lcl(z, y):
   # {{{
      T, p, r = y

      # Compute saturation mixing ratio
      es = saturated_vapor_pressure(T)
      rs = eps * es / p                  

      return r - rs
   # }}}

   lcl.terminal = True
   lcl.direction = 1.

   es0 = saturated_vapor_pressure(T0)
   rs0 = eps * es0 / p0

   if zs is None:
      zs = np.linspace(0, 20e3, 201)

   T  = np.zeros(len(zs), 'd')
   p  = np.zeros(len(zs), 'd')
   r  = np.zeros(len(zs), 'd')
   rs = np.zeros(len(zs), 'd')

   if r0 < rs0:
      # If the parcel is sub-saturated, integrate to LCL
      sln = solve_ivp(mlr_subsaturated, [zs[0], zs[-1]], [T0, p0*100., r0], t_eval = zs, events = [lcl])

      i_sub = len(sln.t)
      T[:i_sub] = sln.y[0, :]
      p[:i_sub] = sln.y[1, :]
      r[:i_sub] = sln.y[2, :]

      if sln.status == 1:
         # We've hit the LCL, carry on in saturated part of profile
         z_lcl = sln.t_events[0][0]
         T_lcl, p_lcl, r_lcl = sln.y_events[0][0]

         sln2 = solve_ivp(mlr_saturated, [z_lcl, zs[-1]], [T_lcl, p_lcl, r_lcl], t_eval = zs[i_sub:])

         T[i_sub:] = sln2.y[0, :]
         p[i_sub:] = sln2.y[1, :]
         r[i_sub:] = sln2.y[2, :]
   else:
      raise ValueError('Starting parcel is supersaturated.')

   es = saturated_vapor_pressure(T)
   rs = eps * es / p

   if on_height:
      z  = pyg.Height(zs)
      T  = pyg.Var((z,), values=T,  name = 'T')
      p  = pyg.Var((z,), values=p / 100.,  name = 'pres')
      r  = pyg.Var((z,), values=r,  name = 'r')
      rs = pyg.Var((z,), values=rs, name = 'rs')
      return pyg.asdataset([T, p, r, rs])
   else:
      pres = pyg.Pres(p / 100.)
      T = pyg.Var((pres,), values=T,  name = 'T')
      z = pyg.Var((pres,), values=zs, name = 'z')
      r = pyg.Var((pres,), values=r,  name = 'r')
      rs = pyg.Var((pres,), values=rs,  name = 'rs')
      return pyg.asdataset([T, z, r, rs])
# }}}
