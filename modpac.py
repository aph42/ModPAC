import json
import datetime

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

from rrtm import rrtmg

import musica
import musica.mechanism_configuration as mc
import musica.tuvx.vTS1

from modpac import astr

class Configuration():
   def __init__(self, config_file, config_path):
# {{{
      self.config_file = config_file
      self.config_path = config_path

      # Read global configuration file
      with open(config_path + config_file, 'r') as f:
         d = json.load(f)

      self.name = d['name']
      self.version = d['version']

      # Initialize constants
      for k, v in d['constants'].items():
         self.__dict__[k] = v
         
      self.grid = d['grid']
      self.dynamics = d['dynamics']
      self.radiation = d['radiation']
      self.chemistry = d['chemistry']
      self.photolysis = d['photolysis']
      self.convection = d['convection']
      self.humidity = d['humidity']
# }}}

class ScalarVariable():
   def __init__(self, name, unit, initial_value):
   # {{{
      self.name = name
      self.value =  initial_value
      self.unit = unit
   # }}}

class ColumnVariable():
   def __init__(self, name, unit, Nz, initial_value, prognostic = False, output = False):
   # {{{
      self.name = name
      self.Nz = Nz

      if type(initial_value) == np.ndarray:
         dtype = initial_value.dtype
      else:
         dtype = type(initial_value)
      self.values = np.ones(Nz, dtype)
      self.values[:] = initial_value
      self.unit = unit
      self.prognostic = prognostic
      self.output = output
   # }}}

class SpeciesVariable(ColumnVariable):
   def __init__(self, name, unit, Nz, initial_value, advect = False, prognostic = False, output = False, **properties):
   # {{{
      ColumnVariable.__init__(self, name, unit, Nz, initial_value, prognostic, output)

      self.advect = advect

      if advect:
         self.surface_flux = 0.
         self.TOA_flux = 0.

      self.properties = properties
   # }}}

class State():
   def __init__(self, columns, scalars, steps = 1):
   # {{{
      self.__dict__['columns'] = {}
      self.__dict__['scalars'] = {}

      for name, c in columns.items():
         self.columns[name] = np.zeros((steps, c.Nz))
         self.columns[name][:, :] = c.values.reshape(1, -1)

      for name, s in scalars.items():
         self.scalars[name] = np.zeros(steps)
         self.scalars[name][:] = s.value
   # }}}

   def __setattr__(self, name, value):
   # {{{
      if name in self.columns:
         self.columns[name][:] = value

      elif name in self.scalars:
         self.scalars[name][:] = value

      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def __getattr__(self, name):
   # {{{
      if name in self.columns:
         return self.columns[name][:]
      elif name in self.scalars:
         return self.scalars[name][:]
      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

def interpolate_matrix(x_new, x_old, method = 'linear'):
   # {{{
      ''' Constructs a CSR sparse matrix that, when applied to a vector
      of quantities defined at locations x_old, yields interpolated values
      at x_new. x_old and x_new must be sorted. '''

      ip = x_old.searchsorted(x_new)
      iL = np.where(ip == 0.)[0]
      iR = np.where(ip == len(x_old))[0]

      ip[iL] = 1
      ip[iR] = len(x_old) - 1

      i0 = ip - 1

      dx = x_old[ip] - x_old[i0]
      tn = (x_new - x_old[i0]) / dx
      tn[iL] = 0.
      tn[iR] = 1.

      otn = 1 - tn

      N = len(x_new)
      M = len(x_old)

      if method == 'linear':
         order = 2
         entries = np.zeros(N * order, 'd')
         indptr = order * np.arange(N + 1)
         cols = np.zeros(N * order, 'i')
         cols[::order]  = i0
         cols[1::order] = ip

         entries[::order] = otn
         entries[1::order] = tn

         L = sparse.csr_array((entries, cols, indptr), shape = (N, M))
      elif method == 'cubic':
         Dl = np.zeros(M - 1)
         Dc = np.zeros(M)
         Dr = np.zeros(M - 1)

         Dr[0] = 1 / (x_old[-1] - x_old[-2])
         Dc[-1] = 1 / (x_old[1] - x_old[0])
         Dr[1:] = 1 / (x_old[2:] - x_old[:-2])
         Dl[:-1] = -Dr[1:]
         Dc[0] = -Dr[0]
         Dl[-1] = -Dc[-1]

         Del = sparse.diags_array([Dl, Dc, Dr], offsets = [-1, 0, 1], shape = (M,M), format = 'csr')

         order = 2
         Cdata = np.zeros(N * order, 'd')
         Ddata = np.zeros(N * order, 'd')
         indptr = order * np.arange(N + 1)
         cols = np.zeros(N * order, 'i')
         cols[::order]  = i0
         cols[1::order] = ip

         Cdata[::order] = otn**3 + 3*otn**2*tn
         Cdata[1::order] = tn**3 + 3*tn**2*otn

         Ddata[::order] = dx*otn**2*tn
         Ddata[1::order] = -dx*tn**2*otn

         C = sparse.csr_array((Cdata, cols, indptr), shape = (N, M))
         D = sparse.csr_array((Ddata, cols, indptr), shape = (N, M))

         L = C + D @ Del

         N = L.sum(0).reshape(-1, 1)
         L = L / N
      else:
         raise ValueError(f'Unrecognized method {method}.')

      return L
   # }}}

class ModPAC():
   '''  Modular Photochemistry in an Atmospheric Column (ModPAC)
        A column model of the atmosphere, including radiative transfer, photochemistry using
        the MUSICA and TUVx components from NCAR, and vertical advection.'''
   def __init__(self, configuration):
   # {{{
      self.__dict__['grid'] = {}

      self.__dict__['variables'] = {}
      self.__dict__['scalars'] = {}

      self.__dict__['output_variables'] = {}

      self.__dict__['cfg'] = configuration

      # Initialize grid
      self.initialize_grid(**self.cfg.grid)

      # Initialize chemistry
      self.initialize_chemistry(**self.cfg.chemistry)

      # Initialize dynamical quantities
      self.initialize_dynamics(**self.cfg.dynamics)

      # Initialize radiation
      self.initialize_radiation(**self.cfg.radiation)

      # Initialize photolysis 
      self.initialize_photolysis(**self.cfg.photolysis)

      # Initialize convection
      self.initialize_convection(**self.cfg.convection)

      # Initialize humidity
      self.initialize_humidity(**self.cfg.humidity)

    # }}} 

   def __setattr__(self, name, value):
   # {{{
      if name in self.grid:
         self.grid[name].values[:] = value

      elif name in self.variables:
         self.variables[name].values[:] = value

      elif name in self.scalars:
         self.scalars[name].value = value

      elif name in self.output_variables:
         self.output_variables[name].values[:] = value

      elif name in self.__dict__.keys():
         raise ValueError(f'{name} is read-only.')

      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def __getattr__(self, name):
   # {{{
      if name in self.grid:
         return self.grid[name].values[:]
      elif name in self.variables:
         return self.variables[name].values[:]
      elif name in self.scalars:
         return self.scalars[name].value
      elif name in self.output_variables:
         return self.output_variables[name].values[:]
      elif name in self.__dict__.keys():
         return self.__dict__[name]
      else:
         raise ValueError(f'{self} has no attribute {name}.')
   # }}}

   def initialize_var(self, name, unit, Nz, initial_value, grid = False):
   # {{{
      if grid:
         if name in self.grid:
            raise ValueError(f'{name} has already been initialized.')
         
         self.grid[name] = ColumnVariable(name, unit, Nz, initial_value)
      else:
         if name in self.variables:
            raise ValueError(f'{name} has already been initialized.')
         
         self.variables[name] = ColumnVariable(name, unit, Nz, initial_value)
   # }}}

   def add_output(self, name, unit, Nz):
   # {{{
      if name in self.output_variables:
         raise ValueError(f'{name} has already been defined as an output variable.')
      
      self.output_variables[name] = ColumnVariable(name, unit, Nz, 0., output = True)
   # }}}

   def initialize_scalar(self, name, unit, initial_value):
   # {{{
      if name in self.scalars:
         raise ValueError(f'{name} has already been initialized.')
      
      self.scalars[name] = ScalarVariable(name, unit, initial_value)
   # }}}

   def initialize_grid(self, *, spacing = 'log_pressure_equal', Nz = 200, p_top = 0.1, **kwargs):
   # {{{
      ''' Initialize the column grid. The levels must run from the top of the atmosphere down
      (increasing pressure). The grid spacing can be set in various ways based on the choice
      of the argument `spacing`. Possible values include
       - 'log_pressure_equal' (default). Specify p_top, the pressure at the top of the domain in hPa, and
            the number of levels.
       - 'specified_pressure'. Provide arrays for phalf and pfull. '''

      # Grid must run from the top of the atmosphere down for RRTMG to work
      if spacing == 'log_pressure_equal':
         p_bot = self.cfg.p0

         z_top = -self.cfg.H * np.log(p_top / self.cfg.p0) 
         z_bot = 0.
         Lz = z_top - z_bot

         zhalf = np.linspace(z_top, z_bot, Nz + 1)
         phalf = self.cfg.p0 * np.exp(-zhalf / self.cfg.H)
         pfull = np.sqrt(phalf[:-1]*phalf[1:])
         zfull = -self.cfg.H * np.log(pfull / self.cfg.p0)

         #self.__dict__['dz'] = z_top / (self.Nz + 1)

      elif spacing == 'specified_pressure':
         # Full and half-levels must be specified
         pfull = kwargs.pop('pfull')
         phalf = kwargs.pop('phalf')
         zfull = -self.cfg.H * np.log(pfull / self.cfg.p0)
         zhalf = -self.cfg.H * np.log(phalf / self.cfg.p0)

         if len(phalf) != len(pfull) + 1:
            raise ValueError('There must be one more half level (phalf) than full levels (pfull)')

         Nz = len(pfull)
         p_top = phalf[0]
         p_bot = phalf[-1]
         z_top = zhalf[0]
         z_bot = zhalf[-1]
      else:
         raise ValueError(f"Unrecognized grid spacing option: {spacing}")

      if p_top > p_bot:
         raise ValueError("Pressure levels must be increasing (from top of the atmosphere down).")

      # Grid parameters
      self.__dict__['Nz'] = Nz
      self.__dict__['p_top'] = p_top
      self.__dict__['p_bot'] = p_bot
      self.__dict__['z_top'] = z_top
      self.__dict__['z_bot'] = z_bot

      self.initialize_var('zhalf', 'm', self.Nz + 1, zhalf, grid = True)
      self.initialize_var('zfull', 'm', self.Nz, zfull, grid = True)
      self.initialize_var('phalf', 'Pa', self.Nz + 1, phalf, grid = True) 
      self.initialize_var('pfull', 'Pa', self.Nz, pfull, grid = True) # DEBUG: pfull inherits units of cfg.p0 from json file, default hPa

      self.initialize_var('vmask', '1', Nz, 1., grid = True)
   # }}}

### Methods related to dynamics/advection
   def initialize_dynamics(self, *, active = True):
   # {{{
      self.__dict__['do_dynamics'] = active

      self.initialize_var('T', 'K', self.Nz, 300.) # Prognostic, advected

      self.initialize_var('w', 'm s-1', self.Nz + 1, 0.02) # not prognostic
      self.w[0] = 0.
      self.w[self.Nz] = 0.

      self.initialize_var('wp', 'm s-1', self.Nz + 1, 0j) # not prognostic
      self.wp[0] = 0.
      self.wp[self.Nz] = 0.

      # Conversion factor from temperature to potential temperature
      exner = (self.pfull / self.cfg.p0)**(-self.cfg.Rd / self.cfg.cp)

      self.initialize_var('Exner', '1', self.Nz, exner, grid = True)

      self.initialize_scalar('omega', 'd-1', 2 * np.pi / (86400. * 840.))

      # Reference grid for semi-lagrangian advection interpolation
      #self.__dict__['zadv'] = np.concatenate([[z_bot], zfull[::-1], [z_top]])
      self.__dict__['zadv'] = self.zfull[::-1]
   # }}}


   def get_courant(self, dt):
   # {{{ 
      dz = np.min(np.absolute(np.diff(self.zhalf)))
      wmax = np.max(np.absolute(self.w) + np.absolute(self.wp))

      return wmax * dt / dz
   # }}}

   def get_origins(self, state, j_now, j_new, dt, I = 2):
   # {{{
      dth = dt / 2.

      dz = self.zhalf[1:] - self.zhalf[:-1]

      # Destinations
      r_dest = self.zfull

      # Interpolate velocity at future time
      wF = 0.5 * (state.w[j_new, 1:] + state.w[j_new, :-1])
      aF = wF * (state.w[j_new, 1:] - state.w[j_new, :-1]) / dz

      # Future half of trajectory
      c2 = dth * wF - 0.5 * dth**2 * aF

      # First guess at origin points
      r_half = r_dest - c2
      zorg = r_half

      # Iterate estimate of past half of trajectory
      for i in range(I):
         # Interpolate velocity and acceleration to origin points
         aS = 0.5 * (state.w[j_now, 1:] + state.w[j_now, :-1]) * (state.w[j_now, 1:] - state.w[j_now, :-1]) / dz
         wS = np.interp(zorg[::-1], self.zhalf[::-1], state.w[j_now, ::-1])[::-1]
         aS = np.interp(zorg[::-1], self.zfull[::-1], aS[::-1])[::-1]

         # Past half of trajectory
         c1 = dth * wS + 0.5 * dth**2 * aS
         zorg = r_half - c1
      
      return zorg
   # }}}

   def build_advection_matrix(self, z_org):
   # {{{
      ''' Constructs matrices to build weights for advection interpolation. 
       Returns a tuple of three matrices, C, D, and Del; the weights for shape-preserving
       interpolation require a further diagonal matrix T to eliminate overshoot and can
       be calculated as C + D @ T @ Del.'''

      ip = self.zadv.searchsorted(z_org)
      iL = np.where(ip == 0.)[0]
      iR = np.where(ip == len(self.zadv))[0]

      ip[iL] = 1
      ip[iR] = len(self.zadv) - 1

      i0 = ip - 1

      dx = self.zadv[ip] - self.zadv[i0]
      tn = (z_org - self.zadv[i0]) / dx
      tn[iL] = 0.
      tn[iR] = 1.

      otn = 1 - tn

      N = len(z_org)
      M = len(self.zadv)

      Dl = np.zeros(M - 1)
      Dc = np.zeros(M)
      Dr = np.zeros(M - 1)

      Dx = np.diff(self.zadv)

      Dr[1:  ] =  1 / (2 * Dx[:-1])
      Dc[1:-1] = (Dx[1:] - Dx[:-1]) / (2 * Dx[1:] * Dx[:-1])
      Dl[ :-1] = -1 / (2 * Dx[1:])

      Del = sparse.diags_array([Dl, Dc, Dr], offsets = [-1, 0, 1], shape = (M,M), format = 'csr')

      order = 2
      Cdata = np.zeros(N * order, 'd')
      Ddata = np.zeros(N * order, 'd')
      indptr = order * np.arange(N + 1)
      cols = np.zeros(N * order, 'i')
      cols[::order]  = i0
      cols[1::order] = ip

      Cdata[::order] = otn**3 + 3*otn**2*tn
      Cdata[1::order] = tn**3 + 3*tn**2*otn

      Ddata[::order] = dx*otn**2*tn
      Ddata[1::order] = -dx*tn**2*otn

      C = sparse.csr_array((Cdata, cols, indptr), shape = (N, M))
      D = sparse.csr_array((Ddata, cols, indptr), shape = (N, M))

      return C, D, Del
   # }}}

   def advect_quantity(self, C, D, Del, X):
   # {{{
      ''' Carry out advection of a given field X, given the components of the
      weights matrix C, D, and Del from build_advection_matrix().  Calculates
      shape-preserving modifications for this field. Normalization (mass
      conservation) is turned off for accuracy reasons. Flux boundary
      conditions are not yet implemented. The array X is oriented in increasing
      height.'''

      # Compute shape-preserving modifications
      T = np.ones(self.Nz)

      def fmt(a): return ' '.join([f'{d:5.1f}' for d in a])

      # One-sided estimates at every grid point
      d0 = (X[1:] - X[:-1]) / (self.zadv[1:] - self.zadv[:-1])

      # Initial derivatives for the interpolation splines
      m = np.zeros(self.Nz)
      m[1:-1] = 0.5 * (d0[1:] + d0[:-1])
      m[0] = d0[0]
      m[-1] = d0[-1]

      # Find indices of local extrema, and indices of complement.
      # Adjacent indices are tested for overshoot, so last element is omitted in latter.
      ext = (d0[1:] * d0[:-1] <= 0.)
      exti = np.where(ext)[0] + 1
      extn = np.concatenate([[0], np.where(~ext)[0] + 1])

      # Set slope at any extrema to zero
      m[exti] = 0
      T[exti] = 0.

      # Where the quantity tau < 1, we need to rescale the slopes
      tau = 3 * np.abs(d0) / np.sqrt(m[:-1]**2 + m[1:]**2 + 1e-32)
      nmt = np.where(tau < 1)[0]

      T[nmt]     *= tau[nmt]
      T[nmt + 1] *= tau[nmt]

      # Convert to matrix to incorporate into interpolation operator
      T = sparse.diags_array([T], offsets = [0], shape = (self.Nz,self.Nz), format = 'csr')

      # Compute full interpolation operator
      L = (C + D @ T @ Del)

      # Normalize weights (this would enforce mass conservation, but 
      # comes at the cost of significant loss of accuracy, so its left off)
      #N = L.sum(0)
      #iN = np.where(N > 0.3)[0]
      #L[1:-1, :] = L[1:-1, :] / N[1:-1].reshape(-1, 1)

      # Apply interpolation
      return L @ X
   # }}}

   def step_advection(self, state, z_org, j_old, j_now, dt):
   # {{{
      C, D, Del = self.build_advection_matrix(z_org[::-1])

      # Construct interpolation matrix for potential temperature 
      # (no shape-preserving adjustments are made, though we could think about that)
      L = C + D @ Del

      # Convert temperature to potential temperature
      Theta = state.T[j_old, :] * self.Exner

      # Advect potential temperature then convert back to temperature
      state.T[j_now, :] = (L @ Theta[::-1])[::-1] / self.Exner

      for s in self.advected:
         v = state.columns[s]
         v[j_now, :] = self.advect_quantity(C, D, Del, v[j_old, ::-1])[::-1]
   # }}}

### Methods related to radiative transfer
   def initialize_radiation(self, *, active = True, scon = 1368.22, zenith = 'fixed_specified', **kwargs):
   # {{{
      self.__dict__['do_radiation'] = active

      # Astronomical settings
      self.__dict__['scon'] = scon
      self.__dict__['zenith'] = zenith

      self.initialize_scalar('Tsfc', 'K', 300.)
      self.initialize_scalar('emissivity', '1', 0.99)
      self.initialize_scalar('albedo', '1', 0.3)
      self.initialize_scalar('solar_zenith_angle', 'deg', 0.)

      # Initialize zenith angle
      if zenith == 'fixed_specified':
         # Zenith angle fixed and explicitly set
         self.solar_zenith_angle = kwargs.get('solar_zenith_angle', 0.)

      elif zenith == 'fixed_computed':
         # Zenith angle fixed, computed from latitude and initial date and (local) time

         self.__dict__['initial_date'] = kwargs.get('initial_date', '2000-01-01')
         self.__dict__['local_hour'] = kwargs.get('local_hour', 12.)
         self.__dict__['latitude'] = kwargs.get('latitude', 0.)

         n = astr.date_to_n(self.initial_date)
         declination = astr.declination(n)

         self.solar_zenith_angle = astr.zenith_from_declination(self.latitude, declination, local_hour)

      elif zenith == 'diurnal_cycle':
         # Zenith angle goes through fixed diurnal cycle, appropriate to given latitude and initial date
         self.__dict__['initial_date'] = kwargs.get('initial_date', '2000-01-01')
         self.__dict__['latitude'] = kwargs.get('latitude', 0.)

         n = astr.date_to_n(self.initial_date)
         declination = astr.declination(n)
         self.__dict__['declination'] = declination
      else:
         raise ValueError(f"Zenith option '{zenith}' unrecognized.")

      # Set up output options from radiation
      self.add_output('lw_uflx', 'W m-2', self.Nz + 1) 
      self.add_output('lw_dflx', 'W m-2', self.Nz + 1) 
      self.add_output('sw_uflx', 'W m-2', self.Nz + 1) 
      self.add_output('sw_dflx', 'W m-2', self.Nz + 1) 

      self.add_output('lw_hr', 'K d-1', self.Nz) 
      self.add_output('sw_hr', 'K d-1', self.Nz) 

      # Initialize rrtmg
      rrtmg.init(self.cfg.cp)

      self.initialize_var('dyn_hr', 'K d-1', self.Nz, 0.)
   # }}}

   def compute_radiation(self, state, output, j_now, i_out):
   # {{{
      # Helper function to reshape grid arrays
      def _g(v): return np.asfortranarray(v.reshape(1, -1).copy(), 'd')

      # Helper function to reshape column arrays
      def _c(v): return np.asfortranarray(v[j_now, :].reshape(1, -1).copy(), 'd')

      # Helper function to reshape scalar quantities
      def _s(v): return np.asfortranarray(np.array(v[j_now:j_now + 1]), 'd')

      pfull = _g(self.pfull)
      phalf = _g(self.phalf)

      T   = _c(state.T)
      CO2 = _c(state.CO2)
      O3  = _c(state.O3 )
      H2O = _c(state.H2O)

      TSfc = _s(state.Tsfc)
      Emis = _s(state.emissivity)
      alb  = _s(state.albedo)

      cosz = np.cos(np.deg2rad(np.min([90., state.solar_zenith_angle[j_now]])))
      cosz = np.asfortranarray(cosz)

      lw = rrtmg.rrtmg_lw(pfull, phalf, \
                          T, TSfc, Emis, \
                          CO2,  H2O,  O3)

      output.lw_uflx[i_out, :] = lw['uflxlw'][0, :]
      output.lw_dflx[i_out, :] = lw['dflxlw'][0, :]
      output.lw_hr[i_out, :]   = lw['lwhr'][0, :]

      #sw = rrtmg.rrtmg_sw_dm(pfull, phalf, \
            #T, TSfc,  self.scon, 4, \
            #alb, lat, dec, \
            #CO2, H2O, O3)

      sw = rrtmg.rrtmg_sw(pfull, phalf, \
            T, TSfc,  self.scon, \
            cosz, alb, \
            CO2, H2O, O3)

      output.sw_uflx[i_out, :] = sw['uflxsw'][0, :]
      output.sw_dflx[i_out, :] = sw['dflxsw'][0, :]
      output.sw_hr[i_out, :]   = sw['swhr'][0, :]
   # }}}

   def set_zenith_angle(self, state, j_now, t):
   # {{{
      if self.zenith == 'diurnal_cycle':
         local_hour = np.mod(t / 3600., 24.)
         state.solar_zenith_angle[j_now] = astr.zenith_from_declination(self.latitude, self.declination, local_hour)
   # }}}

### Methods related to chemistry
   def initialize_chemistry(self, *, mechanism, active = True, **kwargs):
   # {{{  
      self.__dict__['do_chemistry'] = active

      # Regardless of whether chemistry is active, read in the mechanism
      # to initialize species
      parser = mc.Parser()
      mechanism_file = self.cfg.config_path + mechanism + '.json'
      self.__dict__['mechanism'] = parser.parse(mechanism_file)

      species_list = []
      advected_list = []

      for sp in self.mechanism.species:
         properties = {'molecular_weight': sp.molecular_weight_kg_mol}
         properties.update(sp.other_properties)

         name = sp.name
         species_list.append(name)

         advect = properties.pop('__do advect', False)
         if advect == 'true': 
            advect = True
            advected_list.append(name)
         else: 
            advect = False

         if name in self.variables:
            raise ValueError(f'{name} has already been initialized.')
         
         self.variables[name] = SpeciesVariable(name, 'vmr', self.Nz, 0., advect, **properties)

      self.__dict__['species'] = species_list
      self.__dict__['advected'] = advected_list

      if active:
         # We only need the solver if chemistry is active
         self.__dict__['MICMsolver'] = musica.MICM(mechanism = self.mechanism, solver_type = musica.SolverType.rosenbrock_standard_order)
         self.__dict__['MICMstate'] = self.MICMsolver.create_state(self.Nz)
   # }}}

   def step_chemistry(self, state, z_org, j_now, dt):
   # {{{
      # Update MICM state object with temperatures and pressures
      p_org = self.cfg.p0 * np.exp(-z_org / self.cfg.H)
      self.MICMstate.set_conditions(state.T[j_now, :], 100.*p_org)

      nafull = p_org * 100 / (self.cfg.R * state.T[j_now, :])

      # For now update the concentrations manually

      # This will be more efficient if we structure the column
      # state vector to have a compatible memory structure

      mstate = self.MICMstate.get_internal_state()
      stride = mstate.concentration_strides()[0]
      sp = self.MICMstate.get_species_ordering()
      for s, i in sp.items():
         # convert from vmr to mol m-3
         v = musica._musica.VectorDouble(state.columns[s][j_now, :] * nafull)
         mstate.concentrations[i::stride] = v
         
      self.MICMsolver.solve(self.MICMstate, dt)

      # Read out resulting concentrations
      for s, i in sp.items():
         # convert back from mol m-3 to vmr
         state.columns[s][j_now, :] = mstate.concentrations[i::stride] / nafull
   # }}}

   def initialize_photolysis(self, *, mechanism, mapping = {}, active=True):
# {{{
      # tuv-x height coordinates are bottom up
      self.__dict__['do_photolysis'] = active

      if not active: 
         # Nothing to initialize
         return

      self.__dict__['micm_to_tuvx'] = mapping.copy()
      #{'jO2':'jo2_b','jO3->O':'jo3_b','jO3->O1D':'jo3_a'}

      for key in self.micm_to_tuvx:
          self.add_output(self.micm_to_tuvx[key], 's-1', self.Nz) 
       
      # initialize photolysis 
      self.__dict__['tuvx_mechanism_file'] = musica.utils.find_config_path() + '/tuvx/' + mechanism + '.json'

      # Set up grids
      grids = musica.tuvx.GridMap()
        
      heights = musica.tuvx.grid.Grid(name="height", units="km", num_sections=self.Nz)
      heights.edges = self.zhalf[::-1]/1000. 
      heights.midpoints = self.zfull[::-1]/1000.
      
      grids["height", "km"] = heights
      grids["wavelength", "nm"] = musica.tuvx.vTS1.wavelength_grid()
    
      # Set up profiles
      profiles = musica.tuvx.ProfileMap()
      profiles["air", "molecule cm-3"] = musica.tuvx.vTS1.profile("air", grids["height", "km"])
      profiles["O3", "molecule cm-3"] = musica.tuvx.vTS1.profile("O3", grids["height", "km"])
      profiles["O2", "molecule cm-3"] = musica.tuvx.vTS1.profile("O2", grids["height", "km"])
      profiles["temperature", "K"] = musica.tuvx.vTS1.profile("temperature", grids["height", "km"])
      profiles["surface albedo", "none"] = musica.tuvx.vTS1.profile("surface albedo", grids["wavelength", "nm"])
      profiles["extraterrestrial flux", "photon cm-2 s-1"] = musica.tuvx.vTS1.profile(
            "extraterrestrial flux", grids["wavelength", "nm"]
        )
        
      # Set up radiators
      radiators = musica.tuvx.RadiatorMap() # Note: radiators automatically includes air, O2, and O3 without being specified
      radiators["aerosol"] = musica.tuvx.vTS1.radiator("aerosol", grids["height", "km"], grids["wavelength", "nm"])
       
      # Create TUV-x instance with v5.4 configuration file
      self.__dict__['tuvx'] = musica.tuvx.TUVX(
            grid_map     = grids,
            profile_map  = profiles,
            radiator_map = radiators,
            config_path  = self.tuvx_mechanism_file,
        )
# }}}
        
   def compute_photolysis(self, state, output, z_org, j_new, i_out):
# {{{
      # update ozone and temperature, then calculate photolysis rates using TUV-x
      # TUV-x height coordinates are bottom-up  

      def full_to_half(v): return np.interp(self.zhalf, self.zfull, v)
    
      # get the vertical profiles
      grids = self.tuvx.get_grid_map()
      profiles = self.tuvx.get_profile_map()
      
      # update the temperature profile
      T_profile = profiles["temperature", "K"]
      T_profile.midpoint_values =  state.T[j_new,::-1] 
      T_profile.edge_values = full_to_half(state.T[j_new,:])[::-1] 
      
      # convert from vmr to molecules cm-3
      n_air = self.pfull * 100 / (self.cfg.R * state.T[j_new,:])
      o3_mid = state.O3[j_new,:] * n_air * self.cfg.Av * 1e-6

      # update the ozone profile
      o3_profile = profiles["O3", "molecule cm-3"]
      o3_profile.midpoint_values = o3_mid[::-1]
      o3_profile.edge_values = full_to_half(o3_mid)[::-1]
      #o3_profile.edge_values = full_to_half(state.O3[j_new,:])[::-1] * self.nahalf[::-1] * self.cfg.Av * 1e-6 # molec cm-3
      o3_profile.calculate_layer_densities(grids["height", "km"]) # provide the height grid for layer thicknesses
      
      # calculate photolysis rates
      sza = np.deg2rad(np.min([90, state.solar_zenith_angle[j_new]]))
      tuvx_output = self.tuvx.run(sza = sza, \
                                  earth_sun_distance = 1.0)
       
      # update photolysis rates
      for micm_reaction in self.micm_to_tuvx.keys():
         micm_key = f'PHOTO.{micm_reaction}'
         tuvx_key = self.micm_to_tuvx[micm_reaction]
         jval = tuvx_output['photolysis_rate_constants'].sel(reaction=tuvx_key)
         jval = jval.interp(vertical_edge = z_org/1000.).values

         # Set rates in MICM
         self.MICMstate.set_user_defined_rate_parameters({micm_key:jval})

         # Save rates for output
         getattr(output,tuvx_key)[i_out, :] = jval
# }}}


### Methods related to convection/convective adjustment
   def initialize_convection(self, *, active = True, lapse_rate = 'constant'):
   # {{{
      self.__dict__['do_convection'] = active
      self.__dict__['lapse_rate']    = lapse_rate # 'constant' or 'moist'

      self.initialize_var('T_conv', 'K', self.Nz, 300.) # Moist adiabatic temperature profile from Tsfc

      self.T_conv[:] = self.calc_moist_adiabat()

   def moist_adiabatic_lapse_rate(self,T,ws):
       if self.lapse_rate == 'moist':
           # dT/dz for a moist adiabat as a function of T (Kelvin) and ws (saturation mass mixing ratio)
           dTdz_mlr = -self.cfg.g0 / self.cfg.cp * (1 + self.cfg.Lv * ws / (self.cfg.Rd*T))/(1 + self.cfg.Lv**2*ws/(self.cfg.Rv*self.cfg.cp*T**2))
       elif self.lapse_rate == 'constant':
           # hard adjustment to a constant lapse rate
           dTdz_mlr = -6.5e-3 # dT/dz = -6.5 K/km 
           
       else:
         raise ValueError(f"Lapse rate option '{self.lapse_rate}' unrecognized.")
       
       return dTdz_mlr
    
   def calc_moist_adiabat(self):
       # calculate a moist adiabat as a function of altitude
       # begin at the specified surface temperature Tsfc and surface pressure and integrate dT/dz|mlr upwards
       print(self.Tsfc)
       ws_0 = (self.cfg.Rd / self.cfg.Rv) * self.calc_saturation_vmr(self.Tsfc,self.cfg.p0) # mmr at surface
       
       T_conv = np.zeros(self.Nz)
       T_conv[-1] = self.Tsfc
       dTdz_mlr_zi = self.moist_adiabatic_lapse_rate(self.Tsfc,ws_0)
       
       for zi in np.arange(self.Nz-2,0,step=-1):
           # integrate the moist adiabat from the surface upwards
           dz = self.zfull[zi] - self.zfull[zi+1]
           T_conv[zi] = T_conv[zi+1] + dTdz_mlr_zi * dz
           ws_zi = (self.cfg.Rd / self.cfg.Rv) * self.calc_saturation_vmr(T_conv[zi],self.pfull[zi]) # mmr
           dTdz_mlr_zi = self.moist_adiabatic_lapse_rate(T_conv[zi],ws_zi)

       T_conv[np.isnan(T_conv)] = 0.
       return T_conv

    
   def convective_adjustment(self,state,j_now):
       # convective adjustment
       # after Thuburn and Craig (2002) in which T_conv sets the minimum temperature
       # T_conv is calculated as a moist adiabat
       state.T[j_now,:] = np.maximum(state.T[j_now,:],self.T_conv)
    

### Methods related to humidity (remove supersaturation, tropospheric RH)    
   def initialize_humidity(self, *, active = True, RH_trop = 0.7, z_trop = 10000):
      self.__dict__['do_humidity'] = active
 
      self.initialize_var('RH_troposphere','',self.Nz, 0.) # Relative humidity
       
      self.RH_troposphere[:][self.zfull<=z_trop] = RH_trop # relative humidity enforced below z_trop
      self.RH_troposphere[:][self.zfull>z_trop] = np.nan # no RH constraint above z_trop
    
   def relax_humidity(self,state,j_now):
       # This function does 2 things (both of which depend on saturation_vmr):
       # 1) remove water vapor in excess of supersaturation
       # 2) enforce the specified relative humidity profile from self.RH_troposphere

       saturation_vmr = self.calc_saturation_vmr(state.T[j_now,:],self.pfull)
 
       # remove water vapor in excess of supersaturation
       state.H2O[j_now,:] = np.minimum(state.H2O[j_now,:],saturation_vmr)

       # enforce RH
       idx_trop = ~np.isnan(self.RH_troposphere) # indices to overwrite
       state.H2O[j_now,idx_trop] = (saturation_vmr*self.RH_troposphere)[idx_trop]
          
   def calc_saturation_vmr(self,T,p):
       # calculate the saturation volume mixing ratio of water vapor
       e_s = self.cfg.es_0 * np.exp(17.625 * (T-self.cfg.T0Cel)/(T-self.cfg.T0Cel+243.04)) # hPa
       
       saturation_vmr = e_s / p # vmr (units must align between e_s and pfull [e.g., hPa])
       
       return saturation_vmr
    
### Methods related to solver
   def get_internal_state(self, n = 1):
   # {{{
      return State(self.variables, self.scalars, n)
   # }}}

   def create_output_state(self, n = 1):
   # {{{
      return State(self.variables | self.output_variables, self.scalars, n)
   # }}}

   def save_state(self, state, output, j_state, i_out):
   # {{{
      for c in state.columns: 
         output.columns[c][i_out, :] = state.columns[c][j_state, :]

      for s in state.scalars: 
         output.scalars[s][i_out] = state.scalars[s][j_state]
   # }}}

   def update_externals(self, state, j_now, t):
   # {{{
      # Update periodic component of upwelling
      state.w[j_now, :] = self.w + np.real(self.wp * np.exp(1j * self.omega * t))

      # Update zenith angle
      self.set_zenith_angle(state, j_now, t)
   # }}}

   def solve(self, nsteps, dt, output_freq = 1):
   # {{{
      # Output grid
      #nout   = int(nsteps / output_freq) + 1
      nout   = int(np.ceil(nsteps / output_freq)) + 1
      times  = np.arange(nout) * dt * output_freq

      s0 = self.get_internal_state(n = 2)
      o0 = self.create_output_state(nout)

      i = 0
      i_step = 0
      i_out = 0

      j_old, j_now = 0, 1

      # Calculate relevant rates for initial conditions
      # (only used for output)
      self.update_externals(s0, j_old, 0. * dt)

      if self.do_photolysis:
         self.compute_photolysis(s0, o0, self.zfull, j_old, i_out)

      if self.do_radiation:
         self.compute_radiation(s0, o0, j_old, i_out)

      self.save_state(s0, o0, j_old, i_out)

      i_out += 1

      for i in range(nsteps):
         if i % 500 == 0: print(f"Step {i}, day {(i + 1) * dt / 86400}.")

         # Update externally varying parameters
         self.update_externals(s0, j_now, (i + 1) * dt)

         # Compute Lagragian origin points
         z_org = self.get_origins(s0, j_old, j_now, dt)

         # Advect species
         self.step_advection(s0, z_org, j_old, j_now, dt)
        
         if self.do_photolysis:
             # Diagnose photolysis rates
             self.compute_photolysis(s0, o0, z_org, j_now, i_out)
              
         # Run chemistry for the time step
         if self.do_chemistry:
            self.step_chemistry(s0, z_org, j_now, dt)

         # Diabatic tendencies
         if self.do_radiation:
            self.compute_radiation(s0, o0, j_now, i_out)
            dQ = o0.lw_hr[i_out, :] + o0.sw_hr[i_out, :] + self.dyn_hr[:]
            s0.T[j_now] += dt * dQ / 86400.

         if self.do_convection:
             self.convective_adjustment(s0,j_now)

         if self.do_humidity:
             self.relax_humidity(s0,j_now)
          
         i_step += 1

         if i_step >= output_freq:
            self.save_state(s0, o0, j_now, i_out)
            i_out += 1
            i_step = 0

         # Test for instabilities
         if np.max(s0.T[j_now]) > 1000.:
            raise ValueError(f'Temperatures exceeding 1000K produced (step {i}, day {(i + 1) * dt/86400.:.2f}); instability developing?')

         j_old, j_now = j_now, j_old

      return times, o0
   # }}}

import pygeode as pyg
def to_pyg(col, ts, out, init = None):
# {{{
   time = pyg.Yearless(ts / 86400., units = 'days', startdate = dict(year = 1, day = 0))
   pfull = pyg.Pres(col.pfull, name = 'pfull')
   phalf = pyg.Pres(col.phalf, name = 'phalf')
   zfull = pyg.Height(col.zfull, name = 'zfull')
   zhalf = pyg.Height(col.zhalf, name = 'zhalf')

   def add_var(name, values, unit):
      if values.shape[1] == col.Nz:
         axs = (time, zfull,)
      elif values.shape[1] == col.Nz + 1:
         axs = (time, zhalf,)
      else:
         raise ValueError(f'Variable {name} has unrecognized length.')

      v = pyg.Var(axs, name = name, values = values[:].copy())
      v.units = unit
      return v

   def add_scalar(name, values, unit):
      axs = (time, )
      v = pyg.Var(axs, name = name, values = values[:].copy())
      v.units = unit
      return v

   vs = []
   for name, vals in out.columns.items():
      if init is None:
         v = vals
      else:
         v = vals - init.columns[name][:]

      vs.append(add_var(name, v, ''))#col.variables[name].unit))

   for name, vals in out.scalars.items():
      if init is None:
         v = vals
      else:
         v = vals - init.columns[name][:]

      vs.append(add_scalar(name, v, ''))

   #for name, var in col.output_variables.items():
      #vs.append(add_var(name, var))

   return pyg.asdataset(vs)
# }}}

