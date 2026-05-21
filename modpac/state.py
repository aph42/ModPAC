import numpy as np

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
      #self.prognostic = prognostic
      self.output = output
   # }}}

class SpeciesVariable(ColumnVariable):
   def __init__(self, name, unit, Nz, initial_value, advect = False, fixed = False, prognostic = False, output = False, **properties):
   # {{{
      ColumnVariable.__init__(self, name, unit, Nz, initial_value, prognostic, output)

      self.advect = advect
      self.fixed  = fixed

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
