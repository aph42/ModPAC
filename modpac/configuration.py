import os

class Configuration():
   def __init__(self, config_basename, config_root = None):
# {{{
      if config_root == None:
         from modpac import modpac_root
         config_root = f'{modpac_root}'

      self.config_basename = config_basename
      self.config_file = config_basename + '.json'
      self.config_root = config_root

      file_base = f'{self.config_root}/configs/{self.config_file}'

      if os.path.exists(file_base):
         d = self.from_json(file_base)
      else:
         raise ValueError(f"No configuration file {file_base} found.")

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

   def from_json(self, filename):
# {{{
      import json

      # Read global configuration file
      with open(filename, 'r') as f:
         d = json.load(f)

      return d
# }}}

   #def to_toml(self, out_file):
# {{{
      #with open(out_file, 'w') as f:
         #yaml.

# }}}

# Todo: serialize
