import json

class Configuration():
   def __init__(self, config_file, config_root):
# {{{
      self.config_file = config_file
      self.config_root = config_root

      # Read global configuration file
      with open(f'{config_root}/configs/{config_file}', 'r') as f:
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

# Todo: serialize
