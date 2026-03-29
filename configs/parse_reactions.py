import os
import re
import json

import numpy as np

import musica.mechanism_configuration as mimc 

to = '→'

def split(st, ch):
   return [l.strip() for l in st.split(ch)]

def parse_product(st):
# {{{
   if '*' in st:
      factor, sp = split(st, '*')
      factor = float(factor)
   else:
      sp = st
      factor = 1.
   
   return sp, factor
# }}}

def get_param(group_dict, param, default):
   if not (param in group_dict) or group_dict[param] == None:
      return default
   else:
      return group_dict[param]

rate_pattern = r'(?P<A>[E.0-9+-]*)' \
               r'(\*\((?P<D>[0-9]*)/[tT]\))?(\*\*(?P<B>[0-9.-]*))?' \
               r'(\*exp\( *(?P<C>[-0-9.]*)/t\))?' \
               r'( ?+ ?(?P<E>[E.0-9+-]*))?'

#tern_pattern = r'(?P<A>[E.0-9+-]*)'

def parse_rate(rate):
# {{{
   params = {}

   if 'ko' in rate and 'rate' not in rate: 
      rtype = 'Ternary'
      rates = split(rate, ',')

      rts = {}
      for r in rates:
         k, v = split(r, '=')
         rts[k] = v

      komtch = re.fullmatch(rate_pattern, rts['ko'])
      gd = komtch.groupdict()
      params['k0_A'] = float(get_param(gd, 'A', 1.))
      params['k0_B'] = float(get_param(gd, 'B', 0.))
      params['k0_C'] = float(get_param(gd, 'C', 0.))

      kimtch = re.fullmatch(rate_pattern, rts['ki'])
      gd = kimtch.groupdict()
      params['kinf_A'] = float(get_param(gd, 'A', 1.))
      params['kinf_B'] = float(get_param(gd, 'B', 0.))
      params['kinf_C'] = float(get_param(gd, 'C', 0.))

      fmtch = re.fullmatch(rate_pattern, rts['f'])
      gd = fmtch.groupdict()
      params['Fc'] = float(get_param(gd, 'A', 1.))

   elif ',' in rate:
      # Punt on complicated stuff for now
      rtype = 'Unknown'
   elif rate[0] == 'k':
      rtype = 'Related to another rate'
   else:
      # Arrhenius
      match = re.fullmatch(rate_pattern, rate)
      #match = re.match(rate_pattern, rate)
      if match is None:
         rtype = 'Unknown'#   raise Exception('Failed regex match.')
      else: 
         rtype = 'Arrhenius'
         gd = match.groupdict()
         params['A'] = float(get_param(gd, 'A', 1.))
         params['B'] = float(get_param(gd, 'B', 0.))
         params['D'] = float(get_param(gd, 'D', 1.))
         params['C'] = float(get_param(gd, 'C', 0.))
         #params['E'] = float(get_param(gd, 'E', 0.))

   return rtype, params
# }}}

def parse_reactions(filename, categories = 0):
# {{{
   inputs = set()
   outputs = set()
   reactions = []

   ncategories = 0

   with open(filename, 'r', encoding = "utf-8") as f:
      for line in f:
         line = line.strip()

         if len(line) == 0 or line[0] == '%': 
            continue

         if line[0] == '#':
            # Heading
            category = line[1:].strip()
            print(f"{category}")
            ncategories += 1
            if categories > 0 and ncategories > categories: break
            continue

         reaction, rate = split(line, ';')

         # Parse reaction
         react, prod = split(reaction, to)

         reactants = split(react, '+')
         reactants, rccoeffs = np.unique(reactants, return_counts = True)
         prod = [parse_product(pr) for pr in split(prod, '+')]
         products, prcoeffs = zip(*prod)

         inputs = inputs.union(reactants)
         outputs = outputs.union(products)

         try:
            rtype, params = parse_rate(rate)
         except Exception as e:
            print(f'{reaction}\n Parsing failed: {rate}', e)

         #if rtype == 'Arrhenius': params['n'] = len(reactants)
         #if rtype not in ['Arrhenius', 'Ternary']:
            #print(f'{reaction:>75}{rtype:>35}')
            #print(params)

         reactions.append((reaction, rate, reactants, rccoeffs, products, prcoeffs, rtype, params))

   print(f'{len(reactions)} reactions; {len(inputs | outputs)} species.')
   return inputs, outputs, reactions
# }}}

def parse_photolysis(filename, rinputs = None):
# {{{
   inputs = set()
   outputs = set()
   reactions = []

   with open(filename, 'r', encoding = "utf-8") as f:
      for line in f:
         line = line.strip()

         if len(line) == 0 or line[0] == '%': 
            continue

         if line[0] == '#':
            # Heading
            category = line[1:].strip()
            print(f"{category}")
            ncategories += 1
            if categories > 0 and ncategories > categories: break
            continue

         reaction = line

         # Parse reaction
         react, prod = split(reaction, to)

         reactants = split(react, '+')
         reactants, rccoeffs = np.unique(reactants, return_counts = True)
         prod = [parse_product(pr) for pr in split(prod, '+')]
         products, prcoeffs = zip(*prod)

         if (not rinputs is None) and (not reactants[0] in rinputs):
            # Skip this reaction; species is not in our input list
            continue

         inputs = inputs.union(reactants)
         outputs = outputs.union(products)

         rtype = 'Photolysis'

         reactions.append((reaction, reactants, rccoeffs, products, prcoeffs, rtype))

   print(f'{len(reactions)} photolysis reactions; {len(inputs | outputs)} species.')
   return inputs, outputs, reactions
# }}}

def make_mechanism():
# {{{
   inputs, outputs, reactions = parse_reactions('tilmes.reactions.txt', 4)

   phinputs, phoutputs, phreactions = parse_photolysis('tilmes.photolysis.txt', inputs)

   print('Excess species:')
   print('Outputs: ', outputs - inputs)
   print('Photolysis inputs: ', phinputs - inputs)
   print('Photolysis outputs:', phoutputs - inputs)

   species_names  = list(inputs | outputs | phoutputs)
   species_names.sort()

   parser = mimc.Parser()
   ts1mech = parser.parse('ts1.json')

   species = {}
   
   for sp_name in species_names:
      found = False

      if sp_name == 'H1202':
         species[sp_name] = mimc.Species('H1202', 
                                     molecular_weight_kg_mol = 0.20982,
                                     other_properties = {'__description':'dibromodifluoromethane'})
         found = True
         continue

      for sp in ts1mech.species:
         if sp.name == sp_name:
            species[sp_name] = sp
            found = True
            continue

      if not found: print(f'Unrecognized species name {sp_name}.')

   gas = mimc.Phase(name = 'gas', species = list(species.values()))

   arrhen = []
   tern = []
   for (reaction, rate, reactants, rccoeffs, products, prcoeffs, rtype, params) in reactions:
      Rcs = [(c, species[r]) for c, r in zip(rccoeffs, reactants)]
      Prs = [(c, species[r]) for c, r in zip(prcoeffs, products)]

      reaction = reaction.replace(to, '->')

      if rtype == 'Arrhenius':
         R = mimc.Arrhenius(reaction, reactants = Rcs, products = Prs, gas_phase = gas, **params)
         arrhen.append(R)

      elif rtype == 'Ternary':
         R = mimc.TernaryChemicalActivation(reaction, reactants = Rcs, products = Prs, gas_phase = gas, **params)
         tern.append(R)

      else:
         print(reaction, rate)

   phot = []

   for (reaction, reactants, rccoeffs, products, prcoeffs, rtype) in phreactions:
      Rcs = [(c, species[r]) for c, r in zip(rccoeffs[:-1], reactants[:-1])]
      Prs = [(c, species[r]) for c, r in zip(prcoeffs, products)]

      if prcoeffs[0] == 1.:
         name = 'j' + reactants[0] + '->' + products[0]
      else:
         name = 'j' + reactants[0] + '->' + str(prcoeffs[0]) + products[0]

      R = mimc.Photolysis(name, reactants = Rcs, products = Prs, gas_phase = gas)
      phot.append(R)

   print('Species: ', len(species))
   print('Arrhenius reactions: ', len(arrhen))
   print('Ternary/TROE reactions: ', len(tern))
   print('Photolysis reactions: ', len(phot))

   return ([gas], species, arrhen + tern + phot)
# }}}

def write_json(name, filename, version = '1.0.0', overwrite = False):
# {{{
   if os.path.exists(filename):
      if not overwrite:
         print(f'{filename} exists; aborting.')
         return
      else:
         print(f'{filename} exists; overwriting.')

   phases, species, reactions = make_mechanism()

   mechanism = dict(name = name, version = version)

   spc = list(species.keys())
   spc.sort()

   mechanism['species'] = [species[sp].serialize() for sp in spc]
   mechanism['reactions'] = [r.serialize() for r in reactions]
   mechanism['phases'] = [p.serialize() for p in phases]

   with open(filename, 'w') as f:
      json.dump(mechanism, f, indent = '   ')
# }}}


   # Need to convert reaction rate units for A coefficients
   # Trickier reactions: 
   #   HO2 + HO2 → H2O2 + O2 is replaced by two Arrhenius reactions
   #   N2O5 + M → NO2 + NO3 + M is replaced by a TROE reaction
   #   HNO3 + OH → NO3 + H2O also by a TROE reaction 
   #   HO2NO2 + M → HO2 + NO2 + M  also by a TROE reaction




