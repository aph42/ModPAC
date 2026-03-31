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

def parse_rate(rate, n):
# {{{
   params = {}
   # concentrations moved from molecule cm-3 to mol m-3; 
   # need to multiply A coefficients by factor of (1e6 * Av-1)**(nreactants - 1)
   rfactor = (6.022e23 / 1e6) ** (n - 1)
   # also, temperature ratio term is flipped in implementation, so powers (B) need to be multiplied by -1

   if 'ko' in rate and 'rate' not in rate: 
      rtype = 'Ternary'
      rates = split(rate, ',')

      rts = {}
      for r in rates:
         k, v = split(r, '=')
         rts[k] = v

      komtch = re.fullmatch(rate_pattern, rts['ko'])
      gd = komtch.groupdict()
      params['k0_A'] = rfactor * float(get_param(gd, 'A', 1.))
      params['k0_B'] = -float(get_param(gd, 'B', -0.))
      params['k0_C'] = float(get_param(gd, 'C', 0.))

      kimtch = re.fullmatch(rate_pattern, rts['ki'])
      gd = kimtch.groupdict()
      params['kinf_A'] = rfactor * float(get_param(gd, 'A', 1.))
      params['kinf_B'] = -float(get_param(gd, 'B', -0.))
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
         params['A'] = rfactor * float(get_param(gd, 'A', 1.))
         params['B'] = -float(get_param(gd, 'B', -0.))
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
            rtype, params = parse_rate(rate, np.sum(rccoeffs))
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
   print('Inputs alone: ', inputs - outputs)
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

def species_list_to_string(components, inputs = False):
# {{{
   cs = []
   for rc in components:
      if rc.coefficient == 1.:
         c = rc.species_name 
      else:
         coef = rc.coefficient
         if (np.round(coef, 0) - coef) / coef < 0.01:
            coef = int(rc.coefficient)
         c = f'{coef}*{rc.species_name}'
      cs.append(c)

   # Punt 3rd bodies to the end
   if 'M' in cs:
      cs = [c for c in cs if c != 'M'] + ['M']

   if len(cs) == 0:
      if inputs: return 'source'
      else: return 'sink'

   return ' + '.join(cs)
# }}}

def arrhenius_rate_to_string(A, B = 0., C = 0., D = 300., n = 1, to_molecule_per_cm3 = False):
# {{{
   if to_molecule_per_cm3:
      rfactor = (1e6 / 6.022e23) ** (n - 1)
   else:
      rfactor = 1

   rate_text = f'{A * rfactor:G}'

   if B != 0.: rate_text += f'({D}/T)**{-B}'
   if C != 0.: rate_text += f'*exp ( {C}/t )'

   return rate_text
# }}}

def identify_category(inputs, outputs):
# {{{
   input_species = [i.species_name for i in inputs]
   output_species = [o.species_name for o in outputs]

   # Odd Oxygen
   if all([i in ['O', 'O2', 'O3', 'M'] for i in input_species]):
      return 'O'

   if any([i == 'O1D' for i in input_species]):
      return 'O1D'

   if any([i in ['H', 'OH', 'H2', 'H2O2', 'HO2'] for i in input_species]):
      return 'H'

   if any([i in ['N', 'NO', 'NO2', 'NO3', 'N2O5'] for i in input_species]):
      return 'N'

   if any([i.find('CL') > -1 for i in input_species]):
      return 'Cl'

   if any([i.find('BR') > -1 for i in input_species]):
      return 'Br'

   return 'Other'
# }}}

def mechanism_to_text(mc, filename, to_molecule_per_cm3 = False, overwrite = False):
# {{{

   if os.path.exists(filename):
      if not overwrite:
         print(f'{filename} exists; aborting.')
         return
      else:
         print(f'{filename} exists; overwriting.')

   cats = dict(O = [], O1D = [], H = [], N = [], Cl = [], Br = [], Other = [], Photolysis = [])

   category_title = {'O': 'Odd Oxygen',
                     'O1D': 'Odd Oxygen (O1D)',
                     'H': 'Odd Hydrogen',
                     'N': 'Odd Nitrogen',
                     'Cl': 'Odd Chlorine',
                     'Br': 'Odd Bromine',
                     'Other': 'Other Reactions',
                     'Photolysis': 'Photolysis'}

   for rct in mc.reactions.arrhenius:
      react = species_list_to_string(rct.reactants, inputs = True)
      prod  = species_list_to_string(rct.products, inputs = False)
      reaction_text = f'{react} {to} {prod}'

      n = np.sum([r.coefficient for r in rct.reactants])
      rate_text = arrhenius_rate_to_string(rct.A, rct.B, rct.C, rct.D, n, to_molecule_per_cm3)

      cat = identify_category(rct.reactants, rct.products)
      cats[cat].append((reaction_text, rate_text))

   for rct in mc.reactions.troe + mc.reactions.ternary_chemical_activation:
      react = species_list_to_string(rct.reactants, inputs = True)
      prod  = species_list_to_string(rct.products, inputs = False)
      reaction_text = f'{react} {to} {prod}'

      n = np.sum([r.coefficient for r in rct.reactants])
      ko = arrhenius_rate_to_string(rct.k0_A, rct.k0_B, rct.k0_C, 300., n, to_molecule_per_cm3)
      ki = arrhenius_rate_to_string(rct.kinf_A, rct.kinf_B, rct.kinf_C, 300., n, to_molecule_per_cm3)
      f = f'{rct.Fc:0.2f}'

      rate_text = f'ko={ko}, ki={ki}, f={f}'

      cat = identify_category(rct.reactants, rct.products)
      cats[cat].append((reaction_text, rate_text))

   for rct in mc.reactions.photolysis + mc.reactions.user_defined:
      react = species_list_to_string(rct.reactants, inputs = True)
      prod  = species_list_to_string(rct.products, inputs = False)
      reaction_text = f'{react} + hv {to} {prod}'

      rate_text = ''

      cats['Photolysis'].append((reaction_text, rate_text))

   with open(filename, 'w') as f:
      for cat, reactions in cats.items():
         nwidth = 0
         for rc, rt in reactions:
            if len(rc) > nwidth: nwidth = len(rc)

         f.write(f"# {category_title[cat]}; {len(reactions)} reactions\n")

         ind = np.argsort([rc for rc, rt in reactions])

         for i in ind:
            if reactions[i][1] == '':
               f.write(f'{reactions[i][0]:<{nwidth}}\n') 
            else:
               f.write(f'{reactions[i][0]:<{nwidth}} ; {reactions[i][1]}\n') 

         f.write('\n')
# }}}

   # Need to convert reaction rate units for A coefficients
   # Trickier reactions: 
   #   HO2 + HO2 → H2O2 + O2 is replaced by two Arrhenius reactions
   #   N2O5 + M → NO2 + NO3 + M is replaced by a TROE reaction
   #   HNO3 + OH → NO3 + H2O also by a TROE reaction 
   #   HO2NO2 + M → HO2 + NO2 + M  also by a TROE reaction




