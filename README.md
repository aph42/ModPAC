# ModPAC - Modular Photochemistry in an Atmospheric Column

ModPAC is a radiative/photochemical column model built around MUSICA, TUVx, and
RRTMG, developed by Peter Hitchcock and Aaron Match. The original motivation
was to study ozone photochemistry in the tropical stratosphere as it is
affected by the quasibiennial oscillation. It is designed to be easily
configurable, allowing for easy modifications of the chemistry, photochemistry,
and advective details.

It is still very much work in progress and there is no published description
available yet. While the code is available, there are no guarantees that any of
it works out of the box.

## Dependencies

    - [MUSICA](https://musica.readthedocs.io/en/stable/) (including [TUV-x](https://github.com/NCAR/tuv-x)) for the chemical solver and photochemical rates
    - [RRTMG](http://rtweb.aer.com/rrtm_frame.html) for computing radiative heating rates
    - [pygeode](http://pygeode.github.io/) for organizing and serializing model output

## Configuration
The model can be configured using external files, or programmatically. These
details are very much in flux; at some point some documentation might be
written.

## Grid

The column is defined in log-pressure height coordinates from the ground to
some upper boundary; it is designed to be compatible with the RRTMG and TUVx
grids, so there are full levels (i = 1,...,N) and half levels at the interfaces
of the full levels. 

The vertical velocity and radiative fluxes are defined on the half-levels,
while the temperatures and chemical species are defined on full-levels.
The MUSICA solver operates on the full levels; TUV-x requires
both the full level pressures and halflevels.

## Chemistry
The chemistry is solved for using the configurable solver of MUSICA.

The MUSICA mechanism solver called interactively; each timestep it is
initialized with tracer values at the Lagrangian origin points for each grid
point and uses temperatures and pressures at the mid-way point between the
source and destination point. 

The mechanism can be configured using [MICM](https://github.com/NCAR/micm); we
have also written some utility files that allow for a more
compact/human-readable definition of a set of reactions and their rates.

### To Do:
 - [ ] Incorporate external configuration

## Photolysis Rates
Photolysis rates are computed using TUV-x. The calculated rates are then fed
into the MUSICA at every model timestep. 

### To Do:
 - [ ] Allow for easier manual specification of photochemical rates

## Radiative Heating

RRTMG is used to compute clear-sky radiative heating rates. At the moment
ozone, water vapour, and carbon dioxide are radiatively active. One can run
with fixed zenith angle or a diurnal cycle, and some orbital calculations are
included in order to determine appropriate values given a calendar date and
geographic location.

### To Do:
 - [ ] Implement seasonal cycle
 - [ ] Implement daily mean calculations

## Advection

The advection algorithm is a semi-Lagrangian scheme following Kaas (2008;
doi:10.1111/j.1600-0870.2007.00293.x). This involves computing origin points
for Lagrangian trajectories that end at each grid point, and spatially
interpolating the tracer field to these points. Computing the back trajectories
has been implemented following Kass (2008) (though see McGregor 1993,
doi:10.1175/1520-0493(1993)121<0221:EDODPF>2.0.CO;2 as well).

# Numerical Methods
## Technical details

The advection scheme is written to allow for mass conservation: a set of
interpolation weights such that the value of a tracer at the origin points
$z^\ast_j$ for the trajectories ending at $z_k$ can be expressed as a matrix
multiplication: 

$$ X^\ast_j = w_{jk} X_k. $$

The weights matrix will have non-zero entries for 4 levels around the origin
location. This form is then re-normalized to ensure global conservation of the
the tracer in some sense. Given the exponential decrease in density, however,
conserving tracer mass itself doesn't make a lot of sense (overle the atmosphere
much of the mass will be transported meridionally). The re-normalization
provides an estimate of the local divergence which could potentially be used as
a way to parameterize meridional transport.

At the moment the spatial interpolation required to compute tracer values at
the origin points is a cubic Hermite polynomial. This interpolation has been a
source of significant biases around the tropical tropopause (see [Hardiman et
al. 2015](https://doi.org/10.1175/JCLI-D-15-0075.1)), although the difficulties
in their model apparently arose from mostly-reversable wave motions that
generate spurious irreversable transport around sharp changes in gradients.

The interpolation is 'shape-preserving', meaning that it should not produce
extrema that are not in the un-interpolated gridded values (see [Rasch and
Williamson 1990](https://doi.org/10.1137/0911039) and [Fritsch and Carlson
1980](https://doi.org/10.1137%2F0717021)). This requires the interpolation
weights to be (potentially) different for each advected quantity, so this might
get slow particularly with more advected species. It might ultimately be worth
implementing this all in a Cython module for efficiency.

There is a test for the advection scheme in [col_test.py](col_test.py) that
initializes a local patch of tracer concentration then advects it up and down
periodically.  With the current shape-preserving scheme one can get to very
high Courant numbers (~10 at least) and maintains stability.  However, the mass
conservation is currently turned off as it is introducing instability, and
there are boundary effects that can arise which I think have to do with the
normalization of the weights around the upper and lower boundaries of the
domain. 

I think this will be mitigated by working out how to deal with fluxes
at the boundaries in this framework.

#### To Do:
 - [ ] Work out fluxes at boundaries, and a better normalization scheme
