###################################
### dynamics.py
###################################

import numpy as np
from scipy import sparse

## Tools for advection, diffusion processes

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

def build_advection_matrix(z_adv, z_org):
# {{{
   ''' Constructs matrices to build weights for advection interpolation. 
    Returns a tuple of three matrices, C, D, and Del; the weights for shape-preserving
    interpolation require a further diagonal matrix T to eliminate overshoot and can
    be calculated as C + D @ T @ Del.'''

   N = len(z_org)
   M = len(z_adv)

   ip = z_adv.searchsorted(z_org)
   iL = np.where(ip == 0.)[0]
   iR = np.where(ip == M)[0]

   ip[iL] = 1
   ip[iR] = M - 1

   i0 = ip - 1

   dx = z_adv[ip] - z_adv[i0]
   tn = (z_org - z_adv[i0]) / dx
   tn[iL] = 0.
   tn[iR] = 1.

   otn = 1 - tn

   Dl = np.zeros(M - 1)
   Dc = np.zeros(M)
   Dr = np.zeros(M - 1)

   Dx = np.diff(z_adv)

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

def advect_quantity(z_adv, C, D, Del, X):
# {{{
   ''' Carry out advection of a given field X, given the components of the
   weights matrix C, D, and Del from build_advection_matrix().  Calculates
   shape-preserving modifications for this field. Normalization (mass
   conservation) is turned off for accuracy reasons. Flux boundary
   conditions are not yet implemented. The array X is oriented in increasing
   height.'''

   Nz = len(z_adv)

   # Compute shape-preserving modifications
   T = np.ones(Nz)

   def fmt(a): return ' '.join([f'{d:5.1f}' for d in a])

   # One-sided estimates at every grid point
   d0 = (X[1:] - X[:-1]) / (z_adv[1:] - z_adv[:-1])

   # Initial derivatives for the interpolation splines
   m = np.zeros(Nz)
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
   T = sparse.diags_array([T], offsets = [0], shape = (Nz, Nz), format = 'csr')

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

def make_diffusion_operator(z_full):
# {{{
   N = len(z_full)

   Dl = np.zeros(N - 1)
   Dc = np.zeros(N)
   Dr = np.zeros(N - 1)

   Dx = np.diff(z_full)

   Dr[1:  ] =  1 / (Dx[:-1]**2)
   Dc[1:-1] = -2 / (Dx[1:] * Dx[:-1])
   Dl[ :-1] =  1 / (Dx[1:]**2)

   return sparse.diags_array([Dl, Dc, Dr], offsets = [-1, 0, 1], shape = (N, N), format = 'csr')
# }}}
