###################################
### photochem.py
###################################

## Tools for photochemical processes

import numpy as np


def slant_column(constituent, dz, sza, n_air):
# {{{
   sza = np.min([np.pi/2,sza]) # clip above 90 deg
   return np.cumsum(constituent * n_air * dz) / np.cos(sza) # molec cm-2
# }}}

def calc_jNO(self, state, sza, j_new):
# {{{
   # calculate jNO as in Minschwaner et al., 1993 
   # "A new calculation of nitric oxide photolysis in the stratosphere, mesosphere, and lower thermosphere"
   
   # this uses an "offline" approach to calculating jNO 
   # i.e., other radiative fluxes from TUV are not considered
   # this approach follows Tomazelli et al., 2025 ("Impact of the overlapping O2 Schumann-Runge and Herzberg continua with Schumann-Runge bands on photolysis rate coefficients")

   # needs T, O2, O3, NO, N2
   
   # INPUTS:
   dz = -(self.zhalf[1:] - self.zhalf[:-1])*100 # cm
   n_air = self.pfull * 100 / (self.cfg.R * state.T[j_new,:]) * self.cfg.Av * 1e-6 # molec cm-3
    
   N_O2 = slant_column(state.O2[j_new, :], dz, sza, n_air) # molec cm-2
   N_O3 = slant_column(state.O3[j_new, :], dz, sza, n_air) # molec cm-2
   N_NO = slant_column(state.NO[j_new, :], dz, sza, n_air) # molec cm-2
   na_N2 = state.N2[j_new,:]*n_air                         # molec cm-3
   
   N_NO = np.reshape(N_NO,(self.Nz,1,1))
   N_O2 = np.reshape(N_O2,(self.Nz,1,1))
   N_O3 = np.reshape(N_O3,(self.Nz,1,1))
   na_N2 = np.reshape(na_N2,(self.Nz,1,1))

   nb = 3 # number of Schumann-Runge bands
   ns = 6 # number of points within each band
    
   dlam  = np.array([2.3,1.5,1.5]) # nm, spectral width
   Iobar = np.array([3.98e11,2.21e11,2.30e11]) # photons cm-2 nm-1 s-1
   
   dlam = np.reshape(dlam,(1,nb,1))
   Iobar = np.reshape(Iobar,(1,nb,1))
   
   sigma_O2 = np.array([[1.12e-23, 2.45e-23, 7.19e-23, 3.04e-22, 1.75e-21, 1.11e-20],
                        [1.35e-22, 2.99e-22, 7.33e-22, 3.07e-21, 1.69e-20, 1.66e-19],
                        [2.97e-22, 5.83e-22, 2.05e-21, 8.19e-21, 4.80e-20, 2.66e-19]])
   
   wNO_1    = np.array([[0.00e+00, 5.12e-02, 1.36e-01, 1.65e-01, 1.41e-01, 4.50e-02], 
                        [0.00e+00, 0.00e+00, 1.93e-03, 9.73e-02, 9.75e-02, 3.48e-02], 
                        [4.50e-02, 1.80e-01, 2.25e-01, 2.25e-01, 1.80e-01, 4.50e-02]])
   
   sigma_NO_1=np.array([[0.00e+00, 1.32e-18, 6.35e-19, 7.09e-19, 2.18e-19, 4.67e-19],
                        [0.00e+00, 0.00e+00, 3.05e-21, 5.76e-19, 2.29e-18, 2.21e-18],
                        [1.80e-18, 1.50e-18, 5.01e-19, 7.20e-20, 6.72e-20, 1.49e-21]])
                         
   
   wNO_2    = np.array([[0.00e+00, 5.68e-03, 1.52e-02, 1.83e-02, 1.57e-02, 5.00e-03], 
                        [0.00e+00, 0.00e+00, 2.14e-04, 1.08e-02, 1.08e-02, 3.86e-03],
                        [5.00e-03, 2.00e-02, 2.50e-02, 2.50e-02, 2.00e-02, 5.00e-03]])
   
   sigma_NO_2=np.array([[0.00e+00, 4.41e-17, 4.45e-17, 4.50e-17, 2.94e-17, 4.35e-17],
                        [0.00e+00, 0.00e+00, 3.20e-21, 5.71e-17, 9.09e-17, 6.00e-17],
                        [1.40e-16, 1.52e-16, 7.00e-17, 2.83e-17, 2.73e-17, 6.57e-18]])
   
   sigma_O2   = np.reshape(sigma_O2,   (1,nb,ns))
   wNO_1      = np.reshape(wNO_1,      (1,nb,ns))
   sigma_NO_1 = np.reshape(sigma_NO_1, (1,nb,ns))
   wNO_2      = np.reshape(wNO_2,      (1,nb,ns))
   sigma_NO_2 = np.reshape(sigma_NO_2, (1,nb,ns))
   
   D = 1.65e9 # s-1, rate of spontaneous predissociation
   A = 5.1e7 # s-1, rate of spontaneous emission
   kq = 1.5e-9 # cm3 s-1, quenching rate constant
   P = D/(A+D+kq*na_N2) # probability of predissociation
   P = np.reshape(P,(self.Nz,1,1))
   
   sigma_O3 = np.array([4.80e-19,6.88e-19,7.29e-19]) # cm2 molec-1
   sigma_O3 = np.reshape(sigma_O3,(1,nb,1))
   # Ackerman et al., 1971
   # via https://www.uv-vis-spectral-atlas-mainz.org/uvvis/cross_sections/Ozone/O3_Ackerman(1971)_298K_116.5-735nm(int-c).txt
   
   TO3 = np.exp(-N_O3*sigma_O3) # transmission by O3
   
   jNO_band = dlam * Iobar * TO3 * P * np.exp(-sigma_O2   * N_O2) \
                 * (  wNO_1*sigma_NO_1*np.exp(-sigma_NO_1 * N_NO) \
                    + wNO_2*sigma_NO_2*np.exp(-sigma_NO_2 * N_NO) )

   jNO = np.sum(jNO_band,(1,2)) # s-1
    
   return jNO
# }}}
