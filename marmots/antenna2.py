"""
This module parameterizes the response of BEACON antennas to electric fields.
"""
import numpy as np
import pandas as pd
from typing import Any, Tuple

import marmots.sky as sky
from marmots.constants import Z_0, c, k_b
from marmots import data_directory
from scipy.fft import rfftfreq, irfft
from numba import jit, njit
from interpolation.splines import CGrid, eval_linear, extrap_options
from scipy.interpolate import interp1d

import warnings
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter('ignore', NumbaPendingDeprecationWarning)


__all__ = [
    "noise_voltage",
    "voltage_from_field",
    "get_Tsys",
    "get_Tground",
]
 
class Detector:
    def __init__(self, model, freqs, pol="hpol", gain = None):

        """
        `model` specifies which model to load. Currently, there is `prototype`, `rhombic`, and `matched`.
        `freqs` specifies which frequencies to evaluate.
        `gain` should be given if using model `matched`.
        """
        
        if model == "heron":

            if pol == "hpol":
                port = "X"
            else:
                port = "Z"
                
            data = np.load(f"/data2/zeolla/HERON/HERON_XZ_SLOPE20_{port}_port.npz")

            h_theta = data["hoc_theta_global_m"][0, 1]
            h_phi = data["hoc_phi_global_m"][0, 1]

            self.freqs = data["frequency_MHz"]
            self.theta = data["theta_global_deg"]
            self.phi = data["phi_global_deg"]
            self.grid = CGrid(self.theta, self.phi)

            h_theta_interp = interp1d(
            self.freqs,
            h_theta,
            axis=0,
            kind="linear",
            bounds_error=True,
            )

            h_phi_interp = interp1d(
            self.freqs,
            h_phi,
            axis=0,
            kind="linear",
            bounds_error=True,
            )

            self.h_theta = h_theta_interp(freqs)
            self.h_phi = h_phi_interp(freqs)

            h_eff = np.sqrt(abs(self.h_theta)**2 + abs(self.h_phi)**2)


            # Antenna simulation has boresight direction at phi = 90
            # Rotate antenna so boresight is pointed at phi = 0
            dphi = self.phi[1]-self.phi[0]
            shift = int(90 / dphi)

            self.h_eff = np.roll(h_eff, -shift, axis=2)
            
            impedance = data["Zant_ohm"][0, 1]

            impedance_interp = interp1d(
            self.freqs,
            impedance,
            kind="linear",
            bounds_error=True,
            )

            self.impedance = impedance_interp(freqs)

            self.r = 4
            self.Z_L = 200  # Ohms, the impedance at the load
            self.T_L = 100.0 # Kelvin, noise temperature of the first stage beacon amps
    
            self.ground_temp = 300 # Kelvin
            self.sky_frac = 0.5

            
        elif model == "matched":

            if gain == None:
                print("A gain value must be supplied!")

            resistance = 50
            reactance = 0

            self.impedance = resistance + 1j*reactance

            self.r = 1
            self.Z_L = 50  # Ohms, the impedance at the load
            self.T_L = 100.0 # Kelvin, noise temperature of the first stage beacon amps

            self.ground_temp = 300 # Kelvin
            self.sky_frac = 0.3

            self.h_eff = self.effective_height(freqs, gain)

        else:
            print("Model not supported!")


        self.V_div = np.sqrt((1/self.r) 
            * (self.r * np.abs(self.Z_L)) ** 2
            / np.abs(
                np.real(self.impedance)
                + 1j * np.imag(self.impedance)
                + self.r * self.Z_L)
            ** 2)
        

    def effective_height(self, freqs, gain) -> np.ndarray:
        """
        Calculates the antenna effective height at each frequency.
        """
            
        h_eff = 4.0 * np.real(self.impedance) / Z_0 * (c/freqs)**2 / 4.0 / np.pi * 10.**(gain/10.0)
        
        return np.sqrt(h_eff)

    def voltage_from_field(
        self, Epeak: np.ndarray, theta: np.ndarray, phi: np.ndarray
    ) -> np.ndarray:
        """
        Given a peak electric field (in V/m), calculate the voltage seen
        at the load of the BEACON antenna.

        See any RF or antenna textbook for a derivation of this.

        Parameters
        ----------
        Epeak: np.ndarray
            The peak-electric field (in V/m).
        freqs: np.ndarray
            The frequencies (in MHz) at which to evaluate.
        theta: np.ndarray
            The observation zenith angle (in degrees).
        phi: np.ndarray
            The observation azimuth angle (in degrees).
            
        Returns
        -------
        voltage: np.ndarray
            The voltage seen at the load of the antenna.
        """

        if len(self.h_eff.shape) > 1:
            h_eff = response_interp(self.grid,
                            self.h_eff,
                            theta,
                            phi,
                           )
            x = Epeak * self.V_div[:,None] * h_eff
            
        else:
            h_eff = self.h_eff

            x = Epeak * (self.V_div * h_eff)[:,None]
        
        x[np.isnan(x)] = 0.0
            
        out = np.sum(x, axis=0)

        return out

    def Vrms(self, freqs: np.ndarray):
        """
        The RMS voltage created by galactic, extragalactic, ground, and system noise.
        """

        # noise due to galactic, extragalactic, and ground
        noise = 4.0 * k_b * np.real(self.impedance) * (self.sky_frac * sky.noise_temperature(freqs) + (1-self.sky_frac) * self.ground_temp)
        noise *= (self.V_div)**2
        # internal noise
        noise += k_b * self.T_L * np.real(self.Z_L)

        noise[np.isnan(noise)] = 0 # replace all NaNs with 0
        df = freqs[1]-freqs[0]
        
        return np.sqrt(np.sum(df*noise))



@njit
def response_interp(
    grid,
    values: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """
    
    """
    n_freqs  = values.shape[0]
    out = np.zeros((n_freqs, theta.size), dtype=np.float64)

    for i in range(n_freqs):

        pts = np.column_stack((
            theta, phi
        ))
        
        out[i, :] = eval_linear(grid, values[i], pts)

    return out





    
   





