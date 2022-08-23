"""
This module parameterizes the response of BEACON antennas to electric fields.
"""
import numpy as np
import pandas as pd

import marmots.sky as sky
from marmots.constants import Z_0, c_km, k_b
from marmots import data_directory
from scipy.interpolate import Akima1DInterpolator, LinearNDInterpolator


__all__ = [
    "noise_voltage",
    "voltage_from_field",
    "get_Tsys",
    "get_Tground",
]
 

def read_xfdtd_gain(finame):
    gain = pd.read_csv(
        finame,
        skiprows=1,
        names=["freq_MHz", "theta_deg", "phi_deg", "phiGain", "thetaGain"],
    )

    gain.freq_MHz *= 1000.0  # stored in GHz in csv file, convert to MHz here
    gtheta = gain.thetaGain  # dBi
    gphi = gain.phiGain  # dBi

    G = np.sqrt(gtheta ** 2 + gphi ** 2)
    gain["G_dBi"] = 10.0 * np.log10(G)

    return gain

def read_xfdtd_impedance(finame, Z0=50.0):
    impedance = pd.read_csv(
        finame, names=["freq_MHz", "RealZ", "ImagZ"], dtype=float, skiprows=1
    )

    impedance[
        "freq_MHz"
    ] *= 1000.0  # stored in GHz in csv file, convert to MHz here
    impedance["Z"] = impedance.RealZ + 1j * impedance.ImagZ
    impedance["Gamma"] = (impedance.Z - Z0) / (impedance.Z + Z0)
    impedance["S11"] = 20.0 * np.log10(abs(impedance.Gamma))
    return impedance


hpol_gain_file = read_xfdtd_gain(
            data_directory + "/hpol_antenna_model_gain.csv"
        )
hpol_csv_freqs = hpol_gain_file.freq_MHz.values
hpol_theta = hpol_gain_file.theta_deg.values
hpol_az = hpol_gain_file.phi_deg.values
hpol_gain = hpol_gain_file.G_dBi.values

hpol_gain_interp = LinearNDInterpolator((hpol_csv_freqs, hpol_theta, hpol_az), hpol_gain) 

vpol_gain_file = read_xfdtd_gain(
            data_directory + "/vpol_antenna_model_gain.csv"
        )
vpol_csv_freqs = vpol_gain_file.freq_MHz.values
vpol_theta = vpol_gain_file.theta_deg.values
vpol_az = vpol_gain_file.phi_deg.values
vpol_gain = vpol_gain_file.G_dBi.values

vpol_gain_interp = LinearNDInterpolator((vpol_csv_freqs, vpol_theta, vpol_az), vpol_gain)

hpol_impedance = read_xfdtd_impedance(data_directory + "/hpol_antenna_impedance.csv")

Z_re_interp_hpol = Akima1DInterpolator(hpol_impedance.freq_MHz, hpol_impedance.RealZ)
Z_im_interp_hpol = Akima1DInterpolator(hpol_impedance.freq_MHz, hpol_impedance.ImagZ)
        
vpol_impedance = read_xfdtd_impedance(data_directory + "/vpol_antenna_impedance.csv")

Z_re_interp_vpol = Akima1DInterpolator(vpol_impedance.freq_MHz, vpol_impedance.RealZ)
Z_im_interp_vpol = Akima1DInterpolator(vpol_impedance.freq_MHz, vpol_impedance.ImagZ)

Z_L = 200.0  # Ohms, the impedance at the load
T_L = 100.0 # Kelvin, noise temperature of the first stage beacon amps

ground_temp = 290 # Kelvin
sky_frac = 0.5



def voltage_from_field(
    Epeak: np.ndarray, freqs: np.ndarray, antennas: int, gain: float,
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
    antennas: int
        The number of antennas.
    gain: float
        The peak gain (in dBi).

    Returns
    -------
    voltage; np.ndarray
        The voltage seen at the load of the antenna.
    """

    # calculate the linear gain - `gain` must be power gain.
    G = np.power(10.0, gain / 10)

    # this is the voltage seen by the antenna
    V_A = (
        2
        * Epeak
        * antennas
        * (1e3 * c_km / (1e6 * freqs))
        * np.sqrt((Z_A / Z_0) * G / (4 * np.pi))
    )

    # and put this through the voltage divider to get the
    # voltage seen at the load
    V_L = V_A * Z_L / (Z_A + Z_L)

    # and we are done
    return V_L


def V_rms(freqs):
    """
    The RMS voltage created by galactic, extragalactic, ground, and system noise.
    """
    
    # P_div is the power from the voltage divider
    P_div = (
        np.abs(Z_L) ** 2
        / np.abs(
            Z_L
            + Z_re_interp_hpol(freqs)
            + 1j * Z_im_interp_hpol(freqs)
        )
        ** 2
    )
    noise = (
        4.0 * k_b * Z_re_interp_hpol(freqs) * (sky_frac * sky.noise_temperature(freqs) + (1-sky_frac) * ground_temp)
    ) # noise due to galactic, extragalactic, and ground
        
    noise *= P_div
    noise += (
        4.0 * k_b * T_L * np.real(Z_L)
    )  # internal noise
    
    df = freqs[1]-freqs[0]
    noise[np.isnan(noise)] = 0 # replace all NaNs with 0
    
    return np.sqrt(np.sum(noise*df))

def noise_fd(freqs):
    """
    Produces a spectrum of noise.
    """
    N = freqs.size

    rms = V_rms(freqs)

    amplitude = np.random.rayleigh((rms*np.sqrt(N)), size=N)
    phase = 2. * np.pi* np.random.random(N)
    noise = amplitude * np.cos(phase) + 1j * amplitude * np.sin(phase)

    return noise

def get_random_noise(freqs, size):
    freqs = np.tile(freqs, (size,1))
    noise = list(map(noise_fd, freqs))
    rms = np.std(noise, axis=1)

    return rms




