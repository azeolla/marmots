"""
This module parameterizes the response of BEACON antennas to electric fields.
"""
import numpy as np
import pandas as pd

import marmots.sky as sky
from marmots.constants import Z_0, c, k_b
from marmots import data_directory
from scipy.interpolate import Akima1DInterpolator, interpn
from scipy.fft import rfftfreq, irfft

#from scipy.fft import irfft


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
            data_directory + "/beacon/beacon_150m_hpol_realized_gain_middle.csv"
        )
hpol_csv_freqs = np.unique(hpol_gain_file.freq_MHz.values)
hpol_theta = np.unique(hpol_gain_file.theta_deg.values)
hpol_az = np.unique(hpol_gain_file.phi_deg.values)
hpol_gain = hpol_gain_file.G_dBi.values.reshape((hpol_csv_freqs.size, hpol_theta.size, hpol_az.size))


"""
vpol_gain_file = read_xfdtd_gain(
            data_directory + "/beacon/beacon_150m_vpol_realized_gain_middle.csv"
        )
vpol_csv_freqs = vpol_gain_file.freq_MHz.values
vpol_theta = vpol_gain_file.theta_deg.values
vpol_az = vpol_gain_file.phi_deg.values
vpol_gain = vpol_gain_file.G_dBi.values

vpol_gain_interp = LinearNDInterpolator((vpol_csv_freqs, vpol_theta, vpol_az), vpol_gain)
"""

hpol_impedance = read_xfdtd_impedance(data_directory + "/beacon/beacon_150m_hpol_impedance_middle.csv")

Z_re_interp_hpol = Akima1DInterpolator(hpol_impedance.freq_MHz, hpol_impedance.RealZ)
Z_im_interp_hpol = Akima1DInterpolator(hpol_impedance.freq_MHz, hpol_impedance.ImagZ)
        
# vpol_impedance = read_xfdtd_impedance(data_directory + "/beacon/beacon_150m_vpol_impedance_middle.csv")

# Z_re_interp_vpol = Akima1DInterpolator(vpol_impedance.freq_MHz, vpol_impedance.RealZ)
# Z_im_interp_vpol = Akima1DInterpolator(vpol_impedance.freq_MHz, vpol_impedance.ImagZ)

Z_L = 200.0  # Ohms, the impedance at the load
T_L = 100.0 # Kelvin, noise temperature of the first stage beacon amps

ground_temp = 300 # Kelvin
sky_frac = 0.5


def effective_height(freqs):
           
    h_eff = (
        4.0 * Z_re_interp_hpol(freqs) / Z_0 * (c/freqs)**2 / 4.0 / np.pi
    )
    
    P_div = (
        np.abs(Z_L) ** 2
        / np.abs(
            Z_re_interp_hpol(freqs)
            + 1j * Z_im_interp_hpol(freqs)
            + Z_L
        )
        ** 2
    )
    
    h_eff *= P_div
    
    h_eff = np.sqrt(h_eff.astype(np.complex))
        
    return abs(h_eff)



def voltage_from_field(
    Epeak: np.ndarray, freqs: np.ndarray, antennas: int, theta: np.ndarray, phi: np.ndarray
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

    phi = (phi + 360) % 360

    f, t = np.meshgrid(freqs, theta)
    __, p = np.meshgrid(freqs, phi)

    # calculate the linear gain - `gain` must be power gain.
    D = interpn((hpol_csv_freqs, hpol_theta, hpol_az),
        hpol_gain,
        (f, t, p),
       )
    G = (10 ** (D / 10.0))

    return antennas * Epeak * effective_height(freqs) * np.sqrt(G)


def Vrms(freqs: np.ndarray, antennas: int):
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
        k_b * T_L * np.real(Z_L)
    )  # internal noise
    
    noise[np.isnan(noise)] = 0 # replace all NaNs with 0
    df = freqs[1]-freqs[0]
    
    return np.sqrt(np.sum(antennas*df*noise))
    
   





