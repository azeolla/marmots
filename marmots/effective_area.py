"""
This module provides the high-level event loop to calculate
the tau point source effective area.
"""
from typing import Any, Union

import numpy as np
import astropy.coordinates as coordinates
import astropy.units as u

import marmots.antenna as antenna
from marmots.constants import Re

# import marmots.events as events
import marmots.geometry as geometry
#import time


def calculate(
    ra: float,
    dec: float,
    lat: np.ndarray,
    lon: np.ndarray, 
    altitude: np.ndarray,
    orientations: np.ndarray,
    fov: np.ndarray,
    tauexit,
    voltage,
    taudecay,
    maxview: float = np.radians(3.0),
    N: Union[np.ndarray, int] = 1_000_000,
    antennas: int = 4,
    freqs: np.ndarray = np.arange(30,80,10)+5,
    trigger_SNR: float = 5.0,
    min_elev: float = np.deg2rad(-30)
) -> np.ndarray:

    """
    Calculate the effective area of BEACON to a point source
    tau flux.

    Parameters
    ----------
    Enu: float
        The energy of the neutrino that is incident.
    elev: np.ndarray
       The elevation angle (in radians) to calculate the effective area at.
    altitude: float
       The altitude of BEACON (in km) for payload angles.
    prototype: int
        The prototype number for this BEACON trial.
    maxview: float
        The maximum view angle (in degrees).
    N: Union[int, np.ndarray]
        The number of trials to use for geometric area.
    antennas: int
        The number of antennas.
    freqs: numpy array
        The frequencies at which to calculate the electric field (in MHz).
    trigger_SNR: float
        The SNR needed for a trigger.

    Returns
    -------
    Aeff: EffectiveArea
        A collection of effective area components across elevation.
    """

    #begin = time.time()

    # compute the geometric area at the desired elevation angles
    Ag = geometry.geometric_area(
        ra, dec, lat, lon, altitude, maxview, orientations, fov, N=N,min_elev=min_elev
        )

    if Ag.emergence.size == 0:
        geometric = 0
        pexit = 0
        pdet = 0
        effective_area = 0
        coincidence_frac = np.nan
    else:

        # get the exit probability at these elevation angles
        # this is a masked array and will be masked
        # if no tau's exitted at these angles
        Pexit, Etau = tauexit(90.0 - np.rad2deg(Ag.emergence))

        # get a random set of decay lengths at these energies
        decay_length = taudecay.sample_range(Etau)

        # and then sample the energy of the tau's
        Eshower = taudecay.shower_energy(Etau)

        # location of the decay
        decay_point = Ag.trials + (Ag.axis[:,None] * decay_length).T

        # and get the altitude at the decay points
        decay_altitude = geometry.norm(decay_point) - Re

        # get the zenith angle at the exit points
        exit_zenith = (np.pi/2.0) - Ag.emergence

        decay_zenith, decay_azimuth, decay_point_spherical = geometry.decay_zenith_azimuth(decay_point, Ag.axis)

        vrms = antenna.Vrms(freqs, antennas)

        n_stations = len(Ag.stations)

        triggers = np.zeros(Ag.trials.shape[0])

        # iterate over stations
        for i in range(n_stations):
            
            ground_view = geometry.view_angle(Ag.trials, Ag.stations[i]["geocentric"], Ag.axis) 

            trigger = np.zeros(Ag.trials.shape[0])

            in_sight = ground_view <= maxview

            distance_to_decay = geometry.norm(Ag.stations[i]["geocentric"] - decay_point[in_sight])

            # calculate the view angle from the decay points
            decay_view = geometry.decay_view(decay_point[in_sight], Ag.axis, Ag.stations[i]["geocentric"])

            # the zenith and azimuth (measured from East to North) from the station to each decay point
            theta, phi = geometry.obs_zenith_azimuth(Ag.stations[i], decay_point[in_sight], decay_point_spherical[in_sight])

            phi_from_boresight = phi - Ag.orientations[i]

            detector_altitude = Ag.stations[i]["geodetic"][2]

            dbeacon = geometry.norm(Ag.stations[i]["geocentric"] - Ag.trials[in_sight])

            # compute the voltage at each of these off-axis angles and at each frequency
            V = voltage(
                np.rad2deg(decay_view),
                np.rad2deg(exit_zenith[in_sight]),
                decay_altitude[in_sight],
                decay_length[in_sight],
                np.rad2deg(decay_zenith[in_sight]),
                np.rad2deg(decay_azimuth[in_sight]),
                distance_to_decay,
                detector_altitude,
                Ag.stations[i]["geodetic"],
                dbeacon,
                freqs,
                Eshower[in_sight],
                antennas,
                np.rad2deg(theta),
                np.rad2deg(phi_from_boresight),
                Ag.fov[i],
            )
            

            # calculate the SNR
            SNR = V / vrms

            # and check for a trigger
            trigger[in_sight] = SNR > trigger_SNR

            triggers = triggers + trigger

        coincidences = np.sum(triggers > 1)
        Pdet = triggers > 0
        num_triggers = np.sum(Pdet)

        # and save the various effective area coefficients at these angles
        geometric = (Ag.area * np.sum(Ag.dot)) / Ag.N
        pexit = np.mean(Pexit)
        pdet = np.mean(Pdet)
        effective_area = np.sum(Ag.area * Ag.dot * Pexit * Pdet) / Ag.N
        coincidence_frac = coincidences/num_triggers

    #end = time.time()
    # and now return the computed parameters
    return np.array([geometric, pexit, pdet, effective_area, coincidence_frac])
    #return end - begin
