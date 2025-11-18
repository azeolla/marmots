"""
This module provides the high-level event loop to calculate
the tau point source effective area.
"""
#from typing import Any, Union

import numpy as np

from marmots.constants import Re
import marmots.geometry as geometry


def calculate(
    ra: float,
    dec: float,
    lat: np.ndarray,
    lon: np.ndarray, 
    altitude: np.ndarray,
    orientations: np.ndarray,
    fov: np.ndarray,
    antennas: np.ndarray,
    tauexit,
    voltage,
    taudecay,
    detector,
    maxview: float = np.radians(3.0),
    N: int = 1_000_000,
    freqs: np.ndarray = np.arange(30,80,10)+5,
    trigger_SNR: float = 5.0,
    min_elev: float = np.deg2rad(-30),
    time: str = '2025-03-20 12:00:00'
) -> np.ndarray:

    """
    Calculate the effective area of BEACON to a point source
    tau flux.

    Parameters
    ----------
    ra: float
       Right ascension of the point source.
    dec: float
       Declination of the point source.
    lat: np.ndarray
       Latitudes of the phased arrays (degrees).
    lon: np.ndarray
       Longitudes of the phased arrays (degrees).
    altitude: np.ndarray
       The altitudes of the phased arrays (in km).
    orientations: np.ndarray
       The orientation of each phased array (in degrees, relative to east).
    fov: np.ndarray
       The field-of-view of each phased array (in degrees).
    antennas: np.ndarray
       The number of phased antennas in each phased array.
    tauexit: class
        The appropriate TauExitLUT class (from tauexit.py).
    voltage: class
       The EFieldParam class (from efield.py).
    taudecay: class
       The DecayParticle class (from pythia.py).
    detector: class
        The Detector class (from antenna.py).
    maxview: float
        The maximum view angle (in radians). This is the opening angle of the cone projected towards the point source.
    N: int
        The number of trials (exit points) to generate.
    freqs: np.ndarray
        The center frequencies of 10 MHz bins spanning the bandwidth.
        Ex. For 30-80 MHz, freqs = [35, 45, 55, 65, 75]
    trigger_SNR: float
        The voltage SNR needed for a trigger.
    min_elev: float
        Elevation angles below this threshold will not be simulated. Effective area will be assumed to be zero.
    time: str
        The time in which to calculate the instantaneous effective area.

    Returns
    -------
    Aeff: EffectiveArea
        A collection of effective area components at the specified right ascension and declination.
    """

    #begin = time.time()

    # compute the geometric area associated with a point source at the desired ra and dec
    Ag = geometry.geometric_area(
        ra, dec, lat, lon, altitude, maxview, orientations, fov, antennas, N=N, min_elev=min_elev, time=time
        )

    # if no Earth is in view, skip the Monte Carlo
    if Ag.emergence.size == 0:
        geometric = 0
        pexit = 0
        pdet = 0
        effective_area = 0
        coincidence_frac = np.nan
    else:

        # get the exit probability at these emergence angles
        # this is a masked array and will be masked
        # if no tau's exitted at these angles
        Pexit, Etau = tauexit(90.0 - np.rad2deg(Ag.emergence))

        # get a random set of decay lengths at these energies
        decay_length = taudecay.sample_range(Etau)

        # and then sample the energy of the tau's
        Eshower = taudecay.shower_energy(Etau)

        # location of each decay
        decay_point = Ag.trials + (Ag.axis[:,None] * decay_length).T

        # and get the altitude at the decay points
        decay_altitude = geometry.norm(decay_point) - Re 

        # get the zenith angle at the exit points
        exit_zenith = (np.pi/2.0) - Ag.emergence

        # get the zenith and azimuth angle at the decay point
        decay_zenith, decay_azimuth, decay_point_spherical = geometry.decay_zenith_azimuth(decay_point, Ag.axis)

        # calculate the RMS of the antenna noise
        vrms = detector.Vrms(freqs)

        # number of stations
        n_stations = len(Ag.stations)

        triggers = np.zeros(Ag.trials.shape[0])

        # iterate over stations
        for i in range(n_stations):

            # the view angle between the station and the exit points
            ground_view = geometry.view_angle(Ag.trials, Ag.stations[i]["geocentric"], Ag.axis) 

            trigger = np.zeros(Ag.trials.shape[0])

            # only look at exit points within the maxview
            in_sight = ground_view <= maxview

            # distance from the station to each decay point
            distance_to_decay = geometry.norm(Ag.stations[i]["geocentric"] - decay_point[in_sight])

            # calculate the view angle from the decay points
            decay_view = geometry.view_angle(decay_point[in_sight], Ag.stations[i]["geocentric"], Ag.axis) 

            # the zenith and azimuth (measured from East to North) from the station to each decay point
            theta, phi = geometry.obs_zenith_azimuth(Ag.stations[i], decay_point[in_sight], decay_point_spherical[in_sight])

            # redefine phi, given the orientation of the station
            phi_from_boresight = (phi - np.deg2rad(Ag.orientations[i]) + np.pi) % (2*np.pi) - np.pi

            detector_altitude = Ag.stations[i]["geodetic"][2] 

            # distance from the station to the exit point
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
                np.rad2deg(theta),
                np.rad2deg(phi_from_boresight),
                Ag.fov[i],
                detector,
            )
            

            # calculate the phased SNR
            SNR = np.sqrt(Ag.antennas[i]) * (V / vrms)

            # and check for a trigger
            trigger[in_sight] = SNR > trigger_SNR

            # these decays appear to be above horizon
            above = (np.pi/2 - theta) > geometry.horizon_angle(detector_altitude)

            # if the event is above the horizon, we would not find
            # them in the search as they would be treated as background
            trigger[in_sight][above] = 0.0

            triggers = triggers + trigger

        coincidences = np.sum(triggers > 1) # events which trigger more than one station
        Pdet = triggers > 0
        num_triggers = np.sum(Pdet)

        # and save the various effective area coefficients at these angles
        geometric = (Ag.area * np.sum(Ag.dot)) / Ag.N
        pexit = np.mean(Pexit)
        pdet = np.mean(Pdet)
        effective_area = np.sum(Ag.area * Ag.dot * Pexit * Pdet) / Ag.N
        with np.errstate(divide='ignore', invalid='ignore'):
            coincidence_frac = coincidences/num_triggers

    #end = time.time()
    # and now return the computed parameters
    return np.array([geometric, pexit, pdet, effective_area, coincidence_frac])
    #return end - begin
