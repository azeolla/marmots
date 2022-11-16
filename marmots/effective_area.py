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
import marmots.tauola as tauola
from marmots.efield import EFieldParam
from marmots.tauexit import TauExitLUT
import time


def calculate(
    Enu: float,
    source: coordinates.SkyCoord,
    altaz: Any,
    beacon: coordinates.EarthLocation,
    orientations: np.ndarray,
    fov: np.ndarray,
    maxview: float = np.radians(3.0),
    N: Union[np.ndarray, int] = 1_000_000,
    antennas: int = 4,
    freqs: np.ndarray = np.arange(30,80,10)+5,
    trigger_SNR: float = 5.0,
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
    
    start = time.time()

    # load the corresponding tau exit LUT
    tauexit = TauExitLUT(energy=Enu, thickness=0)

    # compute the geometric area at the desired elevation angles
    Ag = geometry.geometric_area(
        beacon, source, altaz, maxview, orientations, fov, N=N,
    )

    if Ag.emergence.size == 0:
        geometric = 0
        pexit = 0
        pdet = 0
        effective_area = 0
    else:

        # get the exit probability at these elevation angles
        # this is a masked array and will be masked
        # if no tau's exitted at these angles
        Pexit, Etau = tauexit(90.0 - Ag.emergence.to(u.deg).value)

        # get a random set of decay lengths at these energies
        decay_length = tauola.sample_range(Etau)*u.km

        # and then sample the energy of the tau's
        Eshower = tauola.sample_shower_energies(Etau, N=decay_length.size)

        # location of the decay
        decay_point = Ag.trials + (Ag.axis[:,None] * decay_length).T

        # and get the altitude at the decay points
        decay_altitude = np.linalg.norm(decay_point,axis=1) - Re

        # get the zenith angle at the exit points
        exit_zenith = (np.pi/2.0)*u.rad - Ag.emergence

        decay_zenith, decay_azimuth = geometry.decay_zenith_azimuth(decay_point, Ag.axis*u.km)

        vrms = antenna.Vrms(freqs, antennas)

        Ptrig = np.zeros((Ag.stations.shape[0], Ag.trials.shape[0]))

        # iterate over stations
        for i in range(Ag.stations.shape[0]):

            distance_to_decay = np.linalg.norm(Ag.stations[i] - decay_point, axis=1)

            # calculate the view angle from the decay points
            view = geometry.decay_view(decay_point, Ag.axis, Ag.stations[i])

            altitudes = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
            detector_altitude = np.linalg.norm(Ag.stations[i]) - Re
            closest_altitude = altitudes[np.abs(altitudes - detector_altitude.value).argmin()]

            # load the field parameterization.
            voltage = EFieldParam(closest_altitude)

            # the zenith and azimuth (measured from East to North) from the station to each decay point
            theta, phi = geometry.obs_zenith_azimuth(Ag.stations[i], decay_point)

            phi_from_boresight = phi - Ag.orientations[i]

            # compute the voltage at each of these off-axis angles and at each frequency
            V = voltage(
                view.to(u.deg),
                exit_zenith.to(u.deg),
                decay_altitude,
                decay_length,
                decay_zenith.to(u.deg),
                decay_azimuth.to(u.deg),
                distance_to_decay,
                Ag.stations[i],
                Ag.dbeacon[i],
                freqs,
                Eshower,
                antennas,
                theta.to(u.deg),
                phi_from_boresight.to(u.deg),
                Ag.fov[i],
            )
            

            # calculate the SNR
            SNR = np.sum(V, axis=1) / vrms

            # and check for a trigger
            Ptrig[i] = SNR > trigger_SNR

            # and use this to compute the angle below ANITA's horizontal
            elev = (np.pi / 2.0)*u.rad - theta

            # calculate the distance (km) to the horizon from ANITA
            horizon_distance = geometry.distance_to_horizon(
                height=detector_altitude, radius=Re
            )

            # the decay points that are further away than the horizon
            beyond = distance_to_decay > horizon_distance
            del horizon_distance

            # and the particles that appear to be below the horizon
            # remember: more negative is below the horizon
            below = elev < geometry.horizon_angle(detector_altitude, radius=Re)

            # those that are beyond the horizon and below the horizon
            invisible = np.logical_and(beyond, below)
            del beyond, below

            # if the trial is invisible, there's no way we can trigger on it
            Ptrig[i][invisible] = 0.0

            # if the event is above ANITA's horizon, we would not find
            # them in the search as they would be treated as background
            Ptrig[i][elev > 0.0] = 0.0

        Pdet = np.sum(Ptrig, axis=0) > 0

        # and save the various effective area coefficients at these angles
        geometric = (Ag.area * np.sum(Ag.dot)) / N
        pexit = np.mean(Pexit)
        pdet = np.mean(Pdet)
        effective_area = np.sum(Ag.area * Ag.dot * Pexit * Pdet) / N

    # and now return the computed parameters
    #return np.array([geometric, pexit, pdet, effective_area])
    end = time.time()
    return end-start