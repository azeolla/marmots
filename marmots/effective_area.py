"""
This module provides the high-level event loop to calculate
the tau point source effective area. 
Modification from the original; it takes into account the tau propagation through terrain, using the TauSurvLUT class and the propagate_tracks function.
"""

import numpy as np

from marmots.constants import Re
import marmots.geometry as geometry
import marmots.grammage as grammage
import marmots.decay as decay
import marmots.topography as topography
from marmots.tau_propagator import TauSurvLUT, propagate_tracks


def calculate(
    ra: float,
    dec: float,
    totalmesh,
    BVH,
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
    psurvival_lut: TauSurvLUT = None,
    maxview: float = np.radians(3.0),
    N: int = 1_000_000,
    freqs: np.ndarray = np.arange(30, 80, 10) + 5,
    trigger_SNR: float = 5.0,
    min_elev: float = np.deg2rad(-30),
    time: str = '2025-03-20 12:00:00',
    tau_prop_model: int = 0,
) -> np.ndarray:
    """
    Calculate the effective area of an observatory to a point source tau flux.

    Parameters
    ----------
    ra: float
        Right ascension of the point source.
    dec: float
        Declination of the point source.
    totalmesh: NamedTuple
        Output of topography.horizon_mesh
    BVH: class
        Bounding Volume Hierarchy (from mesh.py)
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
    psurvival_lut: TauSurvLUT, optional
        Look-Up Table for tau survival probability in rock (from tau_propagator.py).
        If None, it is instantiated automatically from the default file.
        Pass an already-loaded instance when calling calculate() in a loop
        to avoid reloading the file on every call.
    maxview: float
        The maximum view angle (in radians).
    N: int
        The number of trials (exit points) to generate.
    freqs: np.ndarray
        The center frequencies of 10 MHz bins spanning the bandwidth.
        Ex. For 30-80 MHz, freqs = [35, 45, 55, 65, 75]
    trigger_SNR: float
        The voltage SNR needed for a trigger.
    min_elev: float
        Elevation angles below this threshold will not be simulated.
    time: str
        The time in which to calculate the instantaneous effective area.
    tau_prop_model: int
        Energy loss model for tau propagation through rock: 0=ALLM (default), 1=ASW.

    Returns
    -------
    Aeff: np.ndarray
        A collection of effective area components at the specified ra and dec:
        [geometric, pexit, pdet, effective_area, coincidence_frac]
    """

    # load the survival LUT if not provided externally
    if psurvival_lut is None:
        psurvival_lut = TauSurvLUT()

    # compute the geometric area associated with a point source
    Ag = geometry.geometric_area(
        ra, dec, totalmesh, maxview, antennas, N=N, min_elev=min_elev, time=time
    )

    if Ag.N == 0:
        geometric        = 0
        pexit            = 0
        pdet             = 0
        effective_area   = 0
        coincidence_frac = np.nan

    else:

        # determine the grammage and convert to exit angle for the LUT
        exit_theta = grammage.find_exit_angle(Ag.trials, Ag.axis, totalmesh, BVH)

        # exit probability and tau energy at exit point
        Pexit, Etau = tauexit(exit_theta)

        # initial decay length and shower energy — sampled with energy at exit
        decay_length = taudecay.sample_range(Etau)
        Eshower      = taudecay.shower_energy(Etau)

        # initial decay point
        decay_point = Ag.trials + (Ag.axis[:, None] * decay_length).T

        # ----------------------------------------------------------------
        # TAU PROPAGATION: correct tracks that cross terrain
        # ----------------------------------------------------------------

        # check which tracks have terrain between exit and decay point
        path_blocked = ~decay.tau_path_clear(
            Ag.trials, decay_point, totalmesh, BVH
        )  # shape (N,), True where terrain is crossed

        if np.any(path_blocked):

            # propagate blocked tracks through terrain segments:
            #   - compute Psurv_tot = prod(Psurv_air_i * Psurv_rock_i)
            #   - compute E_final after all rock energy losses
            #   - re-sample decay point in last air segment with E_final
            Psurv_tot, E_final, decay_point_new = propagate_tracks(
                Ag.trials[path_blocked],
                decay_point[path_blocked],
                Etau[path_blocked],
                totalmesh,
                BVH,
                psurvival_lut,
                taudecay,
                Ag.axis,
                model=tau_prop_model,
            )

            # update Pexit with terrain survival probability weight
            Pexit[path_blocked] *= Psurv_tot

            # update shower energy and decay point with corrected quantities
            Eshower[path_blocked]     = taudecay.shower_energy(E_final)
            decay_point[path_blocked] = decay_point_new

            # recompute decay_length for blocked tracks
            # (needed downstream by voltage())
            decay_length[path_blocked] = np.linalg.norm(
                decay_point[path_blocked] - Ag.trials[path_blocked], axis=1
            )

        # ----------------------------------------------------------------
        # rest of the calculation — unchanged from original
        # ----------------------------------------------------------------

        decay_point_geodetic = topography.to_geodetic(decay_point * 1e3)
        decay_altitude       = decay_point_geodetic[:, 2] / 1e3

        exit_zenith                 = geometry.exit_zenith(Ag.trials, Ag.axis)
        decay_zenith, decay_azimuth = geometry.decay_zenith_azimuth(decay_point, Ag.axis)

        vrms       = detector.Vrms(freqs)
        n_stations = len(Ag.stations)
        triggers   = np.zeros(Ag.trials.shape[0])

        for i in range(n_stations):

            ground_view = geometry.view_angle(
                Ag.trials, Ag.stations[i]["geocentric"], Ag.axis
            )

            trigger = np.zeros(Ag.trials.shape[0])
            in_view = ground_view <= maxview

            # line-of-sight: decay point → station
            LoS = decay.line_of_sight(
                decay_point[in_view], Ag.stations[i]["geocentric"], totalmesh, BVH
            )
            in_view[in_view] = LoS

            if np.sum(in_view) == 0:
                continue

            distance_to_decay = geometry.norm(
                Ag.stations[i]["geocentric"] - decay_point[in_view]
            )

            decay_view = geometry.view_angle(
                decay_point[in_view], Ag.stations[i]["geocentric"], Ag.axis
            )

            theta, phi = geometry.obs_zenith_azimuth(
                Ag.stations[i], decay_point[in_view], decay_point_geodetic[in_view]
            )

            phi_from_boresight = (
                phi - np.deg2rad(Ag.orientations[i]) + np.pi
            ) % (2 * np.pi) - np.pi

            detector_altitude = Ag.stations[i]["geodetic"][2] / 1e3

            dobservatory = geometry.norm(
                Ag.stations[i]["geocentric"] - Ag.trials[in_view]
            )

            V= voltage(
                np.rad2deg(decay_view),
                np.rad2deg(exit_zenith[in_view]),
                decay_altitude[in_view],
                decay_length[in_view],
                np.rad2deg(decay_zenith[in_view]),
                np.rad2deg(decay_azimuth[in_view]),
                distance_to_decay,
                detector_altitude,
                Ag.stations[i]["geodetic"],
                dobservatory,
                freqs,
                Eshower[in_view],
                np.rad2deg(theta),
                np.rad2deg(phi_from_boresight),
                Ag.fov[i],
                detector,
                decay_point[in_view],
                Ag.stations[i]["geocentric"],
            )

            # SNR: max over frequency bins for the trigger decision
            SNR = np.sqrt(Ag.antennas[i]) * (V / vrms)
            #print(f"V shape: {V.shape}, SNR shape: {SNR.shape}, in_view sum: {np.sum(in_view)}")
            trigger[in_view] = SNR > trigger_SNR
            triggers         = triggers + trigger

        coincidences = np.sum(triggers > 1)
        Pdet         = triggers > 0
        num_triggers = np.sum(Pdet)

        geometric      = (Ag.area * np.sum(Ag.dot)) / Ag.N
        pexit          = np.mean(Pexit)
        pdet           = np.mean(Pdet)
        effective_area = np.sum(Ag.area * Ag.dot * Pexit * Pdet) / Ag.N

        with np.errstate(divide='ignore', invalid='ignore'):
            coincidence_frac = coincidences / num_triggers

    return np.array([geometric, pexit, pdet, effective_area, coincidence_frac])