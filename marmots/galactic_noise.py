"""
This script allows to simualte sky/galactic noise from modern models of the sky brigthness temperature.
The models are spatial and frequency dependent. An antenna model is needed to simulate the signal in a
channel.

This module is an adaptation of the channelGalacticNoiseAdder from NuRadioReco (https://github.com/nu-radio/NuRadioMC).

Unlike NuRadio this file uses MHz as base unit for frequency (instead of GHz) and degrees for angles (instead of rad)
"""

import warnings
import functools
import numpy as np

from scipy import interpolate, constants, integrate
from numpy.random import Generator, Philox

import healpy
import astropy.coordinates
import astropy.units

import logging
logger = logging.getLogger()

# This is the maximum number caching entries for the vector effective length and
# noise temperature.
maxsize = 1024 * 4

degree = np.pi / 180  # converts deg to rad

try:
    from pygdsm import (
        GlobalSkyModel16,
        GlobalSkyModel,
        LowFrequencySkyModel,
        HaslamSkyModel,
    )
except ImportError as e:
    logger.error(
        "To use the galactic_noise.py, 'pygdsm' needs to be installed:\n\n"
        "\t pip install git+https://github.com/telegraphic/pygdsm\n"
        )
    raise(e)


def get_electric_field_from_temperature(frequencies, noise_temperature, solid_angle):
    """
    Calculate the electric field amplitude from the radiance of a radio signal.

    The radiance is calculated using the Rayleigh-Jeans law per frequency bin, by adjusting
    the value with the frequency spacing. After this, the electric field amplitude per bin
    is calculated using the radiance and the vacuum permittivity.

    This function  assumes NuRadio's FFT normalization. See ``freq2time``!

    Parameters
    ----------
    frequencies: array of floats
        The frequencies at which to calculate the electric field amplitude
    noise_temperature: float
        The noise temperature to use in the Rayleigh-Jeans law
    solid_angle: float
        The solid angle over which the radiance is integrated

    Returns
    -------
    efield_amplitude: array of floats
        The electric field amplitude at each frequency
    """
    c_vac = constants.c / 1e6  # Hz -> MHz

    frequencies = np.copy(frequencies)

    # Calculate frequency spacing
    d_f = frequencies[2] - frequencies[1]

    # Calculate spectral radiance of radio signal using Rayleigh-Jeans law
    spectral_radiance = (
        2.0 * constants.k * frequencies**2 * noise_temperature / c_vac**2
    )
    spectral_radiance[np.isnan(spectral_radiance)] = 0

    # calculate radiance per energy bin, e.g., multiplying with the frequency spacing and solid angle
    radiance_per_bin = spectral_radiance * d_f * solid_angle

    # calculate electric field per energy bin from the radiance per bin
    # 1 / (c_vac * epsilon_0) = Z_0 the vaccum impedance, d_f term due to our fft definition
    efield_amplitude = np.sqrt(radiance_per_bin / (c_vac * constants.epsilon_0)) / d_f

    return efield_amplitude


@functools.lru_cache(maxsize=1024)
def get_frequencies(num_samples, sampling_rate):
    """
    Returns frequency bins for FFT.

    Parameters
    ----------
    num_samples: int
        The number of sample in the time domain
    sampling_rate: float
        Sampling rate of the trace

    Returns
    -------
    frequencies: np.array
        Frequency binning
    """
    return np.fft.rfftfreq(num_samples, d = 1 / sampling_rate)


def freq2time(spectrum, sampling_rate, n=None):
    """
    Performs backward FFT with correct normalization that conserves the power.

    Parameters
    ----------
    spectrum: complex np array
        the frequency spectrum
    sampling_rate: float
        sampling rate of the spectrum
    n: int
        the number of sample in the time domain (relevant if time trace has an odd number of samples)

    Returns
    -------
    trace: np.array
        Time series/trace
    """
    return np.fft.irfft(spectrum, axis=-1, n=n) * sampling_rate / 2 ** 0.5


class GalacticNoiseSimulator:
    """
    Class that simulates the noise produced by galactic radio emission

    Uses the pydgsm package (https://github.com/telegraphic/pygdsm), which provides
    radio background data based on Oliveira-Costa et al. (2008) (https://arxiv.org/abs/0802.1525)
    and Zheng et al. (2016) (https://arxiv.org/abs/1605.04920)

    The radio sky model is evaluated on a number of points above the horizon
    folded with the antenna response. Since evaluating every frequency individually
    would be too slow, the model is evaluated for a few frequencies and the log10
    of the brightness temperature is interpolated in between.
    """

    def __init__(self):
        self.__n_side = None
        self.__interpolation_frequencies = None
        self.__radio_sky = None
        self.__noise_temperatures = None

    def begin(
            self,
            skymodel=None,
            debug=False,
            n_side=4,
            freq_range=None,
            seed=None,
            caching=True,
            scaling=1.0
    ):
        """
        Set up important parameters for the module

        Parameters
        ----------
        skymodel: {'gsm2008', 'lfmap', 'lfss', 'gsm2016', 'haslam'}, optional
            Choose the sky model to use. If none is provided, the Global Sky Model (2008) is used as a default.
        debug: bool, default: False
            Deprecated. Will be removed in future versions.
        n_side: int, default: 4
            The n_side parameter of the healpix map. Has to be power of 2
            The radio skymap is downsized to the resolution specified by the n_side
            parameter and for every pixel above the horizon the radio noise coming
            from that direction is calculated. The number of pixels used is
            12 * n_side ** 2, so a larger value for n_side will result better accuracy
            but also greatly increase computing time.
        freq_range: array of len=2, default: [10, 1000]
            The sky brightness temperature will be evaluated for the frequencies
            within this limit. Brightness temperature for frequencies in between are
            calculated by interpolation the log10 of the temperature
            The interpolation_frequencies have to cover the entire passband
            specified in the run method.
        seed : {None, int, array_like[ints], SeedSequence}, optional
            The seed that is passed on to the `numpy.random.Philox` bitgenerator used for random
            number generation.
        caching: bool, default: True
            If True, the antenna response is cached for each channel. This can speed up this module
            by a lot. If the frequencies of the channels change, the cache is cleared.
        scaling: float, default: 1.0
            Scaling factor for the noise. This is useful when doing interferometry with extremely large arrays
            such as SKA-low. For such an array it is very expensive to simulate/interpolate/process all antennas.
            Instead, one can use every nth antenna and scale the noise by a factor of 1/ sqrt(n) (since the SNR
            is expected to scale with the square root of the number of antennas when using interferomtery/beamforming).
        """
        if debug:
            warnings.warn("This argument is deprecated and will be removed in future versions.", DeprecationWarning)

        self.__random_generator = Generator(Philox(seed))
        self.__n_side = n_side
        self.solid_angle = healpy.pixelfunc.nside2pixarea(self.__n_side, degrees=False)

        self.__caching = caching
        self.scaling = scaling
        self.__freqs = None
        if self.__caching and 12 * n_side ** 2 * 2 > maxsize:
            logger.warning(
                f"Caching for the vector effective length is enabled (with `maxsize={maxsize}`) and `n_side={n_side}` is to large, and thus "
                "it produces to many different caching entries for two antenna models to be stored of one `station_time`. "
                "Either decrease `n_side` or increase `maxsize` (has to be done in the source code).")

        if freq_range is None:
            freq_range = np.array([10, 1000])

        # define interpolation frequencies. Set in logarithmic range from freq_range[0] to freq_range[1],
        # rounded to MHz to avoid import errors from LFmap and tabulated models.
        self.__interpolation_frequencies = np.around(np.logspace(*np.log10(freq_range), num=15), 3)

        # initialise sky model
        try:
            if skymodel is None:
                sky_model = GlobalSkyModel(freq_unit="MHz")
                logger.info("No sky model specified. Using standard: Global Sky Model (2008). Available models: "
                            "gsm2008, lfmap, lfss, gsm2016, haslam")
            elif skymodel.lower() == 'lfss':
                sky_model = LowFrequencySkyModel(freq_unit="MHz")
                logger.info("Using LFSS as sky model")
            elif skymodel.lower() == 'gsm2008':
                sky_model = GlobalSkyModel(freq_unit="MHz")
                logger.info("Using GSM2008 as sky model")
            elif skymodel.lower() == 'gsm2016':
                sky_model = GlobalSkyModel16(freq_unit="MHz")
                logger.info("Using GSM2016 as sky model")
            elif skymodel.lower() == 'haslam':
                sky_model = HaslamSkyModel(freq_unit="MHz", spectral_index=-2.53)
                logger.info("Using Haslam as sky model")
            else:
                logger.error(f"Sky model {skymodel} unknown. Defaulting to Global Sky Model (2008).")
                sky_model = GlobalSkyModel(freq_unit="MHz")

        except NameError:
            logger.error(f"Could not find {skymodel} skymodel. Do you have the correct package installed? \n"
                        f"Defaulting to Global Sky Model (2008) as sky model.")
            sky_model = GlobalSkyModel(freq_unit="MHz")

        self.__noise_temperatures = np.zeros(
            (len(self.__interpolation_frequencies), healpy.pixelfunc.nside2npix(self.__n_side))
        )
        logger.info("Generating noise temperatures ..")

        # generating sky maps and noise temperatures from chosen sky model in given frequency range
        for i_freq, noise_freq in enumerate(self.__interpolation_frequencies):
            self.__radio_sky = sky_model.generate(noise_freq)  # passed in MHz
            self.__radio_sky = healpy.pixelfunc.ud_grade(self.__radio_sky, self.__n_side)
            self.__noise_temperatures[i_freq] = self.__radio_sky

        # We can not already interpolate the efield amplitudes because for their normalization the
        # frequency resoltion matters.
        self.__noise_temperature_funcs = np.array([
            interpolate.interp1d(
                self.__interpolation_frequencies,
                np.log10(self.__noise_temperatures[:, i_pixel]), kind='quadratic')
            for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side))
        ])

    @functools.lru_cache(maxsize=maxsize)
    def _get_cached_noise_temperature_for_pixel(self, i_pixel):
        """
        Returns the cached electric field amplitude for a given pixel.
        This wrapper is necessary as arrays and list are not
        hashable (i.e., can not be used as arguments in functions one wants to cache).
        This module ensures that the cache is clearied if the vector `self.__freqs` changes.
        """
        return np.power(10, self.__noise_temperature_funcs[i_pixel](self.__freqs))

    def _check_cache(self, freqs):
        # If we cache the antenna pattern / sky noise temperature, we need to make sure that the frequencies have not changed
        # between stations. If they have, we need to clear the cache.
        if self.__caching:
            if self.__freqs is None:
                self.__freqs = freqs
            else:
                if len(self.__freqs) != len(freqs):
                    self.__freqs = freqs
                    self._get_cached_antenna_response.cache_clear()
                    self._get_cached_noise_temperature_for_pixel.cache_clear()
                    logger.warning(
                        "Frequencies have changed (array length). Clearing antenna response / efield cache. "
                        "(If this happens often, something might be wrong...")
                elif not np.allclose(self.__freqs, freqs, rtol=0, atol=0.01 ):
                    self.__freqs = freqs
                    self._get_cached_antenna_response.cache_clear()
                    self._get_cached_noise_temperature_for_pixel.cache_clear()
                    logger.warning(
                        "Frequencies have changed (values). Clearing antenna response / efield cache. "
                        "(If this happens often, something might be wrong...")

    def _get_noise_temperature(self, i_pixel, freqs=None):
        if self.__caching:
            noise_temperature = self._get_cached_noise_temperature_for_pixel(i_pixel)
        else:
            noise_temperature = np.power(10, self.__noise_temperature_funcs[i_pixel](freqs))

        return noise_temperature

    def run(
            self,
            n_samples,
            sampling_rate,
            antennas,
            location,
            time,
            passband=None,
            homogenous_temperature=None,
            mountain_elevation=0,  # degree
            mountain_temperature=300,  # Kelvin
    ):

        """
        Adds noise resulting from galactic radio emission to the channel traces

        Parameters
        ----------
        n_samples: int
            Number of samples in returned waveforms
        samping_rate: float
            Sampling rate of returned waveforms in MHz
        antennas: list of antenna ob objects
            Off type antenna.Detector
        location: tuple of floats
            Longitude and Latitude of your detector in degree
        time: astropy time object
            Time you want to simulate
        passband: list of float, optional
            Lower and upper bound of the frequency range in which noise shall be
            added. The default (no passband specified) is [10, 1000] MHz
        homogenous_temperature: float, optional
            If set simulate antenna in a homogenous world of the given temperature (for debugging)
        mountain_elevation: float, optional
            Elevation angle (in degree) below which the mountain_temperature is used to simulate the
            antenna signal. Default 0 degree
        mountain_temperature: float, optional
            Temperature of mountain/ground. Default: 300K

        Returns
        -------
        waveforms: list of arrays(float)
            Waveform per antenna
        """

        if self.__noise_temperatures is None: # check if .begin has been called, give helpful error message if not
            msg = "GalacticNoiseSimulator was not initialized correctly. Maybe you forgot to call `.begin()`?"
            logger.error(msg)
            raise ValueError(msg)

        if passband is None:
            passband = [10 , 1000]

        freqs = get_frequencies(n_samples, sampling_rate)

        passband_filter = (freqs > passband[0]) & (freqs < passband[1])

        self._check_cache(freqs[passband_filter])

        site_latitude, site_longitude = location

        local_coordinates = get_local_coordinates((site_latitude, site_longitude), time, self.__n_side)

        channel_spectra = {}
        for idx in range(len(antennas)):
            channel_spectra[idx] = np.zeros_like(freqs, dtype=complex)


        for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side)):
            azimuth = local_coordinates[i_pixel].az.rad
            zenith = np.pi / 2. - local_coordinates[i_pixel].alt.rad # this is the in-air zenith

            if homogenous_temperature is not None:
                noise_temperature = homogenous_temperature * np.ones(np.sum(passband_filter))
            elif 90 * degree - zenith < mountain_elevation * degree:
                noise_temperature = mountain_temperature * np.ones(np.sum(passband_filter))
            else:
                noise_temperature = self._get_noise_temperature(i_pixel, freqs[passband_filter])

            efield_amplitude = get_electric_field_from_temperature(
                freqs[passband_filter], noise_temperature, self.solid_angle)

            # assign random phases to electric field
            noise_spectrum = np.zeros((3, freqs.shape[0]), dtype=complex)
            phases = self.__random_generator.uniform(0, 2. * np.pi, len(efield_amplitude))

            noise_spectrum[1][passband_filter] = np.exp(1j * phases) * efield_amplitude
            noise_spectrum[2][passband_filter] = np.exp(1j * phases) * efield_amplitude

            channel_noise_spec = np.zeros_like(noise_spectrum)

            for idx, antenna in enumerate(antennas):

                # add random polarizations and phase to electric field
                polarizations = self.__random_generator.uniform(0, 2. * np.pi, len(efield_amplitude))

                channel_noise_spec[1][passband_filter] = noise_spectrum[1][passband_filter] * np.cos(polarizations)
                channel_noise_spec[2][passband_filter] = noise_spectrum[2][passband_filter] * np.sin(polarizations)

                # pass direction in degree
                antenna_response = antenna.vel2(freqs[passband_filter], zenith / degree, azimuth / degree)

                channel_noise_spectrum = (
                    antenna_response['theta'] * channel_noise_spec[1][passband_filter]
                    + antenna_response['phi'] * channel_noise_spec[2][passband_filter]
                )

                # scale noise spectrum:
                channel_noise_spectrum *= self.scaling

                # add noise spectrum from pixel in the sky to channel spectrum
                channel_spectra[idx][passband_filter] += channel_noise_spectrum

        return [freq2time(spec, sampling_rate) for spec in channel_spectra.values()]


    def get_electric_field_strength(
            self, location, obs_time, n_samples, sampling_rate, bandpass=None):
        """
        Returns the electric field strength at a given location and time

        Parameters
        ----------
        location: tuple of floats
            The latitude and longitude in deg.
        obs_time: astropy.time.Time
            The time at which the electric field strength is calculated
        n_samples: int
            The number of samples in the time domain
        sampling_rate: float
            The sampling rate of the trace
        bandpass: list of floats, optional
            The lower and upper bound of the frequency range in which the electric field strength
            shall be calculated. By default no bandpass is applied (frequency range is from
            0 to sampling_rate / 2)

        Returns
        -------
        electric_field_strength: float
            The electric field strength at the given location and time
        """

        local_coordinates = get_local_coordinates(location, obs_time, self.__n_side)

        if bandpass is None:
            bandpass = [10 , sampling_rate / 2]

        freqs = get_frequencies(n_samples, sampling_rate)
        spectrum = np.zeros_like(freqs, dtype=complex)

        window = np.zeros_like(freqs, dtype=bool)
        window[np.logical_and(bandpass[0] < freqs, freqs < bandpass[1])] = True

        self._check_cache(freqs[window])

        for i_pixel in range(healpy.pixelfunc.nside2npix(self.__n_side)):
            zenith = np.pi / 2. - local_coordinates[i_pixel].alt.rad # this is the in-air zenith

            if zenith > 90. * degree:
                continue

            noise_temperature = self._get_noise_temperature(i_pixel, freqs[window])
            efield_amplitude = get_electric_field_from_temperature(
                freqs[window], noise_temperature, self.solid_angle)


            phases = self.__random_generator.uniform(0, 2. * np.pi, len(efield_amplitude))
            spectrum_pixel = np.exp(1j * phases) * efield_amplitude
            spectrum[window] += spectrum_pixel

        return np.std(freq2time(spectrum, sampling_rate))


@functools.lru_cache(maxsize=1)
def get_local_coordinates(coordinates, obs_time, n_side):
    """
    Calculates the local coordinates of the pixels of a healpix map given the site coordinates and time.

    Parameters
    ----------
    coordinates: tuple of float
        The latitude and longitude of the site
    obs_time: astropy.time.Time
        The time at which the observation is made (station time)
    n_side: int
        The n_side parameter of the healpix map

    Returns
    -------
    local_coordinates: astropy.coordinates.SkyCoord
        The local coordinates of the pixels of the healpix map
    """
    site_latitude, site_longitude = coordinates
    site_location = astropy.coordinates.EarthLocation(
        lat=site_latitude * astropy.units.deg, lon=site_longitude * astropy.units.deg)

    local_cs = astropy.coordinates.AltAz(location=site_location, obstime=obs_time)

    # because `lonlat=True` function returns angles in degrees
    pixel_longitudes, pixel_latitudes = healpy.pixelfunc.pix2ang(
        n_side, range(healpy.pixelfunc.nside2npix(n_side)), lonlat=True)

    # First convert deg to rad using the NuRadio unit system
    # Than convert them to astropy.Quantities to be used with the
    # astropy class.
    pixel_longitudes = pixel_longitudes * degree * astropy.units.rad
    pixel_latitudes = pixel_latitudes * degree * astropy.units.rad

    galactic_coordinates = astropy.coordinates.Galactic(
        l=pixel_longitudes, b=pixel_latitudes)

    local_coordinates = galactic_coordinates.transform_to(local_cs)
    return local_coordinates



def calculate_vrms_from_temperature(temperature, bandwidth, impedance=50):
    """ Helper function to calculate the noise vrms from a given noise temperature and bandwidth.

    For details see https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise
    (sec. "Maximum transfer of noise power") or our wiki
    https://nu-radio.github.io/NuRadioMC/NuRadioMC/pages/HDF5_structure.html

    Parameters
    ----------
    temperature: float
        The noise temperature of the channel in Kelvin
    bandwidth: float or tuple of 2 floats (list of 2 floats)
        If single float, this argument is interpreted as the effective bandwidth. If tuple, the argument is
        interpreted as the lower and upper frequency of the bandwidth. In units of MHz.
    impedance: float (default: 50)
        Electrical impedance of the channel in Ohm.

    Returns
    -------
    vrms_per_channel: float
        The vrms of the channel
    """

    # (effective) bandwidth, i.e., \Delta f in equation
    if not isinstance(bandwidth, (float, int)):
        bandwidth = bandwidth[1] - bandwidth[0]

    bandwidth *= 1e6  # from MHz to Hz

    return (temperature * impedance * bandwidth * constants.k) ** 0.5