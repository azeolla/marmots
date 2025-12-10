# Multiple Antenna Arrays on Mountains Tau Sensitivity (Marmots)

[![Actions Status](https://github.com/azeolla/marmots/workflows/CI/badge.svg)](https://github.com/azeolla/marmots/actions)
![GitHub](https://img.shields.io/github/license/rprechelt/tapioca?logoColor=brightgreen)
![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Marmots (**M**ultiple **A**ntenna A**r**rays on **Mo**untains **T**au **S**ensitivity) is a suite of tools used to calculate the tau point-source sensitivity of the [Beamforming Elevated Array for Cosmic Neutrinos](https://arxiv.org/abs/2504.13271) (BEACON). 

### Installation

To install `marmots`, you will need `git`, [git-lfs](https://git-lfs.github.com/), and Python >= 3.6. All three should be available in the package manager of any modern OS. It is tested on macOS 10.14, ubuntu 18.04, ubuntu 16.04, Fedora 29, and Fedora 30.

The below instructions are assuming that `python` refers to Python 3.\*. If `python` still refers to a decrepit Python 2.\*, please replace `python` with `python3` and `pip` with `pip3`.

The recommended method of installation is to first clone the package

    $ git clone https://github.com/azeolla/marmots.git
	
To run marmots, you will need to extract the data files:

    $ cd marmots
    $ tar -xvzf data.tar.gz

To use the topography branch, you will also need to download either all or at least part of the SRTM 30m database. The entire database is ~350 GB. Once you have downloaded the necessary .hgt files, you must specify their path with environment variable SRTM_DATA_DIR. The files should be formatted as {lat}{lon}.hgt, for example: N33W128.hgt.
    
Finally, you can install Marmots using pip:

    $ pip install --user -e .


### Usage

To calculate the acceptance of BEACON at different energies and configurations, use the `skymap` script that was installed onto your PATH (or under `marmots/scripts/skymap`).

    $ skymap -h 
    
will print useful documentation. The argument "--stations" should be point to a text file containing 5 columns: latitude (degrees), longitude (degrees), altitude (km), orientation (degrees relative to geographic east), field-of-view (degrees), and number of phased antennas. The "--nside" argument should be a power of 2 and specifies the resolution of the skymap. nside=16 corresponds to a skymap of 3,072 pixels (# of pixels = 12 * nside**2).

As an example, to calculate the instantaneous effective area of 100 BEACON stations to 1 EeV neutrinos using 100,000 Monte Carlo trials per pixel, run:

    $ skymap --nside 16 --Enu 1 --model prototype --ntrials 100_000 --stations /../stations_example.txt
