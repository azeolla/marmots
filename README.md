# Multiple Antenna aRrays on MOuntains Tau Sensitivity (MARMOTS)

[![Actions Status](https://github.com/azeolla/marmots/workflows/CI/badge.svg)](https://github.com/azeolla/marmots/actions)
![GitHub](https://img.shields.io/github/license/rprechelt/tapioca?logoColor=brightgreen)
![Python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Marmots (**M**ultiple **A**ntenna a**R**rays on **MO**untains **T**tau **S**ensitivity) is a suite of tools used to calculate the tau point-source sensitivity of the [Beamforming Elevated Array for Cosmic Neutrinos]() (BEACON). 

### Installation

To install `marmots`, you will need `git`, [git-lfs](https://git-lfs.github.com/), and Python >= 3.6. All three should be available in the package manager of any modern OS. It is tested on macOS 10.14, ubuntu 18.04, ubuntu 16.04, Fedora 29, and Fedora 30.

The below instructions are assuming that `python` refers to Python 3.\*. If `python` still refers to a decrepit Python 2.\*, please replace `python` with `python3` and `pip` with `pip3`.

The recommended method of installation is to first clone the package

    $ git clone https://github.com/azeolla/marmots.git
	
and then change into the cloned directory and install using `pip`

    $ cd marmots
	$ pip install --user -e .
    
To run `marmots`, you will also need a set of parametrized LUT's for the tau exit probability produced using [NuTauSim][https://github.com/harmscho/NuTauSim]]. Please contact the maintainer of this repository for access. This LUT's need to be installed into the `marmots/data/tauexit` directory as shown:

    tauexit/
    |-- 0.0km_ice_midCS_stdEL
    |   |-- LUT_1e+15_eV.npz
        ...
    |   `-- LUT_3e+20_eV.npz
    |-- 1.0km_ice_midCS_stdEL
    |   |-- LUT_1e+15_eV.npz
        ...
    |   `-- LUT_3e+20_eV.npz
    |-- 2.0km_ice_midCS_stdEL
    |   |-- LUT_1e+15_eV.npz
        ...
    |   `-- LUT_3e+20_eV.npz
    |-- 3.0km_ice_midCS_stdEL
    |   |-- LUT_1e+15_eV.npz
        ...
    |   `-- LUT_3e+20_eV.npz
    `-- 4.0km_ice_midCS_stdEL
        |-- LUT_1e+15_eV.npz
        ...
        `-- LUT_3e+20_eV.npz
    
#### Testing and Development 
    
Once the data files are installed, you can verify that the installation was successful by trying to import `marmots`

    $ python -c 'import marmots'

If you wish to develop new features in `marmots`, you will also need to install some additional dependencies so you can run our unit tests

    $ pip install --user -e .[test]
	
Once that is completed, you can run the unit tests directory from the `marmots` directory

    $ python -m pytest tests


### Usage

To calculate the acceptance of BEACON at different energies and configurations, use the `marmots` script that was installed onto your PATH (or under `marmots/scripts/marmots`).

    $ marmots -h 
    
will print useful documentation. For example, to calculate the effective area of BEACON's 2018 prototype to 1 EeV neutrinos using 10,000 Monte Carlo trials per elevation step, run:

    $ marmots --Enu 1 --prototype 2018 --ntrials 10_000

The `skymap` script can produce skymaps of the instantaneous effective area of BEACON as a function of right-ascension and declination.

    $ skymap --Enu 1 --ntrials 50_000 --prototype 2018

will produce a skymap of instantaneous effective area using 50,000 MC trials for BEACON's 2018 prototype.
