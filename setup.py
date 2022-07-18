from setuptools import setup, find_packages
from os import path
from marmots import __version__

# get the absolute path of this project
here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# the standard setup info
setup(
    name='marmots',
    version=__version__,
    description=('A Python package for evaluating tau point '
                 'source sensitivities of radio neutrino experiments.'),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/azeolla/marmots',
    author='Andrew J. Zeolla and Remy L. Prechelt',
    author_email='azeolla@psu.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='BEACON science neutrino point source',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    python_requires='>=3.6*, <4',
    install_requires=['uproot', 'numpy', 'matplotlib', 'astropy', 'tqdm', 'interpolation'],
    extras_require={
        'test': ['pytest', 'coverage'],
    },
    scripts=['scripts/skymap'],
    project_urls={
        'NuTauSim': 'https://github.com/harmscho/NuTauSim',
        'BEACON Github': 'https://github.com/beaconTau/',
    },
)
