# -*- coding: utf-8 -*-
"""
AHRS is the core package to develop and test applications for Attitude and
Heading Reference Systems.

It provides:

- classes to organize and easily identify large amounts of data.
- useful linear algebra, statistical and arithmetic functions.
- interfacing functions and classes for visualizations.
- and much more.

AHRS is compatible with Python 3.6 and above. Using an older version is highly
discouraged.

All AHRS wheels distributed on PyPI are MIT licensed.

"""

import sys
from setuptools import setup, find_packages
from ahrs import get_version

if sys.version_info < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

REPOSITORY_URL = 'https://github.com/Mayitzin/ahrs/'

with open("README.md", "r") as fh:
    long_description = fh.read()

metadata = dict(
    name='AHRS',
    version=get_version(),
    description='Attitude and Heading Reference Systems.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=REPOSITORY_URL,
    download_url=REPOSITORY_URL+'archive/master/ahrs-master.zip',
    author='Mario Garcia',
    author_email='mario.garcia@tum.de',
    project_urls={
        "Bug Tracker": REPOSITORY_URL+"issues"
    },
    install_requires=['numpy',
                      'scipy',
                      'matplotlib'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Embedded Systems',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
],
    packages=find_packages()
)

setup(**metadata)
