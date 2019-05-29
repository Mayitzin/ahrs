# -*- coding: utf-8 -*-
"""
AHRS is the core package to develop and test applications for Attitude and
Heading Reference Systems.

It provides:

- classes to organize and easily identify large amounts of data.
- useful linear algebra, statistical and arithmetic functions.
- interfacing functions and classes for visualizations.
- and much more.

AHRS is compatible with Python 3 and above. Using an older version is highly
discouraged.

All AHRS wheels distributed on PyPI are BSD licensed.

"""

import sys
from setuptools import setup, find_packages
from ahrs import get_version

if sys.version_info.major < 3:
    raise RuntimeError("Python version >= 3 required.")

REPOSITORY_URL = 'https://github.com/Mayitzin/ahrs'

metadata = dict(
    name='AHRS',
    version=get_version(),
    description='Attitude and Heading Reference Systems.',
    long_description=__doc__,
    url=REPOSITORY_URL,
    download_url=REPOSITORY_URL+'-/archive/master/ahrs-master.zip',
    author='Mario Garcia',
    author_email='mario.garcia@tum.de',
    project_urls={
        "Bug Tracker": REPOSITORY_URL+"issues"
    },
    install_requires=['numpy',
                      'scipy'],
    packages=find_packages()
)

setup(**metadata)
