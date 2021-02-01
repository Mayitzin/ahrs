# -*- coding: utf-8 -*-
"""
AHRS is the core package to develop and test applications for Attitude and
Heading Reference Systems.

It provides:

- useful linear algebra, statistical and arithmetic functions.
- classes to organize and easily identify specialized data.
- algorithms to estimate attitude and heading.
- and much more.

AHRS is compatible with Python 3.6 and above. Using an older version is highly
discouraged.

All AHRS wheels distributed on PyPI are MIT licensed.

"""

import sys
from setuptools import setup, find_packages
from tools.versioning import get_version

if sys.version_info < (3, 6):
    raise SystemError("Python version >= 3.6 required.")

__version__ = get_version()

REPOSITORY_URL = 'https://github.com/Mayitzin/ahrs/'

with open("README.md", "r") as fh:
    long_description = fh.read()

metadata = dict(
    name='AHRS',
    version=__version__,
    description='Attitude and Heading Reference Systems.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=REPOSITORY_URL,
    download_url=REPOSITORY_URL+'archive/master/ahrs-master.zip',
    author='Mario Garcia',
    author_email='mariogc@protonmail.com',
    project_urls={
        "Source Code": REPOSITORY_URL,
        "Bug Tracker": REPOSITORY_URL+"issues"
    },
    install_requires=[
        'numpy',
        'scipy'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Embedded Systems',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    include_package_data=True,
    packages=find_packages()
)

setup(**metadata)
