# -*- coding: utf-8 -*-
"""
AHRS
====

The versioning follows the principles of Semantic Versioning as specified in
https://semver.org

"""

from . import common
from . import filters
from . import utils

MAJOR = "0"
MINOR = "1"
PATCH = "1"
RELEASE = "alpha.1"
VERSION = "{}.{}.{}-{}".format(MAJOR, MINOR, PATCH, RELEASE)

def get_version(short=False):
    if short:
        return "{}.{}.{}".format(MAJOR, MINOR, PATCH)
    return VERSION
