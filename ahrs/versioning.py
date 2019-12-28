# -*- coding: utf-8 -*-
"""
AHRS: Versioning
================

"""

MAJOR = 0
MINOR = 2
PATCH = 1
RELEASE = "0"

def get_version(short=False):
    if short or RELEASE == "0":
        return "{}.{}.{}".format(MAJOR, MINOR, PATCH)
    return "{}.{}.{}-{}".format(MAJOR, MINOR, PATCH, RELEASE)
