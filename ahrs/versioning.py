# -*- coding: utf-8 -*-
"""
AHRS: Versioning
================

"""

MAJOR = "0"
MINOR = "1"
PATCH = "2"
RELEASE = "8"

def get_version(short=False):
    if short:
        return "{}.{}.{}".format(MAJOR, MINOR, PATCH)
    return "{}.{}.{}-{}".format(MAJOR, MINOR, PATCH, RELEASE)
