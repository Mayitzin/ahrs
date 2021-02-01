# -*- coding: utf-8 -*-
"""
AHRS: Versioning
================

"""

MAJOR = 0
MINOR = 3
PATCH = 0
RELEASE = "rc1"

def get_version(short=False):
    if short or RELEASE=="0":
        return "{}.{}.{}".format(MAJOR, MINOR, PATCH)
    return "{}.{}.{}-{}".format(MAJOR, MINOR, PATCH, RELEASE)
