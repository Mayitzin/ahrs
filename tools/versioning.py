# -*- coding: utf-8 -*-
"""
AHRS: Versioning
================

"""

MAJOR = 0
MINOR = 3
PATCH = 1
RELEASE = "0"

def get_version(short: bool = False) -> str:
    if short or RELEASE=="0":
        return f"{MAJOR}.{MINOR}.{PATCH}"
    return f"{MAJOR}.{MINOR}.{PATCH}-{RELEASE}"
