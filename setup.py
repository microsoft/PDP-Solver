#!/usr/bin/env python3

# To make sure pylint is happy with the tests, run:
#   python3 setup.py develop
# every time we add or remove new top-level packages to the project.

import os
from setuptools import setup


def _read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as stream:
        return stream.read()


setup(
    name='PDP_Solver',
    version='0.1',
    author="Saeed Amizadeh",
    author_email="saamizad@microsoft.com",
    description="PDP Solver implementation for the paper",
    license="MIT",
    keywords="example documentation tutorial",
    url="https://github/Microsoft/PDP-Solver",
    package_dir={"": "src"},
    packages=["pdp"],
    long_description=_read('./README.md')
)
