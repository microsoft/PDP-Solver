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
    author="Saeed Amizadeh",
    author_email="saamizad@microsoft.com",
    description="PDP Solver implementation for the paper",
    long_description=_read('./README.md'),
    keywords="pdp sat solver pytorch paper icml",
    license="MIT",
    url="https://github/Microsoft/PDP-Solver",
    version='0.1',
    python_requires=">=3.5",
    install_requires=[
        "numpy >= 1.10",
        "torch >= 0.4",
    ],
    package_dir={"": "src"},
    packages=[
        "pdp",
        "pdp.factorgraph",
        "pdp.nn"
    ],
    scripts=[
        "src/satyr.py",
        "src/satyr-train-test.py",
        "src/dimacs2json.py",
        "src/pdp/generator.py"
    ]
)
