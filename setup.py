#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="reg4opt",
    version="0.1.0",
    author="Nicola Bastianello",
    author_email="nicola.bastianello.3@phd.unipd.it",
    description="reg4opt",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nicola-bastianello/reg4opt",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy>=1.19',
        'scipy>=1.5',
        'cvxpy',
        'tvopt',
        'sphinx',
        'sphinx_rtd_theme'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)