#!/usr/bin/env python3
"""Metadata for package to allow installation with pip."""

import os
import setuptools

setuptools.setup(
    name="strain_analysis",
    description="Strain analysis study.",
    version="1.0",
    packages=setuptools.find_packages(),
    install_requires=[pip
        "click",
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "opencv-python",
        "scikit-image",
        "tqdm",
        "sklearn"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)