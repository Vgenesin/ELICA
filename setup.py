import os
import time

from setuptools import setup

file_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(file_dir)

setup(
    name="elica",
    version="0.1",
    description="E-mode Likelihood with Cross-correlation Analysis (ELiCA): external Cobaya likelihood package",
    zip_safe=False,  # set to false if you want to easily access bundled package data files
    packages=["elica", "elica.tests"],
    package_data={
        "elica": [
            "*.yaml",
            "*.bibtex",
            "data/*.pickle",
            "data/*.pkl",
        ],
    },
    install_requires=["cobaya", "numpy", "scipy"],
    test_suite="elica.tests",
)
