#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh.readlines() if line.strip() and not line.startswith("#")]

setup(
    name="cmd",
    version="0.1.0",
    author="Unknown",
    author_email="unknown@example.com",
    description="Genotype-Phenotype Association Analysis based on Mamba2",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/unknown/cmd1023",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "cmd1023-train=train.train:main",
            "cmd1023-predict=model.get_bv:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["README.md", "requirements.txt"],
        "dataset": ["dataset/*"],
    },
)