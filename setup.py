#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

def load_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="CMD",
    version="1.0.0",
    description="Genotype-Phenotype association framework based on Mamba2",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=load_requirements(),
    entry_points={
        "console_scripts": [
            "cmd-train=train.train:main",
            "cmd-predict=model.get_bv:main",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    project_urls={
        "GitHub": "https://github.com/MGBlab-ai4bio/CMD",
        "Hardware": "Optimized for NVIDIA RTX 4090 (CUDA 11.7, Compute Capability 8.9)",
    },
)