#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WGAN-ECANet Installation Script
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='wgan-ecanet',
    version='1.0.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='Radio Signal Modulation Recognition with WGAN and ECA-Net Attention',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/WGAN-ECANet',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.0',
            'black>=21.0',
            'flake8>=3.9',
        ]
    },
)
