#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='MEMO',
    version='0.0.0',
    description='Multiple Experts, Multiple Objectives',
    author='Florentine (Tyna) Eloundou',
    author_email='mfe25@cornell.edu',
    url='https://github.com/feloundou/memo',
    install_requires=['pytorch'],
    packages=find_packages(),
)

