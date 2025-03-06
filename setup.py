#!/usr/bin/env python

from io import open
from setuptools import setup

"""
:authors: EgorcaA
:license: MIT License, see LICENSE file
:copyright: (c) 2025 Egor M. Agapov
"""

version = '1.0.6'

with open('pypi_intro.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='qeview',
    version=version,

    author='EgorcaA',
    author_email='agapov.em@phystech.edu',

    description=(
        u'Quantum Espresso Analysis and Visualization Tool '
    ),
    long_description=long_description,
    long_description_content_type='text/markdown',

    url='https://github.com/EgorcaA/QEView',
    download_url='https://github.com/EgorcaA/QEView/archive/main.zip',

    license='MIT License, see LICENSE file',

    packages=['qeview'],
    package_dir = {"": "src"},
    install_requires=['qeschema', 'tqdm', 'matplotlib', 'numpy'],

    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Intended Audience :: End Users/Desktop',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Programming Language :: Python :: Implementation :: CPython',
    ]
)