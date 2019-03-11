#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


setup(
    author="Bruno Calogero",
    author_email='brunocalogero@hotmail.com',
    description="Deployment of machine learning model to AWS lambda / S3",
    install_requires=[],
    license="MIT license",
    include_package_data=True,
    keywords='lambda machine learning',
    name='lambda-ml',
    packages=find_packages(exclude=['*.test']),
    setup_requires = ['pytest-runner'],
    tests_require = ['pytest'],
    version='0.1.0',
    zip_safe=False,
    url="https://github.com/brunocalogero/LowPowerActionRecognition",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
