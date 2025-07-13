#!/usr/bin/env python3
"""
Setup script for LogGuard ML package.
"""

from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements directly defined here
def get_requirements():
    return [
        "pandas>=1.5.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "plotly>=5.0.0",
        "pyyaml>=6.0",
        "jinja2>=3.0.0",
    ]

# Get version from __version__.py
def get_version():
    version = {}
    with open("logguard_ml/__version__.py") as fp:
        exec(fp.read(), version)
    return version["__version__"]

setup(
    name="logguard-ml",
    version=get_version(),
    author="Saahil Parmar",
    author_email="your.email@example.com",  # Update with actual email
    description="AI-Powered Log Analysis & Anomaly Detection Framework",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/SaahilParmar/LogGuard_ML",
    project_urls={
        "Bug Tracker": "https://github.com/SaahilParmar/LogGuard_ML/issues",
        "Documentation": "https://github.com/SaahilParmar/LogGuard_ML#readme",
        "Source Code": "https://github.com/SaahilParmar/LogGuard_ML",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: System Administrators",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: System :: Logging",
        "Topic :: System :: Monitoring",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=23.0",
            "isort>=5.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "test": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "logguard=logguard_ml.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "logguard_ml": ["config/*.yaml"],
    },
    zip_safe=False,
    keywords="log-analysis, anomaly-detection, machine-learning, monitoring, logs",
)
