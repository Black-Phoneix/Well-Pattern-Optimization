"""
Setup script for well_layout_optimization package.
"""

from setuptools import setup, find_packages
import os

# Read requirements
def read_requirements():
    with open('requirements.txt') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read README for long description
def read_readme():
    if os.path.exists('README.md'):
        with open('README.md', 'r', encoding='utf-8') as f:
            return f.read()
    return ""

setup(
    name="well_layout_optimization",
    version="1.0.0",
    description="Geothermal well field optimization with physics-based models",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Factor2-Energy",
    author_email="",
    url="https://github.com/Factor2-Energy/Well_Layout_Optimization",
    packages=find_packages(),
    install_requires=read_requirements(),
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="geothermal optimization well-layout sCO2",
    project_urls={
        "Source": "https://github.com/Factor2-Energy/Well_Layout_Optimization",
        "Bug Reports": "https://github.com/Factor2-Energy/Well_Layout_Optimization/issues",
    },
)
