# Optional setup.py for pip installs
from setuptools import setup, find_packages

setup(
    name='ising-ml-criticality',
    version='0.1.0',
    description='ML for Ising model criticality',
    author='Chuene Mosomane',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'h5py',
        'torch',
        'torchvision',
    ],
)
