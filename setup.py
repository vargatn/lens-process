from setuptools import setup, find_packages

setup(name="proclens",
      description="Data processing preparation and analysis functions for DES Y1 cluster lensing",
      packages=find_packages(),
      install_requires=['numpy', 'kmeans_radec'],
      author="Tamas Norbert Varga",
      author_email="vargat@gmail.com",
      version="0.0")
