from setuptools import setup, find_packages, Extension

setup(
    name='mykmeanssp',
    packages=find_packages(),
    ext_modules=[
        Extension('mykmeanssp',['kmeans.c'])
        ]
)