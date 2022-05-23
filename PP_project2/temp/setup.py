from setuptools import setup, find_packages, Extension

setup(
    name='module_name',
    packages=find_packages(),
    ext_modules=[
        Extension('module_name',['c_code.c'])
        ]
)