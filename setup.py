from setuptools import setup, find_packages

setup(
    name='openpose-converter',
    version='0.0.1',
    packages=find_packages(),
    package_data={'openpose': ['*']}
)
