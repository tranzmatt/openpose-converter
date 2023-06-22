from setuptools import setup, find_packages

setup(
    name='openpose-converter',
    version='0.0.1',
    packages=find_packages(),
    package_data={'openpose': ['*']},
    scripts=['canny-maker-recursive.py', 'openpose-converter.py', 'resize768.py'],
    install_requires=[
        'openpose-library',
    ],
    dependency_links=[
        'https://github.com/tranzmatt/openpose-library/archive/master.zip#egg=dependency-package',
    ],
)

