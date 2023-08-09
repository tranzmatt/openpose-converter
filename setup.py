from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import shutil
import subprocess

# Custom install command to clone repository and copy files
class CustomInstallCommand(install):
    def run(self):
        install.run(self)

        # Clone openpose-library repository
        clone_command = [
            'git', 'clone', 'https://github.com/tranzmatt/openpose-library.git'
        ]
        subprocess.run(clone_command)

        # Copy openpose-library files to the package directory
        source_dir = 'openpose-library'
        target_dir = os.path.join(self.install_lib, 'openpose')

        os.makedirs(target_dir, exist_ok=True)
        for filename in ['body.py', 'model.py', 'util.py']:
            source_file = os.path.join(source_dir, filename)
            target_file = os.path.join(target_dir, filename)
            shutil.copy(source_file, target_file)

        # Clean up cloned repository
        shutil.rmtree(source_dir)

# Setup configuration
setup(
    name='openpose-converter',
    version='0.0.1',
    packages=find_packages(),
    package_data={'openpose': ['*']},
    scripts=['canny-maker-recursive.py', 'canny-maker-rembg.py', 'openpose-converter.py', 'resize768.py', 'cropAndResize768.py', 'cropAndResizeFixed768.py', 'rembg-maker.py' ],
    install_requires=[],
    cmdclass={
        'install': CustomInstallCommand,
    }
)

