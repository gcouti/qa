import sys
from setuptools import setup, find_packages

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

with open('README.md', encoding="utf8") as f:
    readme = f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='qa',
    version='0.0.1',
    description='',
    long_description=readme,
    license=license,
    packages=find_packages(exclude=(
        'examples',
    )),
    install_requires=required
)
