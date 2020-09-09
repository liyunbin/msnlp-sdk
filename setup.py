# -*- coding:utf8 -*-
"""Setup for python."""
from setuptools import setup, find_packages


FILE_VERSION = u'version.txt'
with open(FILE_VERSION) as f:
    __version__ = f.readline().splitlines()[0]


def get_requirements():
    """Get python requirement packages."""
    with open('./requirements.txt') as requirements:
        return [line.split('#', 1)[0].strip() for line in requirements
                if line and not line.startswith(('#', '--'))]


setup(
    name='msnlp',
    version=__version__,
    author='dingnan.jin, tong.luo',
    author_email='dingnan.jin@msxf.com, tong.luo@msxf.com',
    url='git@gitlab.msxf.com:ai/ai-experiment/msnlp.git',
    description='',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'msnlp': ['*.*']
    },
    install_requires=get_requirements(),
)
