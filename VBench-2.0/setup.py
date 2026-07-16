#!/usr/bin/env python

from setuptools import find_packages, setup

def fetch_readme():
    with open('README-pypi.md', encoding='utf-8') as f:
        text = f.read()
    return text

install_requires = [
    'decord2',
    'gdown',
    'numpy<2.0',
    'opencv-python-headless',
    'Pillow',
    'PyYAML',
    'scenedetect',
    'torch',
    'torchvision',
    'tqdm',
    'transformers',
]
setup(name='vbench2',
      version='0.1.1',
      description='Video generation benchmark',
      long_description=fetch_readme(),
      long_description_content_type='text/markdown',
      project_urls={
          'Source': 'https://github.com/Vchitect/VBench/tree/master/VBench-2.0',
      },
      entry_points={
          'console_scripts': ['vbench2=vbench2.cli.vbench2:main']
      },
      install_requires=install_requires,
      packages=find_packages(),
      include_package_data=True,
      license='Apache Software License 2.0',
)
