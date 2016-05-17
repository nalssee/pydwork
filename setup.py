import os
from setuptools import setup

dir = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir, 'requirements.txt')) as f:
    required = f.read().splitlines()


setup(name='pydwork',
      version='0.1',
      description='data work tools',
      url='https://github.com/nalssee/pydwork.git',
      author='nalssee',
      author_email='',
      license='MIT',
      packages=['pydwork'],
      install_requires=required,

      zip_safe=False)
