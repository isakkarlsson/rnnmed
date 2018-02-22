from setuptools import setup

setup(name='rnnmed',
      version='0.1',
      description='Analysis of medical data',
      url='http://github.com/isakkarlss/medrnn',
      author='Isak Karlsson',
      author_email='isak-kar@dsv.su.se',
      license='GPLv3',
      packages=['medrnn'],
      install_requires=[
          'numpy==1.14.0',
          'tensorflow==1.5.0'
      ],
      zip_safe=False)
