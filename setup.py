from setuptools import setup, find_packages

setup(name='htof',
      author='G. Mirek Brandt, Daniel Michalik',
      version='0.2.12',
      python_requires='>=3.5',
      packages=find_packages(),
      package_dir={'htof': 'htof'},
      package_data={'banzai_nres': ['data/hip1_flagged.txt', 'data/*.csv', 'data/hip2_dvd_flagged.fits']},
      setup_requires=['pytest-runner'],
      install_requires=['astropy>=2.0', 'pandas>=0.24.0', 'scipy>=1.0.0', 'numpy>=1.16'],
      tests_require=['pytest>=3.5'])
