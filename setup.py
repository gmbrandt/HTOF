from setuptools import setup, find_packages

setup(name='htof',
      author=['G. Mirek Brandt, Daniel Michalik'],
      version='0.2.5',
      python_requires='>=3.5',
      packages=find_packages(),
      package_dir={'htof': 'htof'},
      setup_requires=['pytest-runner'],
      install_requires=['numpy>=1.13', 'astropy', 'pandas', 'matplotlib', 'scipy'],
      tests_require=['pytest>=3.5'])
