from setuptools import setup, find_packages

setup(name='HTOF',
      author=['G. Mirek Brandt, Daniel Michalik'],
      version='0.1.0',
      python_requires='>=3.5',
      packages=find_packages(),
      setup_requires=['pytest-runner'],
      install_requires=['numpy>=1.13', 'astropy', 'pandas', 'matplotlib'],
      tests_require=['pytest>=3.5'])
