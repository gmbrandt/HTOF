"""
Locate files that ship inside the ``htof`` package.

Replaces the deprecated/removed ``pkg_resources.resource_filename`` (setuptools
removed ``pkg_resources`` from the default install on Python 3.12+).  Uses
:mod:`importlib.resources`, which is part of the standard library on every
Python version htof supports (>= 3.10).
"""
import os
from importlib.resources import files as _files

__all__ = ['resource_filename', 'data_path']


def resource_filename(package: str, relative_path: str) -> str:
    """
    Return the absolute filesystem path of ``relative_path`` inside ``package``.

    Drop-in replacement for ``pkg_resources.resource_filename``.  Only valid for
    packages installed as real directories on disk, which is how htof is
    installed (wheel or editable); zip-imported packages are not supported and
    were not supported by the previous implementation in practice either.
    """
    return os.fspath(_files(package).joinpath(relative_path))


def data_path(relative_path: str) -> str:
    """Absolute path of a file under ``htof/data/``."""
    return resource_filename('htof', os.path.join('data', relative_path))
