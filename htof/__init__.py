"""htof -- fitting Hipparcos and Gaia intermediate astrometric data."""

try:                                     # Python >= 3.8
    from importlib.metadata import version as _version, PackageNotFoundError
    try:
        __version__ = _version("htof")
    except PackageNotFoundError:         # running from a source checkout
        __version__ = "1.2.0"
except ImportError:                      # pragma: no cover
    __version__ = "1.2.0"

__all__ = ["__version__"]
