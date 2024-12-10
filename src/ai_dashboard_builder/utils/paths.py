"""This module contains utility functions for paths."""

import os.path as path


def get_root_path() -> str:
    """Get the root path of the project."""
    utils = path.dirname(path.abspath(__file__))
    pkg = path.dirname(utils)
    src = path.dirname(pkg)
    return path.dirname(src)
